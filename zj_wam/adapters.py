"""Thin adapters around the installed official LeWM and PushT implementations."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import warnings
warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*", category=UserWarning)

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as T
from zj_wam.io import resolve


class ExpertDataset:
    def __init__(self, path):
        from stable_worldmodel.data import HDF5Dataset
        self.dataset = HDF5Dataset(path=resolve(path),
                                  keys_to_load=["action", "state"],
                                  keys_to_cache=["action", "state"])
        self.actions = self.dataset.get_col_data("action")
        self.states = self.dataset.get_col_data("state")
        finite = np.isfinite(self.actions).all(1)
        # Exactly the official evaluator's full-dataset action normalization.
        self.scaler = StandardScaler().fit(self.actions[finite])

    def episode(self, ep):
        start = int(self.dataset.offsets[ep])
        end = start + int(self.dataset.lengths[ep])
        return self.actions[start:end].copy(), self.states[start:end].copy()


class FrozenLeWM:
    def __init__(self, checkpoint, scaler, device="cpu"):
        from stable_worldmodel.wm.utils import load_pretrained
        self.model = load_pretrained(str(resolve(checkpoint))).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.scaler = scaler
        self.transform = T.Compose([
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(size=224),
        ])

    def normalize(self, actions):
        return self.scaler.transform(np.asarray(actions)).astype(np.float32)

    def denormalize(self, actions):
        return self.scaler.inverse_transform(np.asarray(actions).reshape(-1, 2))

    def predictor(self, images, past_actions, action_block=5, representation="final"):
        frames = torch.stack([self.transform(im) for im in images]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb0 = self.model.encode({"pixels": frames})["emb"].detach()
        past = torch.from_numpy(self.normalize(past_actions).reshape(1, -1, action_block * 2)).to(self.device)

        def predict(action):
            actions = action.reshape(1, -1, action_block * 2)
            all_actions = torch.cat([past, actions], 1)
            act_emb = self.model.action_encoder(all_actions)
            emb = emb0
            future = []
            for i in range(actions.shape[1]):
                end = emb0.shape[1] + i
                start = max(0, end - 3)
                z = self.model.predict(emb[:, start:], act_emb[:, start:end])[:, -1:]
                future.append(z)
                emb = torch.cat([emb, z], 1)
            if representation == "final":
                return future[-1].reshape(-1)
            if representation == "all_future":
                return torch.cat(future, 1).reshape(-1) / np.sqrt(len(future))
            raise ValueError(representation)

        return predict, {"frames": frames, "embedding": emb0, "past": past}


class ReplaySimulator:
    """Restore by deterministic full-prefix replay, preserving contact history.

    We deliberately do not treat PushT's seven-number observation as a complete
    snapshot: it omits block velocity and the physics engine's contact cache.
    """
    def __init__(self, actions, recorded_states, timestep, seed, action_block=5, history=3):
        from stable_worldmodel.envs.pusht.env import PushT
        self.env = PushT(render_mode="rgb_array", resolution=224)
        self.actions = actions
        self.recorded_states = recorded_states
        self.timestep = timestep
        self.seed = int(seed)
        self.action_block = action_block
        self.history = history
        self.goal_state = recorded_states[-1].copy()

    def restore(self, capture=False):
        # Official _set_state advances physics. Goal setup can therefore shift
        # the subsequent initial pose via cached collision corrections. Build
        # official geometry, then assign the recorded episode start WITHOUT a
        # physics step. Initial block velocity is unrecorded; zero is the
        # episode-start assumption, audited against the recorded trajectory.
        self.env.reset(seed=self.seed)
        self.env._setup()
        initial = self.recorded_states[0]
        self.env.agent.position = initial[:2].tolist()
        self.env.agent.velocity = initial[5:].tolist()
        self.env.block.angle = float(initial[4])
        self.env.block.position = initial[2:4].tolist()
        self.env.block.velocity = (0., 0.)
        self.env.block.angular_velocity = 0.
        self.env.space.reindex_shapes_for_body(self.env.agent)
        self.env.space.reindex_shapes_for_body(self.env.block)
        self.env._set_goal_state(self.goal_state.copy())
        if not np.allclose(self.env._get_obs(), initial, atol=1e-10, rtol=0):
            raise AssertionError('Episode initial pose advanced during reset')
        self._object_contacts = 0
        def contact(arbiter, space, data):
            self.env._handle_collision(arbiter, space, data)
            bodies = {shape.body for shape in arbiter.shapes}
            if self.env.agent in bodies and self.env.block in bodies:
                self._object_contacts += len(arbiter.contact_point_set.points)
        self.env.space.on_collision(0, 0, post_solve=contact)
        wanted = set(self.timestep - self.action_block * k for k in range(self.history))
        images = []
        for t in range(self.timestep + 1):
            if capture and t in wanted:
                images.append(self.env.render().copy())
            if t < self.timestep:
                self.env.step(self.actions[t])
        state = self.env._get_obs().copy()
        self.restored_state = state
        if capture:
            return images, self.actions[self.timestep - (self.history - 1) * self.action_block:self.timestep]
        return state

    def contact_gap(self):
        center = self.env.agent.position
        radius = next(iter(self.env.agent.shapes)).radius
        return min(shape.point_query(center).distance - radius for shape in self.env.block.shapes)

    def coverage(self):
        # Object-only target coverage, separate from official seven-state distance.
        import pymunk
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        goal = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        goal.center_of_gravity = self.env.block.center_of_gravity
        goal.angle = float(self.goal_state[4])
        goal.position = tuple(self.goal_state[2:4])
        actual_polygons, goal_polygons = [], []
        for shape in self.env.block.shapes:
            vertices = shape.get_vertices()
            actual_polygons.append(Polygon([self.env.block.local_to_world(v) for v in vertices]))
            goal_polygons.append(Polygon([goal.local_to_world(v) for v in vertices]))
        actual, target = unary_union(actual_polygons), unary_union(goal_polygons)
        return float(actual.intersection(target).area / target.area)

    def execute(self, actions, restore=True):
        if restore:
            self.restore()
        states, contacts = [], []
        gap = self.contact_gap()
        for action in actions:
            self._object_contacts = 0
            obs, reward, terminated, truncated, info = self.env.step(action)
            states.append(obs["state"].copy())
            # Count all internal physics substeps, excluding wall collisions.
            contacts.append(self._object_contacts)
        state = states[-1]
        success, distance = self.env.eval_state(self.goal_state, state)
        return {"states": np.asarray(states), "contacts": np.asarray(contacts),
                "initial_gap": float(gap), "task_score": self.coverage(),
                "official_state_distance": float(distance),
                "task_success": bool(success)}

    def close(self):
        self.env.close()
