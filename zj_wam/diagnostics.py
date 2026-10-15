"""Additional Phase 0/1 diagnostics; original experiment implementation unchanged."""
import numpy as np
from scipy.fft import dct, idct
from zj_wam.adapters import ReplaySimulator


def jerk_energy(points, dt):
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        raise ValueError('Jerk needs at least four timestamped positions')
    jerk = np.diff(points, n=3, axis=0) / dt**3
    return float(np.sum(jerk**2) * dt)


def polynomial_basis(horizon):
    """Fixed quintic family, correction value/first derivative zero at endpoints.

    Per axis: u^2 (1-u)^2 [c0 + c1 (2u-1)]. Orthonormal in normalized
    action coordinates; this represents a correction, not the complete action.
    """
    u = np.linspace(0., 1., horizon)
    envelope = u**2 * (1-u)**2
    temporal = np.column_stack([envelope, envelope * (2*u-1)])
    matrix = np.kron(temporal, np.eye(2))
    q, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    return q[:, singular > singular.max()*1e-10]


def frequency_matched_directions(direction, rng, count=3):
    """Preserve low's per-frequency 2D DCT energy, randomize spatial orientation."""
    coefficients = dct(np.asarray(direction).reshape(-1, 2), axis=0, norm='ortho')
    result = []
    for _ in range(count):
        angles = rng.uniform(-np.pi, np.pi, size=len(coefficients))
        cosine, sine = np.cos(angles), np.sin(angles)
        transformed = np.column_stack([cosine*coefficients[:,0]-sine*coefficients[:,1],
                                       sine*coefficients[:,0]+cosine*coefficients[:,1]])
        result.append(idct(transformed, axis=0, norm='ortho').reshape(-1))
    return result


def temporal_properties(direction):
    d = np.asarray(direction).reshape(-1, 2)
    power = np.sum(dct(d, axis=0, norm='ortho')**2, axis=1)
    return {'frequency_centroid':float(np.arange(len(power))@power/power.sum()),
            'high_frequency_fraction':float(power[(len(power)+1)//2:].sum()/power.sum()),
            'first_difference_energy':float(np.sum(np.diff(d,axis=0)**2))}


class DiagnosticSimulator(ReplaySimulator):
    """Record control/physics timestamps including entry and exit boundaries.

    All trajectories use the same expert suffix of three actions. Prefix replay
    is the original adapter. Instrumentation observes state after physics steps
    without changing integration, contacts, actions, or the source environment.
    """
    def restore(self, capture=False):
        controls, physics, targets = [], [], []
        original_control = self.env.step
        instrumented_space = None
        original_physics = None
        def recording_control(action):
            nonlocal instrumented_space, original_physics
            if self.env.space is not instrumented_space:
                instrumented_space = self.env.space
                original_physics = instrumented_space.step
                def recording_physics(dt):
                    original_physics(dt)
                    physics.append(self.env._get_obs().copy())
                instrumented_space.step = recording_physics
            targets.append(np.asarray(self.env.agent.position) + np.asarray(action)*self.env.action_scale)
            result = original_control(action)
            controls.append(result[0]['state'].copy())
            return result
        self.env.step = recording_control
        try:
            result = super().restore(capture=capture)
        finally:
            self.env.step = original_control
            if instrumented_space is not None:
                instrumented_space.step = original_physics
        self.prefix_control = np.asarray(controls[-3:])
        self.prefix_physics = np.asarray(physics[-3:])
        self.prefix_targets = np.asarray(targets[-3:])
        return result

    def execute_diagnostic(self, actions, suffix, restore=True):
        if restore:
            self.restore()
        physics, targets = [], []
        original_physics, original_control = self.env.space.step, self.env.step
        def recording_physics(dt):
            original_physics(dt)
            physics.append(self.env._get_obs().copy())
        def recording_control(action):
            targets.append(np.asarray(self.env.agent.position) + np.asarray(action)*self.env.action_scale)
            return original_control(action)
        self.env.space.step, self.env.step = recording_physics, recording_control
        try:
            result = super().execute(actions, restore=False)
            result['final_image'] = self.env.render().copy()
            tail = super().execute(suffix, restore=False)
        finally:
            self.env.space.step, self.env.step = original_physics, original_control
        controls = np.concatenate([self.prefix_control, result['states'], tail['states']])
        physical = np.concatenate([self.prefix_physics, np.asarray(physics)])
        command = np.concatenate([self.prefix_targets, np.asarray(targets)])
        raw = np.concatenate([self.actions[self.timestep-3:self.timestep], actions, suffix])
        dt = 1/self.env.control_hz
        result.update({'tail_states':tail['states'], 'tail_contacts':tail['contacts'],
                       'control_trace':controls, 'physics_trace':physical,
                       'command_targets':command,
                       'executed_jerk':jerk_energy(controls[:,:2],dt),
                       'physics_jerk':jerk_energy(physical[:,:2],self.env.dt),
                       'target_jerk':jerk_energy(command,dt),
                       'action_roughness':jerk_energy(raw,dt)})
        return result
