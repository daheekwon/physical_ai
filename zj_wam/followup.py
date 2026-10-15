"""Dense counterfactual instrumentation and controller-aware refinement.

Future object traces are available only to execution evaluation, never to the
controller optimizer. New protocol; original diagnostic artifacts are unchanged.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from zj_wam.adapters import ReplaySimulator
from zj_wam.diagnostics import jerk_energy, polynomial_basis


def angular_error(a, b):
    return np.abs(np.arctan2(np.sin(a-b), np.cos(a-b)))


class DenseSimulator(ReplaySimulator):
    def full_observation(self):
        return np.r_[self.env._get_obs(), np.asarray(self.env.block.velocity),
                     self.env.block.angular_velocity]

    def restore(self, capture=False):
        physics, controls = [], []
        original_control = self.env.step
        current_space, original_physics = None, None
        def control(action):
            nonlocal current_space, original_physics
            if self.env.space is not current_space:
                current_space = self.env.space
                original_physics = current_space.step
                def step(dt):
                    original_physics(dt)
                    physics.append(self.full_observation())
                current_space.step = step
            result = original_control(action)
            controls.append(self.full_observation())
            return result
        self.env.step = control
        try:
            result = super().restore(capture=capture)
        finally:
            self.env.step = original_control
            if current_space is not None:
                current_space.step = original_physics
        self.prefix_physics = np.asarray(physics[-30:])
        self.prefix_control = np.asarray(controls[-3:])
        # Proprioception at the same three historical times as LeWM images.
        # Full object fields are retained for audit only; visual selectors strip them.
        if self.timestep >= 11:
            self.context_control = np.asarray([controls[self.timestep-11], controls[self.timestep-6], controls[self.timestep-1]])
        elif self.timestep == 10:
            initial = np.r_[self.recorded_states[0], 0., 0., 0.]
            self.context_control = np.asarray([initial, controls[4], controls[9]])
        self.root_full_state = self.full_observation()
        return result

    def execute_dense(self, actions, suffix, restore=True, capture_endpoints=False):
        if restore:
            self.restore()
        if len(self.prefix_physics) != 30:
            raise ValueError('Need three complete prefix control intervals')
        actions = np.asarray(actions)
        h = len(actions)
        all_actions = np.concatenate([actions, suffix])
        physical, contacts, controls, endpoints = [], [], [], []
        original = self.env.space.step
        def step(dt):
            before = self._object_contacts
            original(dt)
            physical.append(self.full_observation())
            contacts.append(self._object_contacts-before)
        self.env.space.step = step
        initial_gap = self.contact_gap()
        coverage_t = None
        try:
            for i, action in enumerate(all_actions):
                self._object_contacts = 0
                self.env.step(action)
                controls.append(self.full_observation())
                if capture_endpoints and i < h and (i+1) % 5 == 0:
                    endpoints.append(self.env.render().copy())
                if i+1 == h:
                    coverage_t = self.coverage()
            coverage_suffix = self.coverage()
            success, _ = self.env.eval_state(self.goal_state, controls[h-1][:7])
        finally:
            self.env.space.step = original
        physical = np.asarray(physical)
        control = np.asarray(controls)
        physics_trace = np.concatenate([self.prefix_physics, physical])
        control_trace = np.concatenate([self.prefix_control, control])
        return {
            'physics': physical, 'control': control, 'physics_trace': physics_trace,
            'control_trace': control_trace, 'physics_contacts': np.asarray(contacts),
            'coverage_t': float(coverage_t), 'coverage_suffix': float(coverage_suffix),
            'physics_jerk': jerk_energy(physics_trace[:,:2], self.env.dt),
            'control_jerk': jerk_energy(control_trace[:,:2], 1/self.env.control_hz),
            'initial_gap': float(initial_gap), 'nominal_success': bool(success),
            'endpoint_images': np.asarray(endpoints), 'root_full_state': self.root_full_state.copy(),
        }


def dense_metrics(result, reference, tolerances):
    p = np.linalg.norm(result['physics'][:,2:4]-reference['physics'][:,2:4],axis=1)
    r = angular_error(result['physics'][:,4], reference['physics'][:,4])
    c = max(0., reference['coverage_t']-result['coverage_t'])
    cs = max(0., reference['coverage_suffix']-result['coverage_suffix'])
    e = max(p.max()/tolerances['position'], r.max()/tolerances['rotation'],
            c/tolerances['coverage_loss'],cs/tolerances['coverage_loss'])
    nominal = reference['physics_jerk']
    gain = (nominal-result['physics_jerk'])/nominal if nominal>1e-8 else 0.
    cp=np.linalg.norm(result['control'][:,2:4]-reference['control'][:,2:4],axis=1)
    cr=angular_error(result['control'][:,4], reference['control'][:,4])
    pusher=np.linalg.norm(result['physics'][:,:2]-reference['physics'][:,:2],axis=1)
    return {'object_error':float(e),'preserved':bool(e<=1),
            'position_max':float(p.max()),'rotation_max':float(r.max()),
            'position_control_max':float(cp.max()),'rotation_control_max':float(cr.max()),
            'coverage_loss':float(c),'coverage_suffix_loss':float(cs),
            'jerk_gain':float(gain),'physics_jerk':result['physics_jerk'],
            'control_jerk':result['control_jerk'],
            'pusher_rms':float(np.sqrt(np.mean(pusher**2))),
            'pusher_max':float(pusher.max()),
            'contact_changed_substeps':int(np.sum((result['physics_contacts']>0)!=(reference['physics_contacts']>0))),
            'terminal_velocity_delta':float(np.linalg.norm(result['physics'][-1,7:9]-reference['physics'][-1,7:9])),
            'terminal_angular_velocity_delta':float(abs(result['physics'][-1,9]-reference['physics'][-1,9]))}


def controller_rollout(actions, position, velocity, kp=100., kv=20., dt=.01,
                       substeps=10, action_scale=100.):
    """Known kinematic-pusher PD response; no object state is consumed."""
    q=np.asarray(position,dtype=float).copy()
    v=np.asarray(velocity,dtype=float).copy()
    points=[]
    for action in np.asarray(actions,dtype=float):
        target=q+action_scale*action
        for _ in range(substeps):
            v+=(kp*(target-q)-kv*v)*dt
            q+=v*dt
            points.append(q.copy())
    return np.asarray(points)


def controller_design(nominal, suffix, root, prefix, scale, basis=None):
    """Linear response of 100Hz executed jerk to normalized polynomial coefficients."""
    h=len(nominal)
    q=polynomial_basis(h) if basis is None else basis
    base=controller_rollout(np.concatenate([nominal,suffix]),root[:2],root[5:7])
    matrices=[]
    for column in q.T:
        delta=column.reshape(h,2)*np.asarray(scale)
        moved=controller_rollout(np.concatenate([nominal+delta,suffix]),root[:2],root[5:7])
        matrices.append(moved-base)
    response=np.stack(matrices,axis=-1)
    full=np.concatenate([prefix[:,:2],base])
    extended=np.concatenate([np.zeros((len(prefix),2,q.shape[1])),response])
    j0=np.diff(full,n=3,axis=0).reshape(-1)/.01**3
    design=np.diff(extended,n=3,axis=0).reshape(-1,q.shape[1])/.01**3
    return q,base,response,j0,design


def optimize_controller(nominal, suffix, root, prefix, scale, radius, objective='executed', past_actions=None, basis=None):
    q,base,response,j0,design=controller_design(nominal,suffix,root,prefix,scale,basis=basis)
    raw_map=q*np.tile(scale,len(nominal))[:,None]
    if objective=='raw':
        if past_actions is None:raise ValueError('Raw roughness requires action prefix')
        full=np.concatenate([past_actions,nominal,suffix])
        matrix=np.concatenate([np.zeros((len(past_actions)*2,q.shape[1])),raw_map,
                               np.zeros((len(suffix)*2,q.shape[1]))])
        j0=np.diff(full,n=3,axis=0).reshape(-1)
        design=np.diff(matrix.reshape(len(full),2,-1),n=3,axis=0).reshape(-1,q.shape[1])
    norm=max(float(j0@j0),1.)
    def value(z):
        error=j0+design@z
        return float(error@error/norm)
    def gradient(z):return 2*design.T@(j0+design@z)/norm
    limits=LinearConstraint(raw_map,-1.-nominal.reshape(-1),1.-nominal.reshape(-1))
    ball={'type':'ineq','fun':lambda z:radius**2-z@z,'jac':lambda z:-2*z}
    solution=minimize(value,np.zeros(q.shape[1]),jac=gradient,method='SLSQP',
                      constraints=[limits,ball],options={'ftol':1e-10,'maxiter':200})
    z=solution.x
    raw=nominal+(raw_map@z).reshape(-1,2)
    valid=bool(solution.success and np.linalg.norm(z)<=radius+1e-6 and np.max(np.abs(raw))<=1+1e-6)
    if not valid:raw=nominal.copy();z=np.zeros(q.shape[1])
    return raw,{'solver_success':bool(solution.success),'feasible':valid,
                'message':str(solution.message),'iterations':int(solution.nit),
                'coefficient':z,'objective_ratio':value(z),'radius':float(radius)}
