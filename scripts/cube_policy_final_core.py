"""Prespecified nominal phase catalogue for independent Cube policy evaluation."""
import numpy as np

def phase_roots(source,cfg):
    steps=np.arange(15,300,5);ix=steps*25-1;z=source['physics_object_pose'][ix,2];contact=source['physics_robot_object_contacts'][ix]>0;z0=float(source['initial_object_pose'][2]);chosen={'approach':15}
    g=steps[contact&(z<=z0+cfg['grasp_height_above_initial_max_m'])];chosen['grasp']=int(g[0]) if len(g) else None
    t=steps[contact&(z>=z0+cfg['transport_height_above_initial_min_m'])];chosen['transport']=int(t[0]) if len(t) else None
    r=steps[(~contact)&(z<=z0+cfg['grasp_height_above_initial_max_m'])&(steps>(chosen['transport'] if chosen['transport'] is not None else 300))];chosen['release']=int(r[0]) if len(r) else None
    return chosen
