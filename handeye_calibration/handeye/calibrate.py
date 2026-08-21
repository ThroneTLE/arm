import cv2
import numpy as np
from .transforms import inv, make, as_pose

METHOD={'tsai':cv2.CALIB_HAND_EYE_TSAI,'park':cv2.CALIB_HAND_EYE_PARK,'horaud':cv2.CALIB_HAND_EYE_HORAUD}

def camera_pose(raw):
    """Input interface for external vision: {rotation_vector_rad:[3], translation_m:[3]}."""
    return make(raw['rotation_vector_rad'],raw['translation_m'])

def solve(samples, setup, method='park'):
    if len(samples)<10: raise ValueError('need at least 10 samples')
    b_g=[s['base_T_tool'] for s in samples]; t_c=[s['target_T_camera'] for s in samples]
    if setup=='camera_on_tool':
        candidates=[]
        for a in (b_g,[inv(x) for x in b_g]):
         for b in (t_c,[inv(x) for x in t_c]):
          result=cv2.calibrateHandEye([x[:3,:3] for x in a],[x[:3,3] for x in a],[x[:3,:3] for x in b],[x[:3,3] for x in b],method=METHOD[method]); guess=np.eye(4); guess[:3,:3]=result[0]; guess[:3,3]=result[1].reshape(3); candidates.extend((guess,inv(guess)))
        def error(x):
         targets=[g@x@inv(c) for g,c in zip(b_g,t_c)]; ref=targets[0]; return sum(np.linalg.norm(z[:3,3]-ref[:3,3])+np.linalg.norm(cv2.Rodrigues(ref[:3,:3].T@z[:3,:3])[0]) for z in targets)
        matrix=min(candidates,key=error); label='tool_T_camera'
    elif setup=='camera_fixed':
        a=b_g; b=[inv(x) for x in t_c]; c_b=cv2.calibrateHandEye([x[:3,:3] for x in a],[x[:3,3] for x in a],[x[:3,:3] for x in b],[x[:3,3] for x in b],method=METHOD[method]); matrix=np.eye(4); matrix[:3,:3]=c_b[0]; matrix[:3,3]=c_b[1].reshape(3); matrix=inv(matrix); label='base_T_camera'
    else: raise ValueError('setup must be camera_on_tool or camera_fixed')
    return {'setup':setup,'method':method,'sample_count':len(samples),label:as_pose(matrix)}
