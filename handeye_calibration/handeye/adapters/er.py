"""ER controller pose adapter: x/y/z mm, roll/pitch/yaw degrees."""
import numpy as np
from ..transforms import make

def read_er(raw):
    required=('x_mm','y_mm','z_mm','roll_deg','pitch_deg','yaw_deg')
    if any(k not in raw for k in required): raise ValueError(f'ER pose requires {required}')
    rpy=np.radians([raw['roll_deg'],raw['pitch_deg'],raw['yaw_deg']])
    cr,cp,cy=np.cos(rpy); sr,sp,sy=np.sin(rpy)
    rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
    rvec,_=__import__('cv2').Rodrigues(rotation)
    return make(rvec, np.array([raw['x_mm'],raw['y_mm'],raw['z_mm']],float)/1000)
