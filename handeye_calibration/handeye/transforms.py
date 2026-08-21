import cv2
import numpy as np

def make(rvec, t):
    out = np.eye(4); out[:3,:3], _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3,1)); out[:3,3] = np.asarray(t, float).reshape(3); return out

def inv(t):
    out=np.eye(4); out[:3,:3]=t[:3,:3].T; out[:3,3]=-out[:3,:3]@t[:3,3]; return out

def quat_matrix(xyzw, t):
    x,y,z,w=np.asarray(xyzw,float); n=np.linalg.norm([x,y,z,w])
    if n < 1e-9: raise ValueError('zero quaternion')
    x,y,z,w=np.array([x,y,z,w])/n
    out=np.eye(4); out[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]; out[:3,3]=t; return out

def as_pose(t):
    rvec,_=cv2.Rodrigues(t[:3,:3]); return {'translation_m':t[:3,3].round(10).tolist(),'rotation_vector_rad':rvec.reshape(3).round(10).tolist()}
