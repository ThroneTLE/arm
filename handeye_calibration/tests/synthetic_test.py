import cv2,numpy as np
from handeye.calibrate import solve
from handeye.transforms import inv,as_pose,make
def run():
 x=make([.1,-.2,.3],[.04,-.02,.12]); y=make([-.2,.1,.05],[.7,.1,.5]); samples=[]
 for i in range(12):
  b_g=make([.08*i,.05*(i%3),-.06*(i%5)],[.03*i,.02*(i%4),.4+.01*(i%3)]);t_c=inv(y)@b_g@x;samples.append({'base_T_tool':b_g,'target_T_camera':t_c})
 r=solve(samples,'camera_on_tool','tsai')['tool_T_camera']; got=make(r['rotation_vector_rad'],r['translation_m'])
 assert np.allclose(got,x,atol=1e-5),(got,x)
 print('synthetic_handeye=PASS')
