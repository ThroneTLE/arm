import argparse,json
from pathlib import Path
import yaml
from .adapters import read_er,read_inexbot
from .calibrate import camera_pose,solve
from .transforms import as_pose

def load(path):
    with open(path,encoding='utf-8') as f:return json.load(f)
def main():
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
 c=s.add_parser('capture'); c.add_argument('--robot',choices=['er','inexbot'],required=True); c.add_argument('--robot-pose',required=True); c.add_argument('--camera-pose',required=True); c.add_argument('--samples',required=True)
 q=s.add_parser('solve'); q.add_argument('--config',required=True);q.add_argument('--samples',required=True);q.add_argument('--output',required=True);q.add_argument('--method',choices=['tsai','park','horaud'],default='park')
 s.add_parser('test'); a=p.parse_args()
 if a.cmd=='test':
  from tests.adapter_test import run as adapters; from tests.synthetic_test import run as synthetic; adapters(); synthetic(); return
 if a.cmd=='capture':
  robot=(read_er if a.robot=='er' else read_inexbot)(load(a.robot_pose)); record={'base_T_tool':as_pose(robot),'target_T_camera':as_pose(camera_pose(load(a.camera_pose)))}; Path(a.samples).parent.mkdir(parents=True,exist_ok=True);open(a.samples,'a',encoding='utf-8').write(json.dumps(record)+'\n');print('sample captured');return
 cfg=yaml.safe_load(open(a.config,encoding='utf-8')); rows=[]
 for line in open(a.samples,encoding='utf-8'):
  r=json.loads(line); from .transforms import make; rows.append({'base_T_tool':make(r['base_T_tool']['rotation_vector_rad'],r['base_T_tool']['translation_m']),'target_T_camera':make(r['target_T_camera']['rotation_vector_rad'],r['target_T_camera']['translation_m'])})
 result=solve(rows,cfg['setup'],a.method);Path(a.output).parent.mkdir(parents=True,exist_ok=True);yaml.safe_dump(result,open(a.output,'w',encoding='utf-8'),sort_keys=False);print(yaml.safe_dump(result,sort_keys=False))
if __name__=='__main__':main()
