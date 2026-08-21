"""iNeXBot controller pose adapter: metres plus XYZW quaternion."""
from ..transforms import quat_matrix

def read_inexbot(raw):
    required=('x_m','y_m','z_m','qx','qy','qz','qw')
    if any(k not in raw for k in required): raise ValueError(f'iNeXBot pose requires {required}')
    return quat_matrix([raw['qx'],raw['qy'],raw['qz'],raw['qw']], [raw['x_m'],raw['y_m'],raw['z_m']])
