import json
from pathlib import Path
import numpy as np

from handeye.adapters import read_er, read_inexbot

FIXTURES = Path(__file__).parent / "fixtures"


def run():
    er = read_er(json.loads((FIXTURES / "er_pose.json").read_text()))
    inexbot = read_inexbot(json.loads((FIXTURES / "inexbot_pose.json").read_text()))
    assert np.allclose(er[:3, 3], [0.42, -0.12, 0.38])
    assert np.allclose(inexbot[:3, 3], [0.42, -0.12, 0.38])
    assert np.allclose(er[:3, :3] @ er[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.allclose(inexbot[:3, :3] @ inexbot[:3, :3].T, np.eye(3), atol=1e-9)
    print("adapter_coordinates=PASS")
