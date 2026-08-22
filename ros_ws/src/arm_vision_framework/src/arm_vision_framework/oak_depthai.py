"""OAK-D Pro DepthAI settings shared by the ROS camera bridge and tests.

The selected profile is intentionally a still-image profile.  The competition
workflow waits for the arm to stop, takes one or two RGB-D observations, and
then starts motion; it does not need a high-FPS video pipeline.  Keeping RGB
and aligned depth at the same 1920 x 1080 size also prevents accidental use of
the 1280 x 800 mono intrinsics with RGB pixels.
"""

from dataclasses import dataclass


OAK_D_PRO_RGB_SENSOR_SIZE = (4032, 3040)
OAK_D_PRO_MONO_SENSOR_SIZE = (1280, 800)
OAK_D_PRO_STEREO_BASELINE_MM = 75.0
RECOMMENDED_PROFILE_NAME = "static_rgbd_1080p_extended"


@dataclass(frozen=True)
class OakDProProfile:
    """Validated DepthAI configuration for a stopped-arm RGB-D observation."""

    color_width: int = 1920
    color_height: int = 1080
    fps: float = 10.0
    mxid: str = ""
    mono_resolution: str = "800p"
    extended_disparity: bool = True
    subpixel: bool = False
    left_right_check: bool = True
    dot_projector_mA: int = 800
    floodlight_mA: int = 0
    focus_mode: str = "device_default"
    manual_focus: int = None
    minimum_depth_m: float = 0.35
    maximum_depth_m: float = 3.0

    def __post_init__(self):
        object.__setattr__(self, "mxid", str(self.mxid or "").strip())
        if (int(self.color_width), int(self.color_height)) != (1920, 1080):
            raise ValueError(
                "OAK-D Pro competition profile must use RGB 1920x1080; "
                "the imported RGB intrinsics must match exactly"
            )
        if not 1.0 <= float(self.fps) <= 30.0:
            raise ValueError("OAK static-observation FPS must be within 1..30")
        if str(self.mono_resolution).lower() != "800p":
            raise ValueError("OAK-D Pro competition profile requires OV9282 800p")
        if bool(self.extended_disparity) and bool(self.subpixel):
            raise ValueError(
                "DepthAI extended disparity and subpixel cannot be enabled together"
            )
        if not 0 <= int(self.dot_projector_mA) <= 1200:
            raise ValueError("OAK dot-projector current must be within 0..1200 mA")
        if not 0 <= int(self.floodlight_mA) <= 1500:
            raise ValueError("OAK floodlight current must be within 0..1500 mA")
        focus_mode = str(self.focus_mode).strip().lower()
        if focus_mode not in ("device_default", "continuous_auto", "manual"):
            raise ValueError(
                "OAK RGB focus_mode must be device_default, continuous_auto, or manual"
            )
        object.__setattr__(self, "focus_mode", focus_mode)
        if focus_mode == "manual":
            if self.manual_focus is None or not 0 <= int(self.manual_focus) <= 255:
                raise ValueError("OAK manual focus requires manual_focus within 0..255")
            object.__setattr__(self, "manual_focus", int(self.manual_focus))
        if not 0.0 < float(self.minimum_depth_m) < float(self.maximum_depth_m):
            raise ValueError("OAK depth range must be positive and ordered")
        if not bool(self.extended_disparity) and float(self.minimum_depth_m) < 0.70:
            raise ValueError(
                "800p standard disparity is specified from about 0.70 m; enable "
                "extended disparity before accepting a closer depth range"
            )

    @property
    def image_size(self):
        return int(self.color_width), int(self.color_height)

    def metadata(self):
        return {
            "profile": RECOMMENDED_PROFILE_NAME,
            "mxid": self.mxid,
            "rgb_sensor_size": list(OAK_D_PRO_RGB_SENSOR_SIZE),
            "mono_sensor_size": list(OAK_D_PRO_MONO_SENSOR_SIZE),
            "stereo_baseline_mm": OAK_D_PRO_STEREO_BASELINE_MM,
            "color_size": list(self.image_size),
            "fps": float(self.fps),
            "mono_resolution": "800p",
            "depth_aligned_to_color": True,
            "extended_disparity": bool(self.extended_disparity),
            "subpixel": bool(self.subpixel),
            "left_right_check": bool(self.left_right_check),
            "focus_mode": self.focus_mode,
            "manual_focus": self.manual_focus,
            "accepted_depth_range_m": [
                float(self.minimum_depth_m), float(self.maximum_depth_m)
            ],
        }


def profile_from_config(camera_config):
    """Build the fixed competition profile from ``system_parameters.yaml``."""

    oak = dict(camera_config.get("oak_d_pro", {}))
    return OakDProProfile(
        color_width=oak.get("color_width", 1920),
        color_height=oak.get("color_height", 1080),
        fps=oak.get("fps", 10.0),
        mxid=oak.get("mxid", ""),
        mono_resolution=oak.get("mono_resolution", "800p"),
        extended_disparity=oak.get("extended_disparity", True),
        subpixel=oak.get("subpixel", False),
        left_right_check=oak.get("left_right_check", True),
        dot_projector_mA=oak.get("dot_projector_mA", 800),
        floodlight_mA=oak.get("floodlight_mA", 0),
        focus_mode=oak.get("focus_mode", "device_default"),
        manual_focus=oak.get("manual_focus"),
        minimum_depth_m=oak.get("minimum_depth_m", 0.35),
        maximum_depth_m=oak.get("maximum_depth_m", 3.0),
    )


__all__ = [
    "OAK_D_PRO_RGB_SENSOR_SIZE", "OAK_D_PRO_MONO_SENSOR_SIZE",
    "OAK_D_PRO_STEREO_BASELINE_MM", "RECOMMENDED_PROFILE_NAME",
    "OakDProProfile", "profile_from_config",
]
