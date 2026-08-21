# Gemini FoundationPose Debug UI

Launch the debug workbench with:

```bash
cd /home/throne/workspaces/arm
./tool/object_model_builder/run_gemini_foundationpose_ui.sh
```

The workbench starts the Orbbec Gemini Max driver, checks the configured serial
number, and reads `/camera/color/camera_info` from the connected camera. That
live factory calibration is stored under
`/home/throne/workspaces/arm_data/foundationpose_gemini_debug/`; it does not
reuse the Astra Pro calibration.

The current `SV1301S_U3` profile keeps the stable RGB/depth pair at RGB
`640x480@30` and aligned depth `640x400@30`, while requests IR
`1280x800@30` for the higher-resolution infrared preview. The displayed metric
bar shows the actual dimensions delivered by the driver. Do not force
`1280x800` depth on this firmware unless a fresh profile query confirms that
the device exposes that depth mode; an unsupported all-high-resolution request
can reset the USB depth interface.

Choose any Ultralytics *segmentation* `.pt` file from the YOLO field. Leave the
target-class field empty to inspect all detected instances, or enter the desired
class name once the model labels are known.

The Model-free button follows the official FoundationPose route. Place the
object between AprilTag ID0 (left) and ID1 (right), keep the object and the
tags rigidly fixed, then move **only the camera** around the object and take 16
accepted RGB-D reference views. This Gemini profile uses both Tags for every
camera pose. The tags use the `DICT_APRILTAG_25h9` family. The Tag size is 75 mm; the black-frame bottom-right corners of ID0
and ID1 are configured 150 mm apart, which means their top-left origins are also
150 mm apart because both tags have the same yaw and size. Keep both Tags in
view during capture. The button exports
the upstream reference layout, calls
`FoundationPose/bundlesdf/run_nerf.py::run_neural_object_field`, trains a Neural
Object Field, and extracts `model/model.obj`. Select that generated OBJ and
enable live FoundationPose. TSDF fusion remains available as a quick geometry
preview, not as the Model-free representation itself.

The live position panel reports `camera_from_object` directly:

```text
X: object right/left of the camera, in meters
Y: object below/above the camera, in meters
Z: object forward from the camera, in meters
distance: Euclidean camera-to-object-origin distance
```

These are diagnostic camera-relative values only. This tool does not command
the arm or execute a grasp. The model-free step requires CUDA, PyTorch3D,
nvdiffrast and Kaolin; the UI does not install them automatically.
