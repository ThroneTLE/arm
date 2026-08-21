#!/usr/bin/env python3
"""FoundationPose model-free reconstruction through BundleSDF.

The official FoundationPose model-free path does not consume a CAD file. It
trains a Neural Object Field from RGB-D reference views and extracts a mesh
for the normal FoundationPose register/track interface. This module keeps
that vendor dependency lazy so the capture UI and its tests remain usable on
machines without CUDA/Kaolin.
"""

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

from .session_archive import export_foundationpose_reference_directory


@dataclass(frozen=True)
class ModelFreeResult:
    """Files produced by one Neural Object Field reconstruction."""

    job_directory: Path
    reference_directory: Path
    neural_field_directory: Path
    model_obj: Path
    frame_count: int
    elapsed_s: float


def _as_reference_root(path: str, object_id: int = 1) -> Path:
    root = Path(path).expanduser().resolve()
    candidate = root / "ob_{:07d}".format(int(object_id))
    if candidate.is_dir():
        return candidate
    if root.name.startswith("ob_") and root.is_dir():
        return root
    raise FileNotFoundError(
        "FoundationPose reference directory does not contain {}".format(candidate)
    )


def _read_reference(reference_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the exact layout consumed by upstream ``run_nerf.py``."""
    reference_root = Path(reference_root).expanduser().resolve()
    k_path = reference_root / "K.txt"
    if not k_path.is_file():
        raise FileNotFoundError("missing reference camera intrinsics: {}".format(k_path))
    K = np.asarray(np.loadtxt(str(k_path)), dtype=np.float64).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError("reference camera intrinsics are invalid")

    color_files = sorted((reference_root / "rgb").glob("*.png"))
    if not color_files:
        raise ValueError("reference directory contains no RGB images")
    rgbs = []
    depths = []
    masks = []
    cam_in_obs = []
    for color_path in color_files:
        stem = color_path.stem
        depth_path = reference_root / "depth_enhanced" / (stem + ".png")
        mask_path = reference_root / "mask" / (stem + ".png")
        pose_path = reference_root / "cam_in_ob" / (stem + ".txt")
        image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or depth_mm is None or mask is None:
            raise ValueError("reference view {} is incomplete".format(stem))
        if depth_mm.shape != image.shape[:2] or mask.shape != image.shape[:2]:
            raise ValueError("reference view {} has mismatched RGB-D/Mask dimensions".format(stem))
        pose = np.asarray(np.loadtxt(str(pose_path)), dtype=np.float64).reshape(4, 4)
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4):
            raise ValueError("reference view {} has an invalid cam_in_ob pose".format(stem))
        depth = depth_mm.astype(np.float32) * 0.001
        binary = mask > 0
        if not np.any(binary) or not np.any((depth >= 0.001) & binary):
            raise ValueError("reference view {} has no valid masked depth".format(stem))
        # The upstream BundleSDF function consumes RGB arrays. CaptureSession
        # stores OpenCV BGR frames, so convert explicitly after reading the
        # standard PNG bytes.
        rgbs.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        depths.append(depth)
        masks.append(binary)
        cam_in_obs.append(pose)

    return (
        np.asarray(rgbs),
        np.asarray(depths),
        np.asarray(masks),
        np.asarray(cam_in_obs),
        K,
    )


def _load_upstream_runner(foundationpose_root: str):
    """Import upstream BundleSDF without polluting normal module imports."""
    root = Path(foundationpose_root).expanduser().resolve()
    foundationpose = root / "FoundationPose"
    runner_path = foundationpose / "bundlesdf" / "run_nerf.py"
    if not runner_path.is_file():
        raise FileNotFoundError("upstream BundleSDF runner does not exist: {}".format(runner_path))
    # run_nerf.py uses sibling absolute imports (nerf_runner, Utils, ...).
    for entry in (str(foundationpose), str(foundationpose / "bundlesdf"), str(foundationpose / "mycpp" / "build")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location("arm_foundationpose_upstream_run_nerf", str(runner_path))
    if spec is None or spec.loader is None:
        raise ImportError("could not load upstream BundleSDF runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(foundationpose_root: str, config_path: Optional[str]) -> dict:
    if config_path:
        path = Path(config_path).expanduser().resolve()
    else:
        path = Path(foundationpose_root).expanduser().resolve() / "FoundationPose" / "bundlesdf" / "config_ycbv.yml"
    if not path.is_file():
        raise FileNotFoundError("BundleSDF config does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("BundleSDF config root must be a mapping")
    return dict(config)


def _require_model_free_dependencies() -> None:
    """Fail fast with an actionable message before importing vendor code."""
    required = ("torch", "kaolin", "pytorch3d", "nvdiffrast", "open3d")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "FoundationPose Model-free 需要安装依赖：{}。"
            "当前环境仍可拍照/导出 ZIP/运行已有网格，但不能训练 Neural Object Field。".format(
                ", ".join(missing)
            )
        )


def _require_mycuda_extensions(foundationpose_root: str) -> None:
    """Ensure the two local BundleSDF CUDA extensions were compiled."""
    directory = (
        Path(foundationpose_root).expanduser().resolve()
        / "FoundationPose"
        / "bundlesdf"
        / "mycuda"
    )
    common = list(directory.glob("common*.so"))
    gridencoder = list(directory.glob("gridencoder*.so"))
    if not common or not gridencoder:
        raise RuntimeError(
            "FoundationPose Model-free 缺少 mycuda CUDA 扩展（common/gridencoder）。"
            "请先在 foundationpose 环境执行 FoundationPose/bundlesdf/mycuda 的编译，"
            "再重新启动 UI。"
        )


def run_model_free_reconstruction(
    reference_directory: str,
    foundationpose_root: str,
    output_directory: str,
    config_path: Optional[str] = None,
    iterations: Optional[int] = None,
    mesh_resolution_m: Optional[float] = None,
    object_id: int = 1,
) -> ModelFreeResult:
    """Train a Neural Object Field and extract a FoundationPose mesh.

    ``output_directory`` is a new job directory. Existing non-empty output
    directories are rejected so a long GPU reconstruction cannot silently
    overwrite a previous result.
    """
    reference_root = _as_reference_root(reference_directory, object_id=object_id)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("model-free output directory is not empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    rgbs, depths, masks, cam_in_obs, K = _read_reference(reference_root)
    config = _load_config(foundationpose_root, config_path)
    if iterations is not None:
        if int(iterations) < 1:
            raise ValueError("BundleSDF iterations must be positive")
        config["n_step"] = int(iterations)
    if mesh_resolution_m is not None:
        resolution = float(mesh_resolution_m)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("mesh resolution must be positive")
        config["mesh_resolution"] = resolution

    neural_field = output / "nerf"
    model_obj = output / "model" / "model.obj"
    started = time.perf_counter()
    _require_model_free_dependencies()
    _require_mycuda_extensions(foundationpose_root)
    runner = _load_upstream_runner(foundationpose_root)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for BundleSDF model-free reconstruction") from error
    if not torch.cuda.is_available():
        raise RuntimeError("BundleSDF model-free reconstruction requires a CUDA-enabled PyTorch environment")
    model_obj.parent.mkdir(parents=True, exist_ok=True)
    mesh = runner.run_neural_object_field(
        config,
        K,
        rgbs,
        depths,
        masks,
        cam_in_obs,
        save_dir=str(neural_field),
        debug=0,
    )
    if mesh is None or not hasattr(mesh, "export"):
        raise RuntimeError("BundleSDF did not return an exportable mesh")
    mesh.export(str(model_obj))
    metadata = {
        "schema_version": 1,
        "representation": "foundationpose_model_free_neural_object_field",
        "backend": "FoundationPose BundleSDF run_neural_object_field",
        "reference_directory": str(reference_root),
        "frame_count": int(len(rgbs)),
        "units": "meters",
        "mesh_obj": str(model_obj),
        "neural_field_directory": str(neural_field),
        "config": dict(config),
        "elapsed_s": float(time.perf_counter() - started),
    }
    with (output / "model_free_metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
    return ModelFreeResult(
        job_directory=output,
        reference_directory=reference_root,
        neural_field_directory=neural_field,
        model_obj=model_obj,
        frame_count=len(rgbs),
        elapsed_s=float(time.perf_counter() - started),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--foundationpose-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--mesh-resolution-m", type=float)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--session", help="export a CaptureSession to --reference-dir before training")
    args = parser.parse_args(argv)
    if args.session:
        export_foundationpose_reference_directory(
            args.session,
            args.reference_dir,
            object_id=args.object_id,
            object_name=Path(args.session).name,
        )
    result = run_model_free_reconstruction(
        args.reference_dir,
        args.foundationpose_root,
        args.output_dir,
        config_path=args.config,
        iterations=args.iterations,
        mesh_resolution_m=args.mesh_resolution_m,
        object_id=args.object_id,
    )
    print(json.dumps({
        "job_directory": str(result.job_directory),
        "neural_field_directory": str(result.neural_field_directory),
        "model_obj": str(result.model_obj),
        "frame_count": result.frame_count,
        "elapsed_s": result.elapsed_s,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
