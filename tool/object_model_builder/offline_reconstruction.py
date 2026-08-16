#!/usr/bin/env python3
"""Headless reconstruction of a portable RGB-D capture ZIP."""

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .foundationpose_export import export_foundationpose_model
from .mesh_fusion import MeshFusionResult, fuse_session
from .session_archive import (
    ImportedSession,
    create_result_archive,
    extract_capture_archive,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "object_model_builder.yaml"


@dataclass
class OfflineReconstructionResult:
    imported_session: ImportedSession
    fusion_result: MeshFusionResult
    model_directory: Path
    model_obj: Path
    result_zip: Path
    report_path: Path


def _load_config(path: str) -> dict:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported object model builder config schema")
    return config


def _safe_model_name(value: str) -> str:
    safe = "".join(
        character for character in str(value) if character.isalnum() or character in "_-"
    )
    if not safe:
        raise ValueError("model name must contain letters, digits, '-' or '_'")
    return safe


def reconstruct_capture_archive(
    archive_path: str,
    model_name: str,
    config_path: str = str(DEFAULT_CONFIG),
    output_root: Optional[str] = None,
    work_root: Optional[str] = None,
    result_zip: Optional[str] = None,
    keep_extracted: bool = False,
    voxel_length_m: Optional[float] = None,
    sdf_trunc_m: Optional[float] = None,
) -> OfflineReconstructionResult:
    config = _load_config(config_path)
    paths = config["paths"]
    fusion = config["fusion"]
    safe_name = _safe_model_name(model_name)
    output = Path(output_root or paths["mesh_root"]).expanduser().resolve()
    work = Path(
        work_root or (Path(paths["capture_root"]).expanduser() / "offline_imports")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    imported = extract_capture_archive(archive_path, str(work))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_id = "{}_{}".format(safe_name, timestamp)
    job_root = output / job_id
    counter = 1
    while job_root.exists():
        job_id = "{}_{}_{}".format(safe_name, timestamp, counter)
        job_root = output / job_id
        counter += 1
    try:
        job_root.mkdir(parents=True, exist_ok=False)
        fusion_result = fuse_session(
            str(imported.session_path),
            voxel_length_m=float(
                fusion["voxel_length_m"]
                if voxel_length_m is None
                else voxel_length_m
            ),
            sdf_trunc_m=float(
                fusion["sdf_trunc_m"] if sdf_trunc_m is None else sdf_trunc_m
            ),
            maximum_depth_m=float(fusion["maximum_depth_m"]),
            minimum_views=int(fusion["minimum_views"]),
            mask_erosion_pixels=int(fusion["mask_erosion_pixels"]),
            simplify_triangles=int(fusion["simplify_triangles"]),
            workspace_up=fusion["workspace_up"],
            minimum_mask_depth_coverage=float(
                config["capture"]["minimum_mask_depth_coverage"]
            ),
            maximum_object_centroid_shift_m=float(
                config["capture"].get("maximum_object_centroid_shift_m", 0.07)
            ),
        )
        model_obj, validation = export_foundationpose_model(
            fusion_result.mesh,
            str(job_root),
            safe_name,
            fusion_result.workspace_from_object,
            str(Path(archive_path).expanduser().resolve()),
            quality_config=fusion.get("mesh_quality"),
        )
        model_directory = model_obj.parent
        report = {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_capture_zip": str(Path(archive_path).expanduser().resolve()),
            "source_capture_sha256": imported.archive_sha256,
            "source_frame_count": imported.frame_count,
            "views_integrated": fusion_result.views_integrated,
            "voxel_length_m": float(
                fusion["voxel_length_m"]
                if voxel_length_m is None
                else voxel_length_m
            ),
            "sdf_trunc_m": float(
                fusion["sdf_trunc_m"] if sdf_trunc_m is None else sdf_trunc_m
            ),
            "vertex_count": fusion_result.vertex_count,
            "triangle_count": fusion_result.triangle_count,
            "dimensions_m": list(fusion_result.dimensions_m),
            "mesh_valid": validation.valid,
            "capture_quality_gate": {
                "minimum_mask_depth_coverage": float(
                    config["capture"]["minimum_mask_depth_coverage"]
                ),
                "maximum_object_centroid_shift_m": float(
                    config["capture"].get(
                        "maximum_object_centroid_shift_m", 0.07
                    )
                ),
            },
            "mesh_quality_gate": dict(fusion.get("mesh_quality", {})),
            "warnings": list(validation.warnings),
        }
        report_path = model_directory / "reconstruction_report.yaml"
        with open(report_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)
        zip_path = Path(result_zip).expanduser() if result_zip else output / (job_id + ".zip")
        zip_path = create_result_archive(str(model_directory), str(zip_path))
        result = OfflineReconstructionResult(
            imported_session=imported,
            fusion_result=fusion_result,
            model_directory=model_directory,
            model_obj=model_obj,
            result_zip=zip_path,
            report_path=report_path,
        )
    except Exception:
        shutil.rmtree(job_root, ignore_errors=True)
        if not keep_extracted:
            shutil.rmtree(imported.session_path, ignore_errors=True)
        raise
    if not keep_extracted:
        shutil.rmtree(imported.session_path, ignore_errors=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_zip")
    parser.add_argument("--model-name", default="bottle")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    parser.add_argument("--work-root")
    parser.add_argument("--result-zip")
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--voxel-length-m", type=float)
    parser.add_argument("--sdf-trunc-m", type=float)
    args = parser.parse_args(argv)
    result = reconstruct_capture_archive(
        args.capture_zip,
        args.model_name,
        config_path=args.config,
        output_root=args.output_root,
        work_root=args.work_root,
        result_zip=args.result_zip,
        keep_extracted=args.keep_extracted,
        voxel_length_m=args.voxel_length_m,
        sdf_trunc_m=args.sdf_trunc_m,
    )
    print(
        yaml.safe_dump(
            {
                "status": "ok",
                "model_directory": str(result.model_directory),
                "model_obj": str(result.model_obj),
                "result_zip": str(result.result_zip),
                "views_integrated": result.fusion_result.views_integrated,
                "dimensions_m": list(result.fusion_result.dimensions_m),
            },
            sort_keys=False,
            allow_unicode=True,
        ).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
