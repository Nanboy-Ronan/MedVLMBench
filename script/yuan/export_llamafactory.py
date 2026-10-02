#!/usr/bin/env python3
"""Merge a multimodal LoRA adapter with LLaMA-Factory and export a legacy image-processor file.

Recent Transformers versions can place the image processor inside
``processor_config.json``.  The MedVLMBench evaluation environment currently
expects the standalone ``preprocessor_config.json`` used by older versions.
This wrapper runs the normal LLaMA-Factory export and then saves standalone image/video configurations and removes their nested
copies, which older loaders can mistake for processor objects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml
from transformers import AutoImageProcessor, AutoProcessor


def load_config(config_path: Path) -> tuple[str, Path, bool]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_path = config.get("model_name_or_path")
    export_dir = config.get("export_dir")
    if not model_path or not export_dir:
        raise ValueError(f"{config_path} must define model_name_or_path and export_dir")

    return str(model_path), Path(export_dir), bool(config.get("trust_remote_code", False))


def export_image_processor(model_path: str, export_dir: Path, trust_remote_code: bool) -> None:
    if not export_dir.is_dir():
        raise FileNotFoundError(f"Merged checkpoint directory does not exist: {export_dir}")

    processor_config_path = export_dir / "processor_config.json"
    processor_config = {}
    if processor_config_path.is_file():
        processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
        if not isinstance(processor_config, dict):
            raise ValueError(f"Expected a JSON object: {processor_config_path}")

    # Keep the exported settings (including any training-time image limits).
    # Fall back to the base model only for components absent from the export.
    base_processor = None
    def get_base_processor():
        nonlocal base_processor
        if base_processor is None:
            base_processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        return base_processor

    cleaned_config = dict(processor_config)
    for name, filename in (
        ("image_processor", "preprocessor_config.json"),
        ("video_processor", "video_preprocessor_config.json"),
    ):
        target = export_dir / filename
        nested = processor_config.get(name)
        if isinstance(nested, dict):
            if not nested:
                raise ValueError(f"Empty {name} in {processor_config_path}")
            target.write_text(json.dumps(nested, indent=2) + "\n", encoding="utf-8")
            cleaned_config.pop(name)
        elif nested is not None:
            raise ValueError(f"Unsupported {name} configuration in {processor_config_path}")
        elif not target.is_file():
            component = getattr(get_base_processor(), name, None)
            if component is not None:
                component.save_pretrained(export_dir)
                if not target.is_file():
                    raise RuntimeError(f"{name} did not create {target}")
            elif name == "image_processor":
                raise TypeError("Base processor does not expose image_processor")
        if target.is_file():
            print(f"Standalone {name}: {target}")

    if cleaned_config != processor_config:
        backup = export_dir / "processor_config.json.before_legacy_export.bak"
        if not backup.exists():
            backup.write_bytes(processor_config_path.read_bytes())
        # Retain processor_class, auto_map, chat template and other metadata.
        processor_config_path.write_text(
            json.dumps(cleaned_config, indent=2) + "\n", encoding="utf-8"
        )

    preprocessor_path = export_dir / "preprocessor_config.json"
    if not preprocessor_path.is_file():
        raise RuntimeError(f"Image processor did not create {preprocessor_path}")

    with preprocessor_path.open("r", encoding="utf-8") as handle:
        preprocessor_config = json.load(handle)
    if not isinstance(preprocessor_config, dict) or not preprocessor_config:
        raise RuntimeError(f"Invalid image processor configuration: {preprocessor_path}")

    # Validate the standalone legacy file explicitly. AutoProcessor under a
    # newer Transformers version may otherwise succeed from processor_config.
    AutoImageProcessor.from_pretrained(
        preprocessor_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    # Check the whole processor as well as the standalone image component.
    # Final compatibility still needs a processor-only check in the eval env.
    AutoProcessor.from_pretrained(
        export_dir, trust_remote_code=trust_remote_code, local_files_only=True
    )
    print(f"LLaMA-Factory export complete: {export_dir}")
    print(f"Legacy image processor saved: {preprocessor_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to a LLaMA-Factory merge YAML")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Export legacy image/video processor configs without rerunning an existing merge",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    model_path, export_dir, trust_remote_code = load_config(config_path)

    if not args.finalize_only:
        subprocess.run(["llamafactory-cli", "export", str(config_path)], check=True)

    export_image_processor(model_path, export_dir, trust_remote_code)


if __name__ == "__main__":
    main()
