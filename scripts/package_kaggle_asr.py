"""Package the minimal ASR source and fixed test manifest for Kaggle upload."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT_DIR / "kaggle" / "khmer_asr_5k_source_bundle.zip"

SOURCE_FILES = (
    "requirements-asr.txt",
    "scripts/build_asr_manifest.py",
    "scripts/validate_asr_manifest.py",
    "scripts/finetune_asr.py",
    "scripts/run_asr_baseline.py",
    "scripts/score_asr_report.py",
    "khmer_transliteration/__init__.py",
    "khmer_transliteration/asr_audio.py",
    "khmer_transliteration/asr_data.py",
    "khmer_transliteration/asr_decoding.py",
    "khmer_transliteration/asr_inference.py",
    "khmer_transliteration/asr_metrics.py",
    "khmer_transliteration/asr_text.py",
    "khmer_transliteration/paths.py",
    "data/voice_1000/test.csv",
    "data/voice_1000/manifest_summary.json",
    "reports/asr/baseline_test_rescored.json",
    "reports/asr/finetuned_test_clean_rescored.json",
)


def package_bundle(output: Path) -> Path:
    """Write the code and existing fixed-test manifest into one zip file."""
    output = output if output.is_absolute() else ROOT_DIR / output
    missing = [relative for relative in SOURCE_FILES if not (ROOT_DIR / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot package the Kaggle ASR bundle; required files are missing:\n"
            + "\n".join(missing)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for relative in SOURCE_FILES:
            archive.write(ROOT_DIR / relative, arcname=relative)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the one-file source bundle used by the Kaggle ASR notebook."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        bundle = package_bundle(args.output)
    except Exception as exc:
        print(f"Could not create Kaggle ASR bundle: {exc}")
        return 1

    print(f"Created Kaggle upload bundle: {bundle}")
    print(f"Bundle size: {bundle.stat().st_size / (1024 * 1024):.2f} MiB")
    print(f"Included files: {len(SOURCE_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
