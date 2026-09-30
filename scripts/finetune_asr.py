"""Fine-tune the Khmer XLS-R CTC model on the prepared ASR manifests.

The manifests contain metadata and source row indexes rather than local audio.
This script first materializes the selected rows from Hugging Face as WAV files,
then trains on the train/validation splits. The test split is materialized only
for later evaluation and is never passed to Trainer during fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from khmer_transliteration.asr_text import normalize_asr_text
from scripts.run_asr_baseline import (
    TARGET_SAMPLE_RATE,
    decode_and_resample,
    stream_selected_audio,
)


DEFAULT_MODEL = "vitouphy/wav2vec2-xls-r-300m-khmer"
DEFAULT_MANIFEST_DIR = ROOT_DIR / "data" / "voice_1000"
DEFAULT_MATERIALIZED_DIR = DEFAULT_MANIFEST_DIR / "materialized_training"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "khmer_xlsr_ddd_1000"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def read_manifest_rows(manifest_dir: Path, split: str) -> list[dict[str, Any]]:
    path = manifest_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            try:
                row["source_row_index"] = int(row["source_row_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid source_row_index in {path}: {row}") from exc
            row["manifest_split"] = split
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def load_stream_configuration(manifest_dir: Path) -> dict[str, Any]:
    summary_path = manifest_dir / "manifest_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing manifest summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "dataset": summary["dataset"],
        "source_split": summary.get("source_split", "train"),
        "revision": summary.get("revision"),
        "seed": int(summary["seed"]),
        "shuffle_buffer": int(summary["shuffle_buffer"]),
    }


def materialize_dataset(manifest_dir: Path, output_dir: Path) -> None:
    """Fetch manifest audio once and write local WAV files plus JSONL manifests."""
    configuration = load_stream_configuration(manifest_dir)
    all_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        all_rows.extend(read_manifest_rows(manifest_dir, split))

    selected_rows = stream_selected_audio(configuration, all_rows)
    audio_root = output_dir / "audio"
    manifest_root = output_dir / "manifests"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    output_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for row in selected_rows:
        split = row["manifest_split"]
        source_index = int(row["source_row_index"])
        split_audio_dir = audio_root / split
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = split_audio_dir / f"{split}_{source_index:06d}.wav"
        audio_path.write_bytes(row["audio_bytes"])
        output_rows[split].append(
            {
                "audio": str(audio_path),
                "transcript": normalize_asr_text(
                    row.get("normalized_transcript") or row.get("transcript") or ""
                ),
                "source_row_index": source_index,
            }
        )

    for split, rows in output_rows.items():
        path = manifest_root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as manifest_file:
            for row in rows:
                manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": configuration["dataset"],
        "source_split": configuration["source_split"],
        "seed": configuration["seed"],
        "shuffle_buffer": configuration["shuffle_buffer"],
        "counts": {split: len(rows) for split, rows in output_rows.items()},
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "note": "Test files are materialized for evaluation but excluded from training.",
    }
    (output_dir / "materialization_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Materialized audio to: {output_dir}")
    print(f"Counts: {metadata['counts']}")


def load_processor(model_id: str) -> Any:
    from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor

    # Use greedy CTC components. This avoids the optional KenLM processor
    # published with the checkpoint and keeps the training environment simple.
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


def load_datasets(materialized_dir: Path, processor: Any) -> Any:
    from datasets import load_dataset

    manifest_dir = materialized_dir / "manifests"
    data_files = {
        "train": str(manifest_dir / "train.jsonl"),
        "validation": str(manifest_dir / "validation.jsonl"),
    }
    dataset_dict = load_dataset("json", data_files=data_files)

    def prepare_example(example: dict[str, Any]) -> dict[str, Any]:
        audio_bytes = Path(example["audio"]).read_bytes()
        samples, _, _, _ = decode_and_resample(audio_bytes)
        inputs = processor(
            samples,
            sampling_rate=TARGET_SAMPLE_RATE,
        )
        # Recent Transformers versions removed as_target_processor(). The
        # tokenizer is the target-side processor for CTC labels, so calling it
        # directly works across both older and newer versions.
        labels = processor.tokenizer(example["transcript"]).input_ids
        return {
            "input_values": inputs.input_values[0],
            "input_length": len(inputs.input_values[0]),
            "labels": labels,
        }

    columns = dataset_dict["train"].column_names
    return dataset_dict.map(
        prepare_example,
        remove_columns=columns,
        desc="Preparing audio and CTC labels",
    )


class DataCollatorCTCWithPadding:
    """Pad input waveforms and target token sequences independently."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return batch


def make_compute_metrics(processor: Any) -> Any:
    import numpy as np
    from jiwer import cer, wer

    def compute_metrics(prediction_output: Any) -> dict[str, float]:
        logits = prediction_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted_ids = np.argmax(logits, axis=-1)
        label_ids = prediction_output.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.batch_decode(predicted_ids, group_tokens=True)
        references = processor.batch_decode(label_ids, group_tokens=False)
        predictions = [normalize_asr_text(text) for text in predictions]
        references = [normalize_asr_text(text) for text in references]
        return {
            "cer": float(cer(references, predictions)),
            "wer": float(wer(references, predictions)),
            "sentence_accuracy": float(
                sum(reference == prediction for reference, prediction in zip(references, predictions))
                / max(1, len(references))
            ),
        }

    return compute_metrics


def build_training_arguments(output_dir: Path, stage: str) -> Any:
    import torch
    from transformers import TrainingArguments

    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    )
    common: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "logging_steps": 1,
        "report_to": [],
        "fp16": bool(torch.cuda.is_available()),
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }
    if "warmup_ratio" in parameter_names:
        common["warmup_ratio"] = 0.1
    elif "warmup_steps" in parameter_names:
        # Older Transformers releases do not provide warmup_ratio. Keeping
        # this at zero preserves compatibility without changing the dataset.
        common["warmup_steps"] = 0
    if stage == "smoke":
        common.update(
            {
                "num_train_epochs": 1,
                "max_steps": 5,
                "logging_strategy": "steps",
                "save_strategy": "no",
            }
        )
        common[evaluation_key] = "steps"
        common["eval_steps"] = 5
    else:
        common.update(
            {
                "num_train_epochs": 3,
                "max_steps": -1,
                "logging_strategy": "steps",
                "save_strategy": "epoch",
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "cer",
                "greater_is_better": False,
                "gradient_checkpointing": True,
                "group_by_length": True,
                "length_column_name": "input_length",
            }
        )
        common[evaluation_key] = "epoch"
    unsupported = sorted(set(common) - set(parameter_names))
    if unsupported:
        print(f"Skipping unsupported TrainingArguments options: {unsupported}")
    supported_common = {
        name: value for name, value in common.items() if name in parameter_names
    }
    return TrainingArguments(**supported_common)


def train_model(
    materialized_dir: Path,
    output_dir: Path,
    model_id: str,
    stage: str,
) -> None:
    from transformers import AutoModelForCTC, Trainer

    processor = load_processor(model_id)
    dataset_dict = load_datasets(materialized_dir, processor)
    if stage == "smoke":
        dataset_dict["train"] = dataset_dict["train"].select(
            range(min(8, len(dataset_dict["train"])))
        )
        dataset_dict["validation"] = dataset_dict["validation"].select(
            range(min(4, len(dataset_dict["validation"])))
        )

    model = AutoModelForCTC.from_pretrained(model_id)
    model.config.ctc_loss_reduction = "mean"
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = build_training_arguments(output_dir, stage)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset_dict["train"],
        "eval_dataset": dataset_dict["validation"],
        "data_collator": DataCollatorCTCWithPadding(processor),
        "compute_metrics": make_compute_metrics(processor),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor

    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    evaluation = trainer.evaluate()
    summary = {
        "stage": stage,
        "model": model_id,
        "train_rows": len(dataset_dict["train"]),
        "validation_rows": len(dataset_dict["validation"]),
        "train_metrics": train_result.metrics,
        "validation_metrics": evaluation,
        "note": "The test split was not used during fine-tuning.",
    }
    (output_dir / f"{stage}_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Khmer XLS-R on ASR manifests.")
    parser.add_argument("--stage", choices=("materialize", "smoke", "full"), required=True)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--materialized-dir", type=Path, default=DEFAULT_MATERIALIZED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_dir = resolve_path(args.manifest_dir)
    materialized_dir = resolve_path(args.materialized_dir)
    output_dir = resolve_path(args.output_dir)
    try:
        if args.stage == "materialize":
            materialize_dataset(manifest_dir, materialized_dir)
        else:
            if not (materialized_dir / "manifests" / "train.jsonl").exists():
                raise FileNotFoundError(
                    "Training manifests are missing. Run with --stage materialize first."
                )
            train_model(materialized_dir, output_dir, args.model, args.stage)
    except Exception as exc:
        print(f"ASR fine-tuning failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
