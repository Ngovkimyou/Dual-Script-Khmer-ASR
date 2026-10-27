"""Reusable Khmer XLS-R inference service for scripts and the web API."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from khmer_transliteration.asr_audio import (
    TARGET_SAMPLE_RATE,
    decode_and_resample,
)
from khmer_transliteration.asr_decoding import ctc_prefix_beam_search
from khmer_transliteration.asr_text import normalize_asr_text
from khmer_transliteration.paths import ASR_MODEL_DIR, LEGACY_ASR_MODEL_DIR


DEFAULT_MODEL_ID = "vitouphy/wav2vec2-xls-r-300m-khmer"


def resolve_model_source(source: str | Path | None = None) -> str:
    """Choose an explicit model path, environment path, local model, or HF ID."""
    if source:
        return str(source)
    configured = os.getenv("KHMER_ASR_MODEL_PATH")
    if configured:
        return configured
    if ASR_MODEL_DIR.exists():
        return str(ASR_MODEL_DIR)
    if LEGACY_ASR_MODEL_DIR.exists():
        return str(LEGACY_ASR_MODEL_DIR)
    return DEFAULT_MODEL_ID


def resolve_device(requested: str = "auto") -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for ASR inference. Install the local ASR dependencies."
        ) from exc

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no GPU is available.")
    return torch.device(requested)


def _resolve_vocab_file(model_id: str | Path) -> Path:
    """Locate the vocabulary file for a local checkpoint or Hub model."""
    model_path = Path(str(model_id)).expanduser()
    if model_path.exists():
        vocab_file = model_path / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(f"Missing tokenizer vocabulary: {vocab_file}")
        return vocab_file

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the ASR vocabulary."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=str(model_id),
            filename="vocab.json",
        )
    )


def extend_ctc_vocabulary(
    vocabulary: Mapping[str, int],
    extra_tokens: Iterable[str] = (),
) -> dict[str, int]:
    """Append new single-character labels without changing existing IDs."""
    extended = {str(token): int(token_id) for token, token_id in vocabulary.items()}
    next_id = max(extended.values(), default=-1) + 1
    candidates = {
        str(token)
        for token in extra_tokens
        if isinstance(token, str)
        and len(token) == 1
        and token not in {" ", "|", "[UNK]", "[PAD]"}
    }
    for token in sorted(candidates, key=lambda value: (ord(value), value)):
        if token not in extended:
            extended[token] = next_id
            next_id += 1
    return extended


def _read_vocabulary(vocab_file: Path) -> dict[str, int]:
    try:
        vocabulary = json.loads(vocab_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read tokenizer vocabulary: {vocab_file}") from exc

    if not isinstance(vocabulary, dict) or not vocabulary:
        raise RuntimeError(f"Tokenizer vocabulary is empty or invalid: {vocab_file}")
    try:
        vocabulary = {
            str(token): int(token_id) for token, token_id in vocabulary.items()
        }
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Tokenizer vocabulary IDs must all be integers.") from exc
    return vocabulary


def build_clean_tokenizer(
    model_id: str | Path,
    *,
    extra_tokens: Iterable[str] = (),
) -> Any:
    """Rebuild the CTC tokenizer from vocab.json with one clean token map.

    Some saved tokenizer configurations can accumulate duplicate special-token
    entries after repeated ``save_pretrained`` calls.  CTC only needs the
    vocabulary plus its delimiter and special-token IDs, so rebuilding from
    ``vocab.json`` avoids carrying those accidental entries into a new run.
    ``extra_tokens`` can append dataset characters while preserving every
    existing token ID.
    """
    try:
        from transformers import Wav2Vec2CTCTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required to build the ASR tokenizer."
        ) from exc

    vocab_file = _resolve_vocab_file(model_id)
    vocabulary = _read_vocabulary(vocab_file)

    if vocabulary.get("|") != 0:
        raise RuntimeError(
            f"Expected the word-delimiter token '|' to have ID 0; got {vocabulary.get('|')!r}."
        )
    vocabulary_ids = list(vocabulary.values())
    if len(set(vocabulary_ids)) != len(vocabulary_ids):
        raise RuntimeError("Tokenizer vocabulary contains duplicate token IDs.")
    if set(vocabulary_ids) != set(range(max(vocabulary_ids) + 1)):
        raise RuntimeError("Tokenizer vocabulary IDs must be contiguous from zero.")
    for token in ("[UNK]", "[PAD]"):
        if token not in vocabulary:
            raise RuntimeError(f"Tokenizer vocabulary is missing required token {token!r}.")

    extra_tokens = list(extra_tokens)
    temporary_vocab_file: Path | None = None
    if extra_tokens:
        vocabulary = extend_ctc_vocabulary(vocabulary, extra_tokens)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as temporary_file:
            json.dump(vocabulary, temporary_file, ensure_ascii=False, indent=2)
            temporary_vocab_file = Path(temporary_file.name)
        tokenizer_file = temporary_vocab_file
    else:
        tokenizer_file = vocab_file

    try:
        tokenizer = Wav2Vec2CTCTokenizer(
            str(tokenizer_file),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            replace_word_delimiter_char=" ",
            do_lower_case=False,
        )
    finally:
        if temporary_vocab_file is not None:
            temporary_vocab_file.unlink(missing_ok=True)
    validate_tokenizer_configuration(tokenizer)
    return tokenizer


def validate_tokenizer_configuration(tokenizer: Any) -> dict[str, Any]:
    """Validate the token IDs that must remain stable for CTC training."""
    delimiter_id = tokenizer.convert_tokens_to_ids("|")
    unk_id = tokenizer.unk_token_id
    pad_id = tokenizer.pad_token_id
    if tokenizer.word_delimiter_token != "|":
        raise RuntimeError(
            "The tokenizer word delimiter must be '|', "
            f"not {tokenizer.word_delimiter_token!r}."
        )
    if delimiter_id is None or delimiter_id == unk_id:
        raise RuntimeError("The tokenizer word delimiter '|' is not in the vocabulary.")
    if unk_id is None or pad_id is None:
        raise RuntimeError("The tokenizer must define both [UNK] and [PAD] tokens.")
    if getattr(tokenizer, "replace_word_delimiter_char", " ") != " ":
        raise RuntimeError("The tokenizer must map spaces to the '|' delimiter token.")

    # Unknown characters are acceptable, but a space must become the delimiter.
    delimiter_probe = tokenizer("a b", add_special_tokens=False).input_ids
    if delimiter_probe and isinstance(delimiter_probe[0], list):
        delimiter_probe = delimiter_probe[0]
    if delimiter_id not in delimiter_probe:
        raise RuntimeError("Encoding a space did not produce the '|' delimiter token.")

    return {
        "tokenizer_length": len(tokenizer),
        "base_vocab_size": getattr(tokenizer, "vocab_size", None),
        "delimiter_token": tokenizer.word_delimiter_token,
        "delimiter_token_id": int(delimiter_id),
        "unk_token_id": int(unk_id),
        "pad_token_id": int(pad_id),
        "labels_add_special_tokens": False,
    }


def validate_ctc_processor_model(processor: Any, model: Any) -> dict[str, Any]:
    """Ensure tokenizer IDs and the model's CTC output classes agree."""
    diagnostics = validate_tokenizer_configuration(processor.tokenizer)
    model_vocab_size = int(model.config.vocab_size)
    diagnostics["model_vocab_size"] = model_vocab_size
    output_layer = getattr(model, "lm_head", None)
    output_classes = getattr(output_layer, "out_features", None)
    if output_classes is not None:
        diagnostics["model_output_classes"] = int(output_classes)

    if diagnostics["tokenizer_length"] != model_vocab_size:
        raise RuntimeError(
            "Tokenizer/model vocabulary mismatch: "
            f"tokenizer length={diagnostics['tokenizer_length']}, "
            f"model vocab_size={model_vocab_size}."
        )
    if output_classes is not None and int(output_classes) != model_vocab_size:
        raise RuntimeError(
            "Model CTC output mismatch: "
            f"lm_head classes={int(output_classes)}, model vocab_size={model_vocab_size}."
        )
    if int(model.config.pad_token_id) != diagnostics["pad_token_id"]:
        raise RuntimeError(
            "Tokenizer/model padding ID mismatch: "
            f"tokenizer={diagnostics['pad_token_id']}, "
            f"model={model.config.pad_token_id}."
        )
    diagnostics["compatible"] = True
    return diagnostics


def save_clean_processor(processor: Any, output_dir: str | Path) -> None:
    """Save a processor and remove malformed duplicate special-token entries."""
    output_path = Path(output_dir)
    processor.save_pretrained(str(output_path))
    config_path = output_path / "tokenizer_config.json"
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_size = len(processor.tokenizer)
    decoder = config.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        clean_decoder: dict[str, Any] = {}
        for token_id, token_info in decoder.items():
            try:
                numeric_id = int(token_id)
            except (TypeError, ValueError):
                continue
            content = token_info.get("content") if isinstance(token_info, dict) else None
            if numeric_id < expected_size and not (
                isinstance(content, str) and content.startswith("{'content':")
            ):
                clean_decoder[str(numeric_id)] = token_info
        config["added_tokens_decoder"] = clean_decoder

    # This field is where the malformed repeated stringified token objects
    # appeared in the previous checkpoint.  The decoder above is sufficient.
    config.pop("extra_special_tokens", None)
    config["word_delimiter_token"] = "|"
    config["replace_word_delimiter_char"] = " "
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_processor(
    model_id: str | Path,
    *,
    extra_tokens: Iterable[str] = (),
) -> Any:
    """Load a clean greedy CTC processor without the optional KenLM decoder."""
    try:
        from transformers import AutoFeatureExtractor, Wav2Vec2Processor
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for ASR inference."
        ) from exc

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_id))
    tokenizer = build_clean_tokenizer(model_id, extra_tokens=extra_tokens)
    return Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


def resize_ctc_output_layer(model: Any, new_vocab_size: int) -> None:
    """Resize a Wav2Vec2 CTC head while preserving existing output rows."""
    import torch

    if new_vocab_size < 1:
        raise ValueError("new_vocab_size must be positive")
    output_layer = getattr(model, "lm_head", None)
    if output_layer is None or not hasattr(output_layer, "in_features"):
        raise RuntimeError("The CTC model does not expose a compatible lm_head.")

    old_vocab_size = int(output_layer.out_features)
    if old_vocab_size != new_vocab_size:
        new_layer = torch.nn.Linear(
            int(output_layer.in_features),
            int(new_vocab_size),
            bias=output_layer.bias is not None,
        )
        new_layer = new_layer.to(
            device=output_layer.weight.device,
            dtype=output_layer.weight.dtype,
        )
        initializer_range = float(
            getattr(model.config, "initializer_range", 0.02)
        )
        torch.nn.init.normal_(new_layer.weight, mean=0.0, std=initializer_range)
        if new_layer.bias is not None:
            torch.nn.init.zeros_(new_layer.bias)

        rows_to_copy = min(old_vocab_size, int(new_vocab_size))
        with torch.no_grad():
            new_layer.weight[:rows_to_copy].copy_(output_layer.weight[:rows_to_copy])
            if output_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias[:rows_to_copy].copy_(output_layer.bias[:rows_to_copy])
        model.lm_head = new_layer

    model.config.vocab_size = int(new_vocab_size)
    if hasattr(model.config, "num_labels"):
        model.config.num_labels = int(new_vocab_size)


def load_asr_model(model_id: str | Path, device: Any) -> tuple[Any, Any]:
    """Load a CTC model and put it into inference mode."""
    try:
        from transformers import AutoModelForCTC
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for ASR inference."
        ) from exc

    model = AutoModelForCTC.from_pretrained(str(model_id))
    processor = load_processor(model_id)
    validate_ctc_processor_model(processor, model)
    model.to(device)
    model.eval()
    return processor, model


def transcribe(
    samples: Any,
    processor: Any,
    model: Any,
    device: Any,
    *,
    decoder: str = "greedy_ctc",
    beam_width: int = 10,
) -> tuple[str, float, float, float]:
    """Run CTC inference and return text plus timing metadata.

    The beam decoder is opt-in so existing application behavior remains
    unchanged. It does not use a language model; it only searches more CTC
    paths than the greedy argmax decoder.
    """
    import torch

    if decoder not in {"greedy_ctc", "beam_ctc"}:
        raise ValueError("decoder must be 'greedy_ctc' or 'beam_ctc'")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    inputs = processor(
        samples,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_latency = time.perf_counter() - started

    decode_started = time.perf_counter()
    if decoder == "greedy_ctc":
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids, group_tokens=True)[0]
    else:
        tokenizer = processor.tokenizer
        blocked_ids = {
            token_id
            for token_id in (
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
            )
            if token_id is not None
        }
        token_ids = ctc_prefix_beam_search(
            logits[0],
            blank_id=int(tokenizer.pad_token_id),
            beam_width=beam_width,
            blocked_ids=blocked_ids,
        )
        prediction = tokenizer.decode(
            token_ids,
            group_tokens=False,
            skip_special_tokens=False,
        )
    decode_latency = time.perf_counter() - decode_started
    return (
        normalize_asr_text(prediction),
        model_latency + decode_latency,
        model_latency,
        decode_latency,
    )


class ASRService:
    """Lazy-loading inference service suitable for a long-running web app."""

    def __init__(
        self,
        model_source: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.model_source = resolve_model_source(model_source)
        self.requested_device = device
        self._device: Any = None
        self._processor: Any = None
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "model": self.model_source,
            "device": str(self._device) if self._device is not None else self.requested_device,
            "target_sample_rate": TARGET_SAMPLE_RATE,
        }

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        self._device = resolve_device(self.requested_device)
        self._processor, self._model = load_asr_model(self.model_source, self._device)

    def warmup(self) -> dict[str, Any]:
        """Load the model before the first user transcription request."""
        self._ensure_loaded()
        return self.status()

    def transcribe_wav(self, audio_bytes: bytes) -> dict[str, Any]:
        """Transcribe WAV bytes and return both scripts plus timing metadata."""
        from khmer_transliteration.romanization import romanize_khmer_text

        samples, _, original_rate, was_resampled = decode_and_resample(audio_bytes)
        self._ensure_loaded()
        prediction, latency, model_latency, decoder_latency = transcribe(
            samples,
            self._processor,
            self._model,
            self._device,
        )
        romanized = romanize_khmer_text(prediction)
        return {
            "khmer_text": prediction,
            "romanized_text": romanized,
            "latency_seconds": latency,
            "model_latency_seconds": model_latency,
            "decoder_latency_seconds": decoder_latency,
            "sample_rate_before_resampling": original_rate,
            "resampled_to_target_rate": was_resampled,
            "model": self.model_source,
            "device": str(self._device),
        }
