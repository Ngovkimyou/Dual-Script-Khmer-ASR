# Khmer ASR Project Overview

**Project status:** The 1,000-row pilot is complete. The 2,000-row follow-up is prepared and waiting for access to the gated dataset.  
**Last updated:** 17 November 2026

This is the short project overview for reviewers. For commands and implementation
details, see the [ASR workflow guide](docs/asr.md).

## Project aim

This project evaluates whether fine-tuning a Khmer-capable speech-recognition
model on labeled Khmer audio improves transcription quality. It also connects
speech input to the existing Khmer transliteration keyboard application.

The recognizer outputs Khmer script. The application also provides a
best-effort Romanized display using Khmer word segmentation and the existing
Romanized dictionary. Words absent from the dictionary remain in Khmer. This
display is not a separately trained or evaluated model.

## Dataset

The source is Digital Divide Data's [Khmer speech dataset on Hugging
Face](https://huggingface.co/datasets/Digital-Divide-Data/khmer-speech-dataset),
a collection of Khmer speech and matching transcripts about Cambodian
cultural topics. The dataset contains about 450,000 records. Access is gated,
so new downloads require an authorized Hugging Face account.

The completed pilot selected 1,000 rows:

| Split | Rows | Use |
| --- | ---: | --- |
| Train | 793 | Model fine-tuning |
| Validation | 81 | Training checks and checkpoint selection |
| Test | 126 | Final evaluation only |

Audio was prepared as mono, 16 kHz WAV. Splits keep speakers separate to
reduce speaker overlap. The test split was excluded from training.

## Model and method

The project fine-tunes
[`vitouphy/wav2vec2-xls-r-300m-khmer`](https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-khmer),
a pretrained Wav2Vec2/XLS-R CTC model. It does **not** train a model from
scratch.

The tokenizer was rebuilt to cover the Khmer characters in the training and
validation transcripts. The `|` word-delimiter token is retained, and
transcript labels are encoded without adding unrelated special tokens.

## Pilot results

Both models were evaluated on the same 126 held-out test samples. Rates below are percentages; lower error rates are better.

| Model | CER | Khmer-aware WER |
| --- | ---: | ---: |
| Pretrained XLS-R baseline | 34.89% | 70.28% |
| Fine-tuned on the pilot (793 training rows) | **17.66%** | **46.35%** |

The pilot fine-tuning reduced CER by 17.23 percentage points and WER by 23.93
percentage points on this test set. These are pilot results, not a final
performance claim.

CER compares normalized characters. Khmer-aware WER first segments each
transcript with the pinned `khmercut` tokenizer. Because Khmer word boundaries
can be ambiguous, WER is an estimate and should be read together with CER and
example predictions. On a 30-sample diagnostic subset, manual segmentation
gave 43.67% WER, compared with 50.78% using automatic segmentation. This small
subset is not a full-test estimate. The JSON reports store rates as decimals;
this overview displays them as percentages.

Detailed reports: [pretrained baseline](reports/asr/baseline_test_rescored.json)
and [fine-tuned model](reports/asr/finetuned_test_clean_rescored.json).

## Current status and next steps

The next experiment targets 2,000 total rows: it preserves the pilot's 1,000
rows and split assignments, then adds 1,000 new rows to train/validation. The
same 126 test rows remain held out. The target total includes those test rows.

The 2,000-row notebook and manifest workflow are ready. The run cannot fetch
the additional data until the Hugging Face account is granted access. After
access is available, the next steps are to build and validate the expanded
manifest, train from the same pretrained checkpoint, and compare results on
the unchanged test set.

## Known limitations

- The current training set is small relative to the source dataset.
- Khmer-aware WER depends on automatic word segmentation, whose boundaries may be imperfect.
- Romanized output depends on word segmentation and dictionary coverage; unknown words may remain in Khmer, and the output has not been evaluated as an independent model.
- The 2,000-row experiment has not completed, so no result is reported for it yet.

## Setup and project files

Install the ASR dependencies from the repository root:

```powershell
python -m pip install -r requirements-asr.txt
```

On Windows, start the application with 
```powershell
python -X utf8 -m uvicorn app:app --host 127.0.0.1 --port 8000
```
so the Khmer word segmenter can read its model
data correctly.

The fine-tuned checkpoint (approximately 1.26 GB) is distributed separately
and is not stored in Git. To reproduce local ASR inference, download
`models/khmer_xlsr_ddd_1000_clean` from the [shared Google Drive ASR folder](https://drive.google.com/drive/folders/1cCC9Bo5iZ3t-rJs6-qrUiopC4rci4t9a?usp=sharing)
and place it at `models/khmer_xlsr_ddd_1000_clean/`; the application detects
the model there automatically. Set `KHMER_ASR_MODEL_PATH` only if the model is
stored elsewhere. The
[ASR workflow guide](docs/asr.md) describes the local setup and Vercel
deployment constraints.

The ASR fine-tuning notebook is
[`notebooks/finetune_asr_2000.ipynb`](notebooks/finetune_asr_2000.ipynb). Its
cells describe the required uploads, organize files, install packages, check
dataset access, train, and evaluate. For local commands, checkpoint recovery,
evaluation details, and the web API, see the [ASR workflow guide](docs/asr.md).

Main code locations:

- `scripts/`: dataset inspection, manifest creation and validation, inference,
  scoring, and fine-tuning
- `khmer_transliteration/asr_*.py`: shared audio, data, decoding, inference,
  metric, and text helpers
- `requirements-asr.txt`: ASR-specific Python dependencies
- `notebooks/`: dataset preparation, baseline evaluation, and fine-tuning
  workflows

`Generative AI use: ChatGPT/Codex assisted with explaining ASR concepts and metrics, troubleshooting notebook and dataset-access issues, and drafting or revising code and project documentation. The project team reviewed the work and is responsible for the final implementation and interpretation. The audio and transcripts came from the cited dataset; model predictions were produced by the ASR system.`