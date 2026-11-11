# Kaggle 5,000-row ASR run

This workflow prepares a 5,000-row XLS-R fine-tuning run on Kaggle and keeps the
existing 126-row held-out test set unchanged. It does not require another manual
WER audit.

## Prepare the source bundle

From the project root in PowerShell, run:

```powershell
.\.venv\Scripts\python.exe scripts\package_kaggle_asr.py
```

This creates `kaggle/khmer_asr_5k_source_bundle.zip`. The bundle contains the
ASR scripts and modules, the original 1,000-row test manifest and its source
configuration, plus the existing baseline and 1,000-row evaluation reports. It
does not contain audio or model weights; Kaggle downloads those when needed.

## Start Kaggle

1. Create a private Kaggle Dataset containing `khmer_asr_5k_source_bundle.zip`
   (or the extracted bundle files) and attach it to a new Kaggle Notebook.
2. Upload or import `notebooks/finetune_asr_kaggle.ipynb` into that notebook.
3. In the notebook settings, enable a GPU accelerator and Internet access.
4. Run the cells from the top. The notebook builds the expanded manifest,
   materializes audio, checks the vocabulary, runs a small smoke training, then
   launches full training.

The manifest builder copies the original test split, then excludes its speakers
and transcript groups from the new training and validation rows. The 5,000-row
target includes the 126 held-out test examples. Results can therefore be
compared directly with the existing 1,000-row fine-tuned model and pretrained
baseline on those same test examples.

The notebook writes the resumable model directory, training WAVs, and reports
under `/kaggle/working/asr_run`. Keep these files until training and final
evaluation finish. To retain outputs, save a Kaggle notebook version after the
materialization cell and after training. A checkpoint can be resumed only when
both the model output directory and the materialized training directory are
available in the new session. Set `RESTORE_FROM` and `RESUME_TRAINING` in the
notebook as described in its training cell. If Kaggle did not retain a saved
version containing the latest checkpoint, the run cannot resume from it.

At the end, the notebook creates
`khmer_xlsr_ddd_5000_clean_model.zip` with the final model and processor files,
and `finetuned_test_5000_fixed.json` with the full test predictions and metrics.
Save a Kaggle notebook version, then download those two outputs for the local
project.
