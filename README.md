# Khmer Transliteration Keyboard

Romanized Khmer to Khmer script suggestion system with dictionary lookup,
rule-based generation, fuzzy lookup, ML-assisted ranking, a FastAPI test UI,
and a Windows TSF IME prototype.

The current Windows IME uses a local named-pipe engine instead of a web port.
The pipe engine keeps the Python rules/dictionary/model loaded in memory and
records local user selections for personalization.

The project also contains a Khmer speech-recognition extension that fine-tunes
an XLS-R/Wav2Vec2 model and exposes Khmer plus Romanized output in the browser
UI. See [docs/asr.md](docs/asr.md) for the dataset workflow, Colab commands,
evaluation rules, model storage, and API details.

## Fresh Clone Setup

After cloning the repo in VS Code, do these steps first.

### 1. Install Requirements

- Windows 10/11
- Python 3.10+
- Git
- For the real Windows IME prototype only: Visual Studio C++ Build Tools with
  the Windows SDK

### 2. Create Python Environment

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 3. Verify The Python Engine

Run a quick report/test command:

```powershell
python scripts/suggestion_test_report.py
```

Or start the browser UI:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

### 4. Test The Local IME Pipe Engine

The Windows IME no longer needs port `8000`; it talks to a local named pipe.

```cmd
windows_ime\engine\start_pipe_engine.cmd
```

In another terminal:

```powershell
python windows_ime/engine/test_pipe_client.py som
```

### 5. Optional: Build/Register The Windows IME

Only do this if you want the actual Windows keyboard integration.

Open an x64 Developer Command Prompt/PowerShell, then:

```cmd
cd /d C:\Projects\Dual-Script-Khmer-ASR\windows_ime\prototype
build
reregister
check
```

⚠️ If your clone is in a different folder, use that folder path instead of
`C:\Projects\Dual-Script-Khmer-ASR`.

### 6. Optional: Start Pipe Engine At Login

This makes the local engine run silently after Windows login:

```cmd
windows_ime\engine\install_login_startup.cmd
```

To remove it:

```cmd
windows_ime\engine\uninstall_login_startup.cmd
```

### Common Fresh-Clone Problems

- `pip` points to an old moved `.venv`: delete `.venv`, recreate it, then run
  `python -m pip install -r requirements.txt`.
- `cl` shows x86 instead of x64: open the x64 Developer shell before building
  the IME.
- IME does not appear in Windows language options: run `reregister`, then
  sign out/in or restart Explorer.
- Suggestions work in browser UI but not IME: make sure
  `windows_ime\engine\start_pipe_engine.cmd` is running, or install login
  startup.
- Port `8000` conflict: only the browser UI uses port `8000`; the Windows IME
  uses `\\.\pipe\KhmerRomanizedIme`.

## Project Structure

- `khmer_transliteration/` - core Python package
  - `candidate_generator.py` - rule-based candidate generation
  - `suggestion_engine.py` - dictionary/rule/fuzzy/ML suggestion merge
  - `dictionary_lookup.py` - dataset lookup helpers
  - `mapping_rules.py` - mapping rule loader
  - `collection.py` - manual ranking-label CSV collection
  - `web.py` - FastAPI app implementation
  - `paths.py` - shared project paths
- `khmer_transliteration/asr_*.py` - shared ASR data, audio, inference, metrics,
  and normalization utilities
- `data/` - CSV datasets and `mapping_rules.json`
  - `user_selection_history.csv` - auto-created from clicked suggestions
  - `word_pair_frequency.csv` - auto-created previous-word pair counts
- `models/` - trained ranking model artifacts
- `scripts/` - command-line reports, training, and data utilities
- `reports/` - generated HTML/text reports
- `static/` and `assets/` - UI files and images
- `windows_ime/` - Windows TSF IME prototype and local pipe engine
- `app.py` - compatibility wrapper for `uvicorn app:app`
- `docs/scoring_logic_snapshot.md` - current scoring/ranking reference
- `docs/asr.md` - Khmer ASR preparation, fine-tuning, evaluation, and UI guide

## Common Commands

```powershell
python scripts/candidate_test_report.py
python scripts/suggestion_test_report.py
python scripts/train_ranking_model.py --max-inputs 500 --candidates-per-input 10
python scripts/collect_ranking_examples.py somtos leakk
python scripts/promote_label_data.py
python scripts/auto_label_training_examples.py --dry-run
python scripts/auto_label_training_examples.py --overwrite-auto
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The FastAPI server is still useful for the browser UI and reports. The Windows
IME prototype uses the named pipe at `\\.\pipe\KhmerRomanizedIme` instead of
port `8000`.

## Windows IME Prototype

Start the local pipe engine manually:

```cmd
windows_ime\engine\start_pipe_engine.cmd
```

Install the pipe engine at Windows login:

```cmd
windows_ime\engine\install_login_startup.cmd
```

Remove login startup:

```cmd
windows_ime\engine\uninstall_login_startup.cmd
```

Rebuild/register the TSF IME from an x64 Developer shell:

```cmd
cd /d C:\Projects\Dual-Script-Khmer-ASR\windows_ime\prototype
reregister
```

The TSF IME records selected candidates locally in:

```text
data/user_selection_history.csv
data/word_pair_frequency.csv
```

Both history files are capped at 10,000 rows when written.

Retrain the ranking model after collecting selection history or word-pair data
so ML can learn the new context/history features.

By default, training skips fuzzy and compound generation for speed. Add
`--include-fuzzy` or `--include-compound` when you want slower, broader training.

## Manual Label Review Flow

The browser UI Confirm button writes generated candidates to `data/label_data.csv`.
Review that file manually:

- `label = 1` means good; it will rank high.
- `label = 0` means bad; it will be hidden from UI suggestions.
- blank means not reviewed yet; during promotion it is ignored.

When review is done, promote the inbox into the training/label file:

```powershell
python scripts/promote_label_data.py
```

That command copies only explicit `1` and `0` rows into
`data/ranking_training_examples.csv`. Blank rows are skipped, then
`data/label_data.csv` is cleared so the next UI collection starts fresh.

To preview without clearing the inbox:

```powershell
python scripts/promote_label_data.py --keep-review
```

Retrain after promotion when you want ML to learn the new labels:

```powershell
python scripts/train_ranking_model.py
```

## Auto-label Training Rows

`data/ranking_training_examples.csv` is the promoted training/label file.
Use the auto-label tool to compare those rows against `data/all_words.csv`.

Preview labels without writing:

```powershell
python scripts/auto_label_training_examples.py --dry-run
```

Write auto labels. Existing human labels are preserved:

```powershell
python scripts/auto_label_training_examples.py --overwrite-auto
```

Label only specific collected inputs:

```powershell
python scripts/auto_label_training_examples.py chir som --overwrite-auto
```

Process only the first few collected inputs:

```powershell
python scripts/auto_label_training_examples.py --max-inputs 10 --dry-run
```

The default mode only labels rows already collected from the UI. It labels
dataset-matching Khmer candidates as `1`, labels wrong candidates above/between
correct matches as `0`, and leaves lower rows blank.

Dataset-wide generation is still available, but it must be requested explicitly:

```powershell
python scripts/auto_label_training_examples.py --from-dataset --max-inputs 100 --dry-run
```
