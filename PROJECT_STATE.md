# Khmer Transliteration Keyboard Project State

Current workspace:
C:\Projects\Dual-Script-Khmer-ASR

Project goal:
Build a Romanized Khmer to Khmer script input/prediction system for a Machine Learning course.

Current focus:
The keyboard remains the base application. The active extension adds Khmer ASR
with XLS-R/Wav2Vec2, held-out evaluation, and dual-script browser output. The
ASR workflow and current limitations are documented in `docs/asr.md`.

Completed steps:
1. Dataset prepared in `data/all_words.csv`.
2. Custom informal words stored in `data/custom_words.csv`.
3. Input normalization implemented in `khmer_transliteration/normalizer.py`.
4. Exact + fuzzy dictionary lookup implemented in `khmer_transliteration/dictionary_lookup.py`.
5. Mapping rules stored in `data/mapping_rules.json`.
6. Candidate generation started in `khmer_transliteration/candidate_generator.py`.
7. HTML candidate test report implemented in `scripts/candidate_test_report.py`.
8. Suggestion engine implemented in `khmer_transliteration/suggestion_engine.py`.
9. HTML suggestion test report implemented in `scripts/suggestion_test_report.py`.

Important files:
- `data/all_words.csv`: final dataset used by lookup.
- `data/custom_words.csv`: manually curated informal words.
- `data/mapping_rules.json`: consonants, vowels, consonant classes, muusikatoan, triisap, etc.
- `khmer_transliteration/mapping_rules.py`: loads mapping rules and returns token patterns.
- `khmer_transliteration/normalizer.py`: normalizes Romanized input.
- `khmer_transliteration/dictionary_lookup.py`: exact/fuzzy lookup.
- `khmer_transliteration/candidate_generator.py`: Step 6 rule-based generation.
- `scripts/candidate_test_report.py`: generates HTML visual test report.
- `khmer_transliteration/suggestion_engine.py`: merges dictionary and rule-based suggestions.
- `scripts/suggestion_test_report.py`: generates HTML report for final merged suggestions.


Current folder structure after refactor:
- `khmer_transliteration/`: importable application package.
- `scripts/`: command-line scripts for reports, training, and data maintenance.
- `reports/`: generated HTML/text report outputs.
- `data/`: CSV data and JSON mapping rules only.
- `app.py`: small compatibility wrapper so `python -m uvicorn app:app --host 127.0.0.1 --port 8000` still works.

Current Step:
Step 7B multi-syllable rule sequence generation is implemented.

Step 6 implemented:
- 6A: longest-first tokenization.
- 6B: multiple tokenization.
- 6C: token to Khmer options.
- 6D: CV and CVC generation.
- 6E: CC and inherent-vowel generation.
- 6F: candidate de-duplication and ranking across tokenizations.

Step 7 implemented:
- Exact dictionary suggestions.
- Left-to-right dictionary completion suggestions.
- Rule-generated fallback suggestions.
- 7B: multi-syllable rule sequence generation using smaller CV/CVC/CC chunks.
- 7C: subscript cluster syllable patterns: CCV, CCVC, CCCV, CCCVC, CCCC.
- Sequence generation can combine chunk sizes 2, 3, 4, and 5 for longer words.
- Cluster syllables fall back to written vowel signs when class rules return only inherent vowel.
- `subscript.allowed_following_consonants` can restrict which subscript consonants a base consonant may combine with.
- Current explicit subscript compatibility rules are defined for bases:
  `ក`, `ខ`, `គ`, `ឃ`, `ច`, `ឆ`, `ជ`, `ឈ`, `ត`, `ថ`, `ទ`, `ធ`, `ប`, `ផ`, `ព`, `ភ`, `ម`, `ល`, `ស`.

Step 8 implemented:
- Ranking model training script in `scripts/train_ranking_model.py`.
- Ranking features in `khmer_transliteration/ranking_features.py`.
- Model artifact saved to `models/ranking_model.joblib`.
- Training labels include full exact cases and prefix-completion cases.
- Leaky source-priority/manual-score features are excluded from model features.
- Manual ranking label queue can be generated with `khmer_transliteration/collection.py` / `scripts/collect_ranking_examples.py`.
- Manual labels are stored in `data/ranking_training_examples.csv`.
- `scripts/train_ranking_model.py` uses manual rows only when `label` is `0` or `1`.
- `khmer_transliteration/suggestion_engine.py` also applies manual labels directly: `1` boosts and `0` demotes exact `(input, khmer)` candidates.
- Suggestions expose `rank_score` and `rank_reason` for UI/debugging.
- Manual `score` is the rule/dictionary score; `ml_score` is model probability; `rank_score` is the combined debug ranking number.

Step 9 implemented:
- `khmer_transliteration/suggestion_engine.py` loads `models/ranking_model.joblib` when available.
- Suggestions include `ml_score` from the trained model.
- Sorting uses source priority first, then manual score, then ML score.
- `scripts/suggestion_test_report.py` displays ML score when present.

Step 10 implemented:
- Simple FastAPI UI in `app.py` wrapper / `khmer_transliteration/web.py`.
- Frontend files in `static/index.html`, `static/styles.css`, and `static/app.js`.
- Angkor Wat themed UI uses `assets/angkor-wat-temples.jpg`.
- UI endpoint: `/api/suggest?q=...`.
- UI review endpoint: `/api/collect?q=...` appends candidates to `data/ranking_training_examples.csv`.
- UI Confirm button sends the current input and generated candidates to the manual label CSV.
- Run UI with `python -m uvicorn app:app --host 127.0.0.1 --port 8000`.


Context and personalization implemented:
- Previous-word context is auto-collected when the user clicks suggestions.
- Pair counts are stored in `data/word_pair_frequency.csv`.
- User selection history is auto-collected in `data/user_selection_history.csv`.
- `khmer_transliteration/suggestion_engine.py` applies capped score boosts for both signals.
- `khmer_transliteration/ranking_features.py` includes `previous_word_context_score` and `user_history_score` for ML retraining.
- `scripts/train_ranking_model.py` adds training cases from both history CSVs.
- The current old model may skip ML scores until retrained with the new feature list.

Important Step 6 rules:
- Initial consonants may use normal, muusikatoan, or triisap variants.
- Final consonants use normal consonants only.
- `អ` is not allowed as final/last consonant in CVC or CC.
- Plain CC:
  - first C can use muusikatoan/triisap
  - second C normal only
- Subscript CC:
  - both consonants normal only
- `subscript.coeng` is `្`.
- `muusikatoan_consonants` behave like first series using `consonant_class_overrides`.
- `triisap_consonants` behave like second series using `consonant_class_overrides`.
- `vowels_by_consonant_class` should be checked before fallback `vowels`.

Recent important changes:
- `khmer_transliteration/mapping_rules.py` includes `triisap_consonants`.
- `khmer_transliteration/candidate_generator.py` includes `generate_ranked_candidates()` for 6F.
- `khmer_transliteration/candidate_generator.py` includes `rule_sequence` candidates for Step 7B.
- `khmer_transliteration/candidate_generator.py` includes subscript cluster sources for Step 7C.
- `khmer_transliteration/suggestion_engine.py` combines exact lookup, left-to-right completion, and generated candidates.
- Suggestion priority is exact dictionary > dictionary completion > rule-generated.
- Direct token suggestions have highest priority for consonants and independent vowels.
- Dependent vowel signs can be requested with `allow_vowels=true` after a character is selected.
- Default suggestion limit is 10.
- Rule-generated suggestions are displayed only when score >= 1.60.
- Dictionary completion is left-to-right: dataset romanization must start with the typed input.
- Dictionary completion is limited to at most 2 extra Roman characters beyond the typed input.
- Fuzzy lookup is available through `khmer_transliteration/dictionary_lookup.py` and guarded in `khmer_transliteration/suggestion_engine.py` for longer inputs.
- Whitelisted compound vowel tokenizations, such as `am`, `om`, `um`, and `av`, receive a ranking bonus over split CVC forms.
- Tokenizations that split known consonant tokens, such as `n+h` instead of `nh`, receive a ranking penalty.
- `scripts/suggestion_test_report.py` generates `reports/suggestion_test_report.html`.
- `scripts/train_ranking_model.py` generates `reports/ranking_model_report.txt`.
- `app.py` wrapper / `khmer_transliteration/web.py` serves the local keyboard UI.
- `scripts/candidate_test_report.py` shows all tokenizations, including those with no candidates.
- `scripts/candidate_test_report.py` shows ranked candidates before per-tokenization details.
- `reports/candidate_test_report.html` is generated by running:
  `python scripts/candidate_test_report.py`

Testing:
Run:
```powershell
python scripts/candidate_test_report.py
```

Then open:
reports/candidate_test_report.html
Expected examples:
vav should show ['v', 'av'] and ['v', 'a', 'v'].
['v', 'av'] should generate វ៉ៅ.
['v', 'a', 'v'] should generate វ៉ាវ.
mj should generate plain/subscript CC candidates.
Subscript CC should not generate muusikatoan/triisap variants.

## Khmer ASR extension

The ASR extension is implemented on branch `Application-Integration` and uses
the pretrained `vitouphy/wav2vec2-xls-r-300m-khmer` checkpoint as its baseline
and fine-tuning starting point.

Completed ASR work:

- Dataset inspection, deterministic manifest building, and manifest validation
  scripts are in `scripts/inspect_asr_dataset.py`, `scripts/build_asr_manifest.py`,
  and `scripts/validate_asr_manifest.py`.
- Shared ASR logic is in `khmer_transliteration/asr_data.py`,
  `asr_audio.py`, `asr_inference.py`, `asr_metrics.py`, and `asr_text.py`.
- Baseline evaluation is in `scripts/run_asr_baseline.py`.
- Fine-tuning is in `scripts/finetune_asr.py`; train/validation are used for
  training and the test split remains held out.
- A 1,000-row Colab pilot was fine-tuned and copied to Google Drive. The
  original model folder is `models/khmer_xlsr_ddd_1000/`; the corrected
  tokenizer experiment will use `models/khmer_xlsr_ddd_1000_clean/`. Both are
  ignored by Git.
- The FastAPI app exposes `/api/asr/status` and `POST /api/asr`.
- The browser UI supports microphone recording or WAV upload and displays both
  Khmer script and dictionary-backed canonical Romanized Khmer.
- Unit tests are in `tests/`; run them with
  `python -m unittest discover -s tests -v`.

Important limitation:

The Khmer-to-Romanized output is currently a dictionary-backed canonical
approximation because reverse romanization is ambiguous. A manually verified
Romanized evaluation subset is still needed before claiming Romanization
accuracy. See `docs/asr.md` for the complete workflow and known limitations.
