# Information Extraction from Regional Construction Price Lists

This repository contains the code accompanying **Leveraging Semantic Decomposition for AI Training towards Automating Information Extraction from Regional Construction Price Lists**, accepted at ICAI-TEMS 2026. It extracts and classifies entity mentions in Italian construction cost-item descriptions using GLiNER and Large Language Models (LLMs).

The code is organized into entity recognizers, shared LLM providers, and evaluation tools. YAML files define individual experiments, and Bash scripts in the repository root group the runs for each approach.

**Data and predictions are available upon request to the authors at [giorgia.marcellino@phd.unipd.it](mailto:giorgia.marcellino@phd.unipd.it).**

The paper studies 250 descriptions from the 2025 Veneto Regional Price List, split into 160 training, 40 validation, and 50 test descriptions. It compares zero-shot and fine-tuned GLiNER with zero-shot, fixed few-shot, and semantic few-shot LLM prompting.

## Citation

If you use this code or the associated data, please cite:

```bibtex
@inproceedings{marcellino2026leveraging,
  author    = {Giorgia Marcellino and Marco Martinelli and Ornella Irrera and
               Gianmaria Silvello and Carlo Zanchetta},
  title     = {Leveraging Semantic Decomposition for {AI} Training towards
               Automating Information Extraction from Regional Construction
               Price Lists},
  booktitle = {Proceedings of the 2nd IEEE International Conference on
               Application of Information Technologies in Engineering,
               Management and Science (ICAI-TEMS 2026)},
  year      = {2026},
  address   = {Pisa, Italy},
  month     = nov,
  note      = {Accepted for publication}
}
```

## Repository structure

```text
PriceList_IE/
├── run_gliner_entity_recognizer.py   # GLiNER inference wrapper
├── run_llm_entity_recognizer.py      # LLM inference wrapper
├── runGliner.sh                     # GLiNER zero-shot, training, and inference grid
├── runLLM.sh                        # Zero-shot LLM grid
├── runFewshot.sh                    # Fixed and semantic few-shot LLM grids
├── batchEval_NER.sh                 # Evaluate, merge, and format NER results
├── scripts/configs/
│   └── EntityRecognizer/           # GLiNER and LLM experiment configurations
├── src/
│   ├── entityRecognizer/          # Recognizers, few-shot selectors, IT/EN prompts
│   ├── termExtractor/             # Shared tokenizers, training bases, and evaluators
│   ├── modules/LLMProvider.py      # Hosted and local LLM backends
│   ├── utils/                     # Shared labels, data utilities, response parsing
│   └── evaluate/                  # NER scoring and CSV aggregation
├── data/                          # Requested dataset splits
├── predictions/                   # Generated entity predictions
└── checkpoints/                   # GLiNER models and optional LLM resume files
```

Evaluation creates `evaluation/`; semantic few-shot retrieval creates `cache/LLMEntityRecognizer/`.

| Component | Models and implementation |
| --- | --- |
| GLiNER | `VAGOsolutions/SauerkrautLM-GLiNER` (reported in the paper), plus `DeepMount00/universal_ner_ita` and `urchade/gliner_multi-v2.1`; zero-shot inference and supervised fine-tuning. |
| Zero-shot LLMs | GPT-4o and GPT-4o Mini through Azure OpenAI; Gemma 3 4B, 12B, and 27B through Ollama (reported in the paper). Alternative local Hugging Face configurations use `google/gemma-3-{4,12,27}b-it` (not reported in the paper). |
| Fixed few-shot | The same LLMs with four training examples; the supplied configs select a deterministic set with seed 42. Explicit IDs can be supplied through `few_shot.fixed_example_ids`. |
| Semantic few-shot | The same LLMs with four examples retrieved by embedding similarity. Supplied configs use `nickprock/sentence-bert-base-italian-uncased` through Sentence Transformers. |

## Setup

Run all commands **from the repository root**, using Bash on Linux, macOS, or WSL. The scripts use both `python` and `python3`; ensure both refer to the same environment.

Run these pip installations for dependencies:

```bash
python -m pip install torch transformers accelerate numpy pandas requests tqdm pyyaml python-dotenv openai huggingface_hub sentencepiece scikit-learn

# Install the packages needed by your experiments:
python -m pip install gliner                 # GLiNER
python -m pip install sentence-transformers  # Semantic few-shot retrieval
```

Local model memory requirements depend on model size. `scikit-learn` is required by the shared evaluator imports used by the recognizers.

### Data

Place the requested splits at:

```text
data/training.json
data/dev.json
data/test.json
```

Each file is a JSON object keyed by document ID. Records contain `text` and an `entities` list with `start_idx`, `end_idx`, `text_span`, and `label`. Character offsets are zero-based with an inclusive end, so `text[start_idx:end_idx + 1]` is the entity mention. Predictions use the same structure.

Use the canonical labels in `src/utils/utils.py`: `Elemento`, `Prodotto edilizio`, `Materiale`, `Proprietà nome`, `Proprietà_valore`, `Geometria nome`, `Geometria_valore`, `Metodo`, and `Voce di costo`.

Training and few-shot examples come from `training.json`; validation uses `dev.json`, and final evaluation uses `test.json`. Keep the supplied splits and record IDs to reproduce the experiments.

### API keys and local models

**Azure OpenAI.** Export the credentials for your Azure resource:

```bash
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
```

In every Azure YAML you use, replace the literal `<YOUR_API_KEY>` and `<YOUR_AZURE_ENDPOINT>` placeholders with `null` to enable environment-variable fallback. Set `model` to your Azure deployment name, and set `azure_api_version` to the version used by your deployment (the code default is `2024-02-01`). The LLM root wrapper also loads a local `.env` file; if using one, add it to your Git ignore rules before storing credentials there.

**Ollama.** Start the Ollama server with `ollama serve` if it is not already running, then download the configured models:

```bash
ollama pull gemma3:4b
ollama pull gemma3:12b
ollama pull gemma3:27b
```

The provider connects to `http://localhost:11434/v1` by default and requires no hosted API key. Set `base_url` in the YAML if your server uses another address.

**Hugging Face.** For Gemma inference, obtain access to the configured model repositories and authenticate locally:

```bash
hf auth login
```

These configurations load models through Transformers on the local machine. They do not use the YAML `api_key` field. GLiNER and the semantic embedding model are also downloaded from Hugging Face.

### Configuration

Edit the appropriate YAML to change paths, models, prompts, generation settings, or training parameters. **YAML values overwrite matching command-line arguments** in these runners; there is no `--override` interface. For LLM experiments, the supplied configs select `src/entityRecognizer/prompts_IT.json`, with both prompt keys set to `base`.

## Run the experiments

The root Bash scripts execute their listed experiments sequentially and stop on the first failure. They do not run evaluation automatically. Select the entries you need before launching a full model grid.

### GLiNER: zero-shot and fine-tuned

Run zero-shot inference, training, and fine-tuned inference for all three configured models:

```bash
bash runGliner.sh
```

For just the SauerkrautLM experiments reported in the paper:

```bash
python src/entityRecognizer/GLiNEREntityRecognizer.py \
  --config scripts/configs/EntityRecognizer/GLiNEREntityRecognizer/SauerkrautLM_zero_shot.yaml
python src/entityRecognizer/GLiNEREntityRecognizer.py \
  --config scripts/configs/EntityRecognizer/GLiNEREntityRecognizer/SauerkrautLM_train.yaml
python src/entityRecognizer/GLiNEREntityRecognizer.py \
  --config scripts/configs/EntityRecognizer/GLiNEREntityRecognizer/SauerkrautLM_finetuned_inference.yaml
```

Use the module entry point above for training: `run_gliner_entity_recognizer.py` invokes inference only. The training YAML controls the number of steps, learning rates, batch size, and validation interval. The supplied inference threshold is 0.5. Models are saved under `checkpoints/GLiNEREntityRecognizer/`; predictions go to `predictions/GLiNEREntityRecognizer/<experiment>/raw/`.

### LLM zero-shot

Run both Azure models and all three Ollama Gemma sizes:

```bash
bash runLLM.sh
```

Run one configuration instead:

```bash
python run_llm_entity_recognizer.py \
  --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/azure_gpt-4o-mini.yaml
```

Use an `ollama_gemma-<size>.yaml` or `huggingface_gemma-<size>.yaml` configuration for local inference, where `<size>` is `4b`, `12b`, or `27b`. Predictions are saved in `predictions/LLMEntityRecognizer/raw/`. Despite the directory name, the default `raw: false` produces parsed entity annotations ready for evaluation; `raw: true` saves response text instead.

### LLM few-shot: fixed and semantic

Run both selection strategies across the five Azure/Ollama models:

```bash
bash runFewshot.sh
```

Run either strategy individually:

```bash
python run_llm_entity_recognizer.py \
  --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/ollama_gemma-4b_fewshot_fixed.yaml
python run_llm_entity_recognizer.py \
  --config scripts/configs/EntityRecognizer/LLMEntityRecognizer/ollama_gemma-4b_fewshot_semantic.yaml
```

Both use `few_shot.k: 4` and `data/training.json`. Fixed selection reuses a seeded example ordering; semantic selection retrieves examples for each input and caches training embeddings. Matching record IDs and, with `exclude_same_text: true`, identical normalized texts are excluded from the candidate examples. Outputs share the LLM prediction directory, with filenames identifying the strategy and example count. Equivalent Hugging Face configs are also included.

## Evaluation

All approaches use `src/evaluate/evaluate_entityRecognizer.py`. It reports macro and micro precision, recall, and F1 for **strict** matching (span and label) and **span-only** matching. Use tolerance 0 for exact boundaries. The batch settings also discard illegal labels and resolve overlapping predictions by greedily keeping the longest spans.

Evaluate the zero-shot and few-shot LLM outputs:

```bash
python src/evaluate/evaluate_entityRecognizer.py \
  --predictions_path predictions/LLMEntityRecognizer/raw \
  --ground_truth_path data/test.json \
  --output_path evaluation/LLMEntityRecognizer \
  --tolerance 0 \
  --ignore_illegal_labels \
  --resolveOverlappingEntities
```

For a GLiNER experiment, use its actual prediction directory, for example `--predictions_path predictions/GLiNEREntityRecognizer/SauerkrautLM_finetuned/raw`, and a separate output directory. To evaluate just one model, supply a directory containing only that model's prediction JSON.

**Batch evaluation:** `batchEval_NER.sh` evaluates both families, merges their CSV files, and creates a summary table. The evaluator does not scan subdirectories, whereas the supplied inference configs write into nested `raw/` directories. Before invoking the batch script, collect copies of the generated entity files at the two locations it expects:

```bash
for directory in predictions/GLiNEREntityRecognizer/*/raw; do
  cp "$directory"/entities_*.json predictions/GLiNEREntityRecognizer/
done
cp predictions/LLMEntityRecognizer/raw/entities_*.json predictions/LLMEntityRecognizer/

bash batchEval_NER.sh
```

This assumes predictions exist for both model families; use direct evaluation for a partial run. Keep these directories limited to the parsed predictions you intend to compare. Refresh the copies after rerunning inference.

The batch produces per-family CSVs under `evaluation/GLiNEREntityRecognizer/` and `evaluation/LLMEntityRecognizer/`, plus:

```text
evaluation/MergedEntityRecognizer/results_0_overlap-longest.csv
evaluation/MergedEntityRecognizer/results_0_overlap-longest_parsed.csv
```

The parsed file is semicolon-separated and contains `BaseMethod`, `AdvancedMethod`, `ModelName`, `MacroPrecision`, `MacroRecall`, and `MacroF1`. Its model-name mappings are filename-based and may need updating for new runs. 