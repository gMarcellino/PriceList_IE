#!/bin/bash
set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="evaluation"
mkdir -p "$RESULTS_DIR"
evaluation_paths=""

TOLERANCE="0"

echo "Starting batch evaluation for NER models with tolerance: ${TOLERANCE}"
for f in \
  "GLiNEREntityRecognizer" \
  "LLMEntityRecognizer"
do
  echo "Running evaluation for: $f"
  python -u src/evaluate/evaluate_entityRecognizer.py \
    --predictions_path "predictions/${f}" \
    --output_path "$RESULTS_DIR/${f}" \
    --ground_truth_path "data/test.json" \
    --tolerance "${TOLERANCE}" \
    --ignore_illegal_labels \
    --resolveOverlappingEntities
  evaluation_paths+=" evaluation/${f}/entity_recognizer_eval_results_${TOLERANCE}_overlap-longest.csv"
done
echo "Merging evaluation results..."
python src/evaluate/merge_evaluations.py \
  --evaluation_paths ${evaluation_paths# } \
  --output_path evaluation/MergedEntityRecognizer/results_${TOLERANCE}_overlap-longest.csv
echo "Parsing merged evaluation results..."
python src/evaluate/parse_merged_evaluations.py \
  --input_path evaluation/MergedEntityRecognizer/results_${TOLERANCE}_overlap-longest.csv \
  --output_path evaluation/MergedEntityRecognizer/results_${TOLERANCE}_overlap-longest_parsed.csv  
echo "Done."