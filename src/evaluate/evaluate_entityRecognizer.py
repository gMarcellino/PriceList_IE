import json
from argparse import ArgumentParser
import os
import sys
import pandas as pd

# Add workspace root to Python path for relative imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, workspace_root)

from src.utils.utils import ENTITY_LABEL_LIST as LEGAL_ENTITY_LABELS

parser = ArgumentParser()
parser.add_argument('--predictions_path', type=str, default='C:/Users/marti/Desktop/DEESCALATE/DeescIE/predictions/LLMEntityRecognizer/raw', help="Path to the directory containing JSON files with predicted entities")
parser.add_argument('--output_path', type=str, default='C:/Users/marti/Desktop/DEESCALATE/DeescIE/evaluation/entityRecognizer/', help="Path to the output directory for evaluation results")
parser.add_argument('--ground_truth_path', type=str, default='C:/Users/marti/Desktop/DEESCALATE/DeescIE/data/annotations/parsed/entities.json', help="Path to the JSON file containing the ground truth entities")
parser.add_argument('--tolerance', type=int, default=0, help="The number of characters by which the predicted entity boundaries can deviate from the ground truth boundaries while still being considered a match (default is 0, meaning exact match)")
parser.add_argument('--evaluation_mode', type=str, default='both', choices=['strict', 'span', 'both'], help="Evaluation mode: 'strict' (span + label), 'span' (span-only), or 'both' (default)")
parser.add_argument('--ignore_illegal_labels', action='store_true', help="Whether to ignore entities with illegal labels instead of raising an error (default is False)")
parser.add_argument('--entityMapper', action='store_true', help="Whether to evaluate entities as coming from the entity mapper module instead of entity recognizer (default is False)")
parser.add_argument('--revisions_file', type=str, default=None, help="Optional path to a JSON file containing reviewed predictions marked as 'correct', 'ignore', or 'wrong'")

VALID_REVISION_VALUES = {'correct', 'ignore', 'wrong'}
OVERLAP_RESOLUTION_LONGEST = 'longest'

def matches_ground_truth_within_tolerance(predicted_entry, ground_truth_entries, tolerance, include_label=True):
    """
    Check if a predicted entity matches any ground truth entity within the tolerance threshold.
    
    :param predicted_entry: Tuple (start_idx, end_idx, location, text_span, label) for strict or (start_idx, end_idx, location, text_span) for span-only
    :param ground_truth_entries: List of ground truth entity tuples
    :param tolerance: Maximum character offset deviation allowed
    :param include_label: If True, label must match; if False, only span is compared
    :return: Tuple (match_found: bool, matched_gt_index: int or None)
    """
    if include_label:
        pred_start, pred_end, pred_location, pred_text, pred_label = predicted_entry
        for idx, gt_entry in enumerate(ground_truth_entries):
            gt_start, gt_end, gt_location, gt_text, gt_label = gt_entry
            if (pred_label == gt_label and
                abs(pred_start - gt_start) <= tolerance and
                abs(pred_end - gt_end) <= tolerance):
                return True, idx
    else:
        # Span-only mode: ignore label, return the index of the first matching GT entity
        pred_start, pred_end, pred_location, pred_text = predicted_entry
        for idx, gt_entry in enumerate(ground_truth_entries):
            gt_start, gt_end, gt_location, gt_text = gt_entry
            if (abs(pred_start - gt_start) <= tolerance and
                abs(pred_end - gt_end) <= tolerance):
                return True, idx  # Return the index of matched GT entity
    return False, None


def matches_revision_within_tolerance(predicted_entity, revision, tolerance):
    """Return whether a prediction refers to a reviewed entity."""
    return (
        predicted_entity['label'] == revision['label'] and
        abs(predicted_entity['start_idx'] - revision['start_idx']) <= tolerance and
        abs(predicted_entity['end_idx'] - revision['end_idx']) <= tolerance
    )


def find_matching_revision(predicted_entity, revisions, used_revision_indices, tolerance):
    """Find the closest unused revision matching a prediction within tolerance."""
    candidates = []
    for revision_idx, revision in enumerate(revisions):
        if revision_idx in used_revision_indices:
            continue
        if matches_revision_within_tolerance(predicted_entity, revision, tolerance):
            boundary_distance = (
                abs(predicted_entity['start_idx'] - revision['start_idx']) +
                abs(predicted_entity['end_idx'] - revision['end_idx'])
            )
            candidates.append((boundary_distance, revision_idx))

    if not candidates:
        return None
    return min(candidates)[1]


def spans_overlap(first_entity, second_entity):
    """Return whether two inclusive entity spans overlap."""
    return (
        first_entity['start_idx'] <= second_entity[1] and
        second_entity[0] <= first_entity['end_idx']
    )


def find_ground_truth_match(predicted_entry, ground_truth_entries, tolerance,
                            include_label, excluded_indices=None):
    """Find a non-excluded ground-truth match and return its index."""
    excluded_indices = excluded_indices or set()
    pred_start, pred_end = predicted_entry[0], predicted_entry[1]
    pred_label = predicted_entry[4] if include_label else None

    for gt_idx, gt_entry in enumerate(ground_truth_entries):
        if gt_idx in excluded_indices:
            continue
        if include_label and pred_label != gt_entry[4]:
            continue
        if (abs(pred_start - gt_entry[0]) <= tolerance and
                abs(pred_end - gt_entry[1]) <= tolerance):
            return gt_idx
    return None


def prediction_entities_overlap(first_entity, second_entity):
    """Return whether two inclusive prediction spans overlap in the same location."""
    return (
        first_entity['location'] == second_entity['location'] and
        first_entity['start_idx'] <= second_entity['end_idx'] and
        second_entity['start_idx'] <= first_entity['end_idx']
    )


def resolve_overlaps_longest(entities):
    """Greedily retain the longest non-overlapping prediction entities."""
    ranked_entities = sorted(
        enumerate(entities),
        key=lambda item: (
            -(item[1]['end_idx'] - item[1]['start_idx'] + 1),
            item[1]['start_idx'],
            item[1]['end_idx'],
            item[0],
        ),
    )

    retained = []
    for original_idx, candidate in ranked_entities:
        if any(prediction_entities_overlap(candidate, selected) for _, selected in retained):
            continue
        retained.append((original_idx, candidate))

    # Resolution priority must not reorder the predictions that survive.
    return [entity for _, entity in sorted(retained, key=lambda item: item[0])]


OVERLAP_RESOLUTION_STRATEGIES = {
    OVERLAP_RESOLUTION_LONGEST: resolve_overlaps_longest,
}


def resolve_overlapping_entities(entities, strategy):
    """Resolve prediction overlaps using a registered strategy."""
    try:
        resolver = OVERLAP_RESOLUTION_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f'Unknown overlap resolution strategy {strategy!r}; expected one of '
            f'{sorted(OVERLAP_RESOLUTION_STRATEGIES)}'
        )
    return resolver(entities)


parser.add_argument(
    '--resolveOverlappingEntities',
    action='store_true',
    help='Resolve overlapping prediction entities before evaluation (default strategy: longest)',
)
parser.add_argument(
    '--overlapResolutionStrategy',
    type=str,
    default=OVERLAP_RESOLUTION_LONGEST,
    choices=sorted(OVERLAP_RESOLUTION_STRATEGIES),
    help='Strategy used by --resolveOverlappingEntities (default: longest)',
)


def normalize_revisions(revisions):
    """Validate and normalize the revision JSON structure."""
    if not isinstance(revisions, dict):
        raise ValueError('The revisions file must contain a JSON object keyed by document ID')

    normalized = {}
    for pmid, entries in revisions.items():
        if not isinstance(entries, list):
            raise ValueError(f'{pmid} - Revisions must be provided as a list')

        normalized[str(pmid)] = []
        for revision in entries:
            try:
                revision_value = str(revision['revision']).strip().lower()
                normalized_revision = {
                    'start_idx': int(revision['start_idx']),
                    'end_idx': int(revision['end_idx']),
                    'label': str(revision['label']),
                    'revision': revision_value,
                }
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f'{pmid} - Invalid revision entry; expected start_idx, end_idx, '
                    f'label, and revision: {revision}'
                )

            if revision_value not in VALID_REVISION_VALUES:
                raise ValueError(
                    f'{pmid} - Invalid revision value {revision_value!r}; expected one of '
                    f'{sorted(VALID_REVISION_VALUES)}'
                )
            normalized[str(pmid)].append(normalized_revision)

    return normalized


def eval_predicted_entities(predictions_path, ground_truth, tolerance=0, evaluation_mode='both', ignore_illegal_labels=False, revisions=None, overlap_resolution_strategy=None):
    """
    Evaluate the predicted entities against the ground truth entities and compute precision, recall, and F1-score
    at both micro and macro levels for both strict and span-only modes.

    :param predictions_path: path to the JSON file containing the predicted entities, in GBIE format
    :param ground_truth: dict containing the ground truth entities, structured as the ground truth JSON file
    :param tolerance: the number of characters by which the predicted entity boundaries can deviate from the ground truth boundaries while still being considered a match (default is 0, meaning exact match)
    :param evaluation_mode: 'strict' for span+label matching, 'span' for span-only matching, or 'both' for both modes
    :param ignore_illegal_labels: Whether to ignore entities with illegal labels instead of raising an error (default is False)
    :param revisions: optional dict of reviewed predictions, keyed by document ID
    :param overlap_resolution_strategy: optional registered strategy for resolving overlapping predictions
    :return: a dict containing evaluation metrics for the requested modes
    """
    try:
        with open(predictions_path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
    except OSError:
        raise OSError(f'Error in opening the specified json file: {predictions_path}')

    # Build ground truth entity structures
    ground_truth_entities_strict = dict()  # For strict (span + label) evaluation
    ground_truth_entities_span = dict()    # For span-only evaluation
    count_annotated_entities_per_label = {}
    
    for pmid, article in ground_truth.items():
        if pmid not in ground_truth_entities_strict:
            ground_truth_entities_strict[pmid] = []
            ground_truth_entities_span[pmid] = []
        
        for entity in article.get('entities', []):
            start_idx = int(entity["start_idx"])
            end_idx = int(entity["end_idx"])
            location = str(entity.get("location", "placeholder"))
            text_span = str(entity["text_span"])
            label = str(entity["label"])

            if label not in LEGAL_ENTITY_LABELS:
                if ignore_illegal_labels:
                    print(f'{pmid} - Illegal label {label} for entity: {entity}')
                    continue
                else:
                    raise ValueError(f'{pmid} - Illegal label {label} for entity: {entity}')
            
            # For strict evaluation: include label
            entry_strict = (start_idx, end_idx, location, text_span, label)
            ground_truth_entities_strict[pmid].append(entry_strict)
            
            # For span-only evaluation: exclude label
            entry_span = (start_idx, end_idx, location, text_span)
            ground_truth_entities_span[pmid].append(entry_span)
            
            if label not in count_annotated_entities_per_label:
                count_annotated_entities_per_label[label] = 0
            count_annotated_entities_per_label[label] += 1

    use_revisions = revisions is not None
    if use_revisions:
        revisions = normalize_revisions(revisions)

    # Initialize counters for predictions
    count_predicted_entities_per_label_strict = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_true_positives_per_label_strict = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    
    count_predicted_entities_per_label_span = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    count_true_positives_per_label_span = {label: 0 for label in list(count_annotated_entities_per_label.keys())}
    
    # A ground-truth entity may be matched at most once in each evaluation mode.
    matched_gt_indices_strict_per_pmid = {pmid: set() for pmid in ground_truth_entities_strict.keys()}
    matched_gt_indices_per_pmid = {pmid: set() for pmid in ground_truth_entities_span.keys()}

    # Process predictions
    for pmid in predictions.keys():
        try:
            entities = predictions[pmid].get('entities', [])
        except (KeyError, TypeError):
            raise KeyError(f'{pmid} - Not able to find or access field "entities" within article')
        
        parsed_entities = []
        used_revision_indices = set()
        document_revisions = revisions.get(str(pmid), []) if use_revisions else []

        for entity in entities:
            try:
                start_idx = int(entity["start_idx"])
                end_idx = int(entity["end_idx"])
                location = str(entity.get("location", "placeholder"))
                text_span = str(entity["text_span"])
                label = str(entity["label"])
            except KeyError:
                raise KeyError(f'{pmid} - Not able to find one or more of the expected fields for entity: {entity}')
            
            if label not in LEGAL_ENTITY_LABELS:
                if ignore_illegal_labels:
                    print(f'{pmid} - Illegal label {label} for entity: {entity}')
                    continue
                else:
                    raise ValueError(f'{pmid} - Illegal label {label} for entity: {entity}')

            parsed_entity = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'location': location,
                'text_span': text_span,
                'label': label,
                'revision': None,
            }
            parsed_entities.append(parsed_entity)

        if overlap_resolution_strategy is not None:
            parsed_entities = resolve_overlapping_entities(
                parsed_entities, overlap_resolution_strategy
            )

        # Resolve overlaps before associating revisions, so a discarded prediction cannot
        # consume the revision intended for a retained one.
        if use_revisions:
            for parsed_entity in parsed_entities:
                revision_idx = find_matching_revision(
                    parsed_entity, document_revisions, used_revision_indices, tolerance
                )
                if revision_idx is not None:
                    used_revision_indices.add(revision_idx)
                    parsed_entity['revision'] = document_revisions[revision_idx]['revision']

        if not use_revisions:
            # Preserve the original evaluation behavior exactly when no revisions file is supplied.
            entities_to_evaluate = parsed_entities
            reserved_gt_indices = set()
        else:
            entities_to_evaluate = parsed_entities
            reserved_gt_indices = set()

            # A reviewed correct prediction amends the annotation set. If it overlaps an
            # existing annotation, that annotation is reserved as its single hit. Otherwise,
            # it is a new synthetic annotation, keeping both precision and recall bounded.
            gt_entries_for_pmid = ground_truth_entities_strict.get(str(pmid), [])
            for parsed_entity in entities_to_evaluate:
                if parsed_entity['revision'] != 'correct':
                    continue

                overlapping_gt_idx = next(
                    (
                        gt_idx for gt_idx, gt_entry in enumerate(gt_entries_for_pmid)
                        if gt_idx not in reserved_gt_indices and spans_overlap(parsed_entity, gt_entry)
                    ),
                    None,
                )
                label = parsed_entity['label']
                count_annotated_entities_per_label.setdefault(label, 0)

                if overlapping_gt_idx is None:
                    count_annotated_entities_per_label[label] += 1
                else:
                    reserved_gt_indices.add(overlapping_gt_idx)
                    gt_label = gt_entries_for_pmid[overlapping_gt_idx][4]
                    if gt_label != label:
                        count_annotated_entities_per_label[gt_label] -= 1
                        count_annotated_entities_per_label[label] += 1

            matched_gt_indices_strict_per_pmid.setdefault(str(pmid), set()).update(reserved_gt_indices)
            matched_gt_indices_per_pmid.setdefault(str(pmid), set()).update(reserved_gt_indices)

        for parsed_entity in entities_to_evaluate:
            start_idx = parsed_entity['start_idx']
            end_idx = parsed_entity['end_idx']
            location = parsed_entity['location']
            text_span = parsed_entity['text_span']
            label = parsed_entity['label']
            revision_value = parsed_entity['revision']

            # Ignored reviewed predictions are removed before any counters are updated.
            if revision_value == 'ignore':
                continue

            if use_revisions:
                count_predicted_entities_per_label_strict.setdefault(label, 0)
                count_true_positives_per_label_strict.setdefault(label, 0)
                count_predicted_entities_per_label_span.setdefault(label, 0)
                count_true_positives_per_label_span.setdefault(label, 0)

            # For strict evaluation (span + label)
            if evaluation_mode in ['strict', 'both']:
                if label in count_predicted_entities_per_label_strict:
                    count_predicted_entities_per_label_strict[label] += 1
                
                entry_strict = (start_idx, end_idx, location, text_span, label)
                if revision_value == 'correct':
                    count_true_positives_per_label_strict[label] += 1
                elif revision_value != 'wrong':
                    matched_gt_idx = find_ground_truth_match(
                        entry_strict,
                        ground_truth_entities_strict.get(str(pmid), []),
                        tolerance,
                        include_label=True,
                        excluded_indices=matched_gt_indices_strict_per_pmid.get(str(pmid), set()),
                    )
                    if matched_gt_idx is not None:
                        count_true_positives_per_label_strict[label] += 1
                        matched_gt_indices_strict_per_pmid.setdefault(str(pmid), set()).add(matched_gt_idx)
            
            # For span-only evaluation (span only, ignore label)
            if evaluation_mode in ['span', 'both']:
                if label in count_predicted_entities_per_label_span:
                    count_predicted_entities_per_label_span[label] += 1
                
                entry_span = (start_idx, end_idx, location, text_span)
                if revision_value == 'correct':
                    count_true_positives_per_label_span[label] += 1
                elif revision_value != 'wrong':
                    matched_gt_idx = find_ground_truth_match(
                        entry_span,
                        ground_truth_entities_span.get(str(pmid), []),
                        tolerance,
                        include_label=False,
                        excluded_indices=matched_gt_indices_per_pmid.get(str(pmid), set()),
                    )
                    if matched_gt_idx is not None:
                        count_true_positives_per_label_span[label] += 1
                        matched_gt_indices_per_pmid.setdefault(str(pmid), set()).add(matched_gt_idx)

    results = {}

    # Calculate metrics for strict mode if requested
    if evaluation_mode in ['strict', 'both']:
        count_annotated_entities = sum(count_annotated_entities_per_label.values())
        count_predicted_entities = sum(count_predicted_entities_per_label_strict.values())
        count_true_positives = sum(count_true_positives_per_label_strict.values())

        micro_precision_strict = count_true_positives / (count_predicted_entities + 1e-10)
        micro_recall_strict = count_true_positives / (count_annotated_entities + 1e-10)
        micro_f1_strict = 2 * ((micro_precision_strict * micro_recall_strict) / (micro_precision_strict + micro_recall_strict + 1e-10))

        precision_strict, recall_strict, f1_strict = 0, 0, 0
        n_labels = len(count_annotated_entities_per_label)
        if n_labels > 0:
            for label in list(count_annotated_entities_per_label.keys()):
                current_precision = count_true_positives_per_label_strict[label] / (count_predicted_entities_per_label_strict[label] + 1e-10)
                current_recall = count_true_positives_per_label_strict[label] / (count_annotated_entities_per_label[label] + 1e-10)
                
                precision_strict += current_precision
                recall_strict += current_recall
                f1_strict += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))
            
            precision_strict = precision_strict / n_labels
            recall_strict = recall_strict / n_labels
            f1_strict = f1_strict / n_labels
        
        results['strict_precision'] = precision_strict
        results['strict_recall'] = recall_strict
        results['strict_f1'] = f1_strict
        results['strict_micro_precision'] = micro_precision_strict
        results['strict_micro_recall'] = micro_recall_strict
        results['strict_micro_f1'] = micro_f1_strict

    # Calculate metrics for span-only mode if requested
    if evaluation_mode in ['span', 'both']:
        count_annotated_entities = sum(count_annotated_entities_per_label.values())
        count_predicted_entities = sum(count_predicted_entities_per_label_span.values())
        count_true_positives = sum(count_true_positives_per_label_span.values())

        micro_precision_span = count_true_positives / (count_predicted_entities + 1e-10)
        micro_recall_span = count_true_positives / (count_annotated_entities + 1e-10)
        micro_f1_span = 2 * ((micro_precision_span * micro_recall_span) / (micro_precision_span + micro_recall_span + 1e-10))

        precision_span, recall_span, f1_span = 0, 0, 0
        n_labels = len(count_annotated_entities_per_label)
        if n_labels > 0:
            for label in list(count_annotated_entities_per_label.keys()):
                current_precision = count_true_positives_per_label_span[label] / (count_predicted_entities_per_label_span[label] + 1e-10)
                current_recall = count_true_positives_per_label_span[label] / (count_annotated_entities_per_label[label] + 1e-10)
                
                precision_span += current_precision
                recall_span += current_recall
                f1_span += 2 * ((current_precision * current_recall) / (current_precision + current_recall + 1e-10))
            
            precision_span = precision_span / n_labels
            recall_span = recall_span / n_labels
            f1_span = f1_span / n_labels
        
        results['span_precision'] = precision_span
        results['span_recall'] = recall_span
        results['span_f1'] = f1_span
        results['span_micro_precision'] = micro_precision_span
        results['span_micro_recall'] = micro_recall_span
        results['span_micro_f1'] = micro_f1_span

    return results


def parse_filename(filename, entity_mapper=False):
    """Parse prediction filename to extract metadata (language, model_source, model_name)."""
    parts = filename.split('_')
    if not entity_mapper:
        if len(parts) >= 4:
            language = parts[1]
            model_source = parts[2]
            model_name = parts[3].replace('.json', '').replace('-Persona', '')
            return language, model_source, model_name
        return 'unknown', 'unknown', 'unknown'
    else:
        if len(parts) >= 6:
            language = parts[1]
            model_source = parts[2]
            model_name = parts[3]
            mapper_model_source = parts[4]
            mapper_model_name = parts[5].replace('.json', '')
            return language, model_source, model_name, mapper_model_source, mapper_model_name
        return 'unknown', 'unknown', 'unknown', 'unknown', 'unknown'

def main():
    args = parser.parse_args()

    try:
        with open(args.ground_truth_path, 'r', encoding='utf-8') as file:
            ground_truth = json.load(file)
    except OSError:
        raise OSError(f'Error in opening the specified json file: {args.ground_truth_path}')

    revisions = None
    if args.revisions_file is not None:
        try:
            with open(args.revisions_file, 'r', encoding='utf-8') as file:
                revisions = json.load(file)
        except OSError:
            raise OSError(f'Error in opening the specified json file: {args.revisions_file}')
    
    results = {}
    for file in os.listdir(args.predictions_path):
        if file.endswith('.json') and not file.endswith('.mapping_stats.json'):
            predictions_file_path = os.path.join(args.predictions_path, file)
            metrics = eval_predicted_entities(
                predictions_file_path,
                ground_truth,
                tolerance=args.tolerance,
                evaluation_mode=args.evaluation_mode,
                ignore_illegal_labels=args.ignore_illegal_labels,
                revisions=revisions,
                overlap_resolution_strategy=(
                    args.overlapResolutionStrategy
                    if args.resolveOverlappingEntities
                    else None
                ),
            )
            if args.entityMapper:
                language, model_source, model_name, mapper_model_source, mapper_model_name = parse_filename(file, entity_mapper=True)
                results[file] = {
                    'language': language,
                    'model_source': model_source,
                    'model_name': model_name,
                    'mapper_model_source': mapper_model_source,
                    'mapper_model_name': mapper_model_name,
                    **metrics  # Unpack all evaluation metrics
                }
            else:
                language, model_source, model_name = parse_filename(file, entity_mapper=False)
                results[file] = {
                    'language': language,
                    'model_source': model_source,
                    'model_name': model_name,
                    **metrics  # Unpack all evaluation metrics
                }
    
    output_path = args.output_path
    if args.entityMapper:
        output_path = output_path.replace('entityRecognizer', 'entityMapper')
    os.makedirs(output_path, exist_ok=True)
    results_df = pd.DataFrame.from_dict(results, orient='index')

    # Generate output filename based on parameters
    mode_suffix = f"_{args.evaluation_mode}" if args.evaluation_mode != 'both' else ""
    overlap_suffix = (
        f'_overlap-{args.overlapResolutionStrategy}'
        if args.resolveOverlappingEntities
        else ''
    )
    output_filename = f'entity_recognizer_eval_results_{args.tolerance}{mode_suffix}{overlap_suffix}.csv'
    sort_column = 'strict_micro_f1' if 'strict_micro_f1' in results_df.columns else 'span_micro_f1'
    results_df.sort_values(by=[sort_column], inplace=True, ascending=False)
    results_df.to_csv(os.path.join(output_path, output_filename), index=False)
    print(f"Evaluation results saved to {os.path.join(output_path, output_filename)}")


if __name__ == "__main__":
    main()
