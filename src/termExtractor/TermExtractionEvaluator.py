from sklearn.metrics import precision_recall_fscore_support

###########################
# TermExtractionEvaluator #
###########################
class TermExtractionEvaluator:
    """
    Evaluates term extraction quality by comparing predicted entities against ground-truth annotations at the span level (exact match).

    This class is API-compatible with HFTermExtractor.TermExtractionEvaluator and can be used interchangeably to benchmark LLM-based and BIO-tagger-based predictions against the same ground-truth files.

    Each entity is represented as a 4-tuple: (start_idx, end_idx, location, text_span)

    Precision, recall, and F1 are computed with binary averaging.
    """

    def evaluate(self, predictions: dict, ground_truth: dict) -> dict:
        """
        Computes precision, recall, and F1 across all papers in the dataset.

        :param predictions:  Dict mapping paper IDs to content with predicted entities.
        :param ground_truth: Dict mapping paper IDs to content with gold entities.
        :return: A dict with keys 'precision', 'recall', and 'f1' (floats 0–1).
        """
        gt_sets   = self._build_entity_sets(ground_truth)
        pred_sets = self._build_entity_sets(predictions)

        y_true = []
        y_pred = []

        # Union of all IDs ensures papers with only predictions or only gold
        # are included in the evaluation, penalising accordingly.
        all_ids = set(gt_sets.keys()) | set(pred_sets.keys())
        for paper_id in all_ids:
            gt_entities   = gt_sets.get(paper_id, set())
            pred_entities = pred_sets.get(paper_id, set())
            all_entities  = gt_entities | pred_entities

            for entity in all_entities:
                y_true.append(1 if entity in gt_entities   else 0)
                y_pred.append(1 if entity in pred_entities else 0)

        if not y_true:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        return {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
        }

    @staticmethod
    def _build_entity_sets(data: dict) -> dict:
        """
        Converts entity lists to sets of comparable tuples for fast lookup.

        :param data: Dict mapping paper IDs to content dicts with 'entities' lists.
        :return: Dict mapping paper IDs to sets of (start_idx, end_idx, location, text_span) tuples.
        """
        entity_sets = {}
        for paper_id, content in data.items():
            entity_sets[paper_id] = {
                (
                    entity["start_idx"],
                    entity["end_idx"],
                    entity["location"],
                    entity["text_span"],
                )
                for entity in content.get("entities", [])
            }
        return entity_sets
