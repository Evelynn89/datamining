import ast
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import time

import nbformat
import numpy as np
import pandas as pd


def notebook_function(filename, function_name, namespace):
    notebook = nbformat.read(Path(__file__).parent / 'code' / filename, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
                return namespace[function_name]
    raise LookupError(function_name)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth = pd.DataFrame({
            'customer_key': [1, 1, 2], 'article_num': [10, 30, 40], 'label': [1, 1, 1],
        })
        self.candidates = pd.DataFrame({
            'customer_key': [1, 1, 1], 'article_num': [10, 20, 30], 'score': [3., 2., 1.],
        })
        namespace = {'pd': pd, 'np': np, 'json': json, 'target_truth': self.truth}
        self.model_metric = notebook_function(
            '04_final_candidate_feature_model.ipynb', 'evaluate_candidates', namespace,
        )
        self.mining_metric = notebook_function(
            '03_pattern_mining_features.ipynb', 'evaluate_candidates', namespace,
        )

    def test_multiple_hits_and_missing_user(self):
        expected_map = ((1 + 2 / 3) / 2) / 2
        model = self.model_metric('test', self.candidates, users=np.array([1, 2], dtype=np.uint64))
        mining = self.mining_metric('test', self.candidates, self.truth, 2, show_output=False)
        for result in [model, mining]:
            self.assertAlmostEqual(result['map@12'], expected_map)
            self.assertAlmostEqual(result['recall@12'], 0.5)
            self.assertAlmostEqual(result['hit_rate@12'], 0.5)

    def test_no_hits(self):
        candidates = self.candidates.assign(article_num=[70, 80, 90])
        result = self.model_metric('test', candidates, users=[1, 2])
        self.assertEqual(result['map@12'], 0)

    def test_empty_predictions_keep_truth_users(self):
        result = self.model_metric('test', self.candidates.iloc[:0], users=[1, 2])
        self.assertEqual(result['map@12'], 0)
        self.assertEqual(result['evaluated_users'], 2)

    def test_duplicate_predictions_are_not_double_counted(self):
        candidates = pd.concat([self.candidates, self.candidates.iloc[[0]]], ignore_index=True)
        expected = ((1 + 2 / 3) / 2) / 2
        result = self.model_metric('test', candidates, users=[1, 2])
        self.assertAlmostEqual(result['map@12'], expected)
        mining = self.mining_metric('test', candidates, self.truth, 2, show_output=False)
        self.assertAlmostEqual(mining['map@12'], expected)

    def test_rank_truncation_direction(self):
        function = notebook_function(
            '04_final_candidate_feature_model.ipynb', 'top_k_by_user', {'pd': pd},
        )
        candidates = self.candidates.assign(rank=[1, 2, 3])
        self.assertEqual(function(candidates, 'rank', top_n=1, ascending=True)['article_num'].tolist(), [10])
        self.assertEqual(function(candidates, 'score', top_n=1)['article_num'].tolist(), [10])

    def test_unsigned_customer_keys_remain_distinct(self):
        keys = np.array([2**64 - 1, 2**64 - 2], dtype=np.uint64)
        truth = pd.DataFrame({'customer_key': keys, 'article_num': [10, 20], 'label': [1, 1]})
        candidates = pd.DataFrame({'customer_key': keys, 'article_num': [10, 99], 'score': [1., 1.]})
        metric = notebook_function(
            '04_final_candidate_feature_model.ipynb', 'evaluate_candidates',
            {'np': np, 'pd': pd, 'json': json, 'target_truth': truth},
        )
        result = metric('test', candidates, users=keys)
        self.assertEqual(result['evaluated_users'], 2)
        self.assertEqual(result['map@12'], 0.5)

    def test_baseline_fills_short_lists_without_duplicates(self):
        notebook = nbformat.read(Path(__file__).parent / 'code' / '04_final_candidate_feature_model.ipynb', as_version=4)
        baseline_source = next(
            cell.source for cell in notebook.cells
            if cell.cell_type == 'code' and 'baseline_started = time.time()' in cell.source
        )
        history = pd.DataFrame({
            'customer_key': [8, 8] + [9] * 15,
            'article_num': [10, 20] + list(range(10, 160, 10)),
            't_dat': pd.to_datetime(['2020-09-15'] * 17),
        })
        rules = pd.DataFrame({
            'antecedent_num': [10], 'consequent_num': [20], 'pair_count': [3],
            'interest_score': [1.], 'confidence': [0.5], 'lift': [2.],
        })
        namespace = {
            'pd': pd, 'np': np, 'json': json, 'time': time,
            'history': history, 'target_users': np.array([8, 9, 18], dtype=np.uint64),
            'target_user_set': {8, 9, 18}, 'TARGET_START': pd.Timestamp('2020-09-16'),
            'target_truth': self.truth.assign(customer_key=[8, 8, 9]), 'MINING_DIR': Path('.'),
        }
        namespace['evaluate_candidates'] = notebook_function(
            '04_final_candidate_feature_model.ipynb', 'evaluate_candidates', namespace,
        )
        namespace['top_k_by_user'] = notebook_function(
            '04_final_candidate_feature_model.ipynb', 'top_k_by_user', namespace,
        )
        with TemporaryDirectory() as directory:
            namespace.update(FINAL_DIR=Path(directory), TABLE_DIR=Path(directory))
            with patch.object(pd, 'read_parquet', return_value=rules):
                exec(baseline_source, namespace)
        for name in ['baseline1_candidates', 'baseline2_candidates']:
            candidates = namespace[name]
            self.assertTrue(candidates.groupby('customer_key').size().eq(12).all())
            self.assertFalse(candidates.duplicated(['customer_key', 'article_num']).any())


if __name__ == '__main__':
    unittest.main()
