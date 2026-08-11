#!/usr/bin/env python3
"""Focused regression tests for semantic-ID churn accounting."""

import unittest

import numpy as np

from run_generative_prefix import RQ
from run_wsdm_web_recsys import _prefix_churn_metrics


class ChurnAlignmentTest(unittest.TestCase):
    def test_pure_token_permutation_is_not_genuine_churn(self):
        embeddings = np.asarray([[0.0], [0.1], [9.9], [10.0]], dtype=np.float32)
        source = RQ(1, [2], 1)
        current = RQ(1, [2], 1)
        source.cb = [np.asarray([[0.0], [10.0]], dtype=np.float32)]
        current.cb = [np.asarray([[10.0], [0.0]], dtype=np.float32)]

        churn = _prefix_churn_metrics(source, current, embeddings, 1)

        self.assertEqual(churn["raw"], 1.0)
        self.assertEqual(churn["centroid_aligned"], 0.0)
        self.assertEqual(churn["assignment_aligned"], 0.0)


if __name__ == "__main__":
    unittest.main()
