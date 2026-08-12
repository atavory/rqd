#!/usr/bin/env python3
"""Regression tests for Tier-C retraining-necessity diagnostics."""

import unittest

import numpy as np

from run_generative_prefix import RQ
from run_wsdm_web_recsys import (
    _prefix_drift_diagnostics,
    _task_shift_diagnostics,
)


class TierCDiagnosticsTest(unittest.TestCase):
    def test_prefix_orthogonal_drift_has_zero_xi_and_no_crossing(self):
        rq = RQ(1, [2], 2)
        rq.cb = [np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)]
        source = np.asarray(
            [[-1.2, 0.0], [-0.8, 0.0], [0.8, 0.0], [1.2, 0.0]],
            dtype=np.float32,
        )
        orthogonal_target = source + np.asarray([0.0, 5.0], dtype=np.float32)
        prefix_target = np.asarray(
            [[1.2, 0.0], [0.8, 0.0], [-0.8, 0.0], [-1.2, 0.0]],
            dtype=np.float32,
        )

        orthogonal = _prefix_drift_diagnostics(rq, source, orthogonal_target, 1)
        prefix = _prefix_drift_diagnostics(rq, source, prefix_target, 1)

        self.assertLess(orthogonal["xi_s"], 1e-6)
        self.assertEqual(orthogonal["epsilon_s_temporal"], 0.0)
        self.assertGreater(prefix["xi_s"], 0.99)
        self.assertEqual(prefix["epsilon_s_temporal"], 1.0)

    def test_delta_task_detects_stable_bucket_conditional_shift(self):
        rq = RQ(1, [2], 1)
        rq.cb = [np.asarray([[0.0], [10.0]], dtype=np.float32)]
        embeddings = np.asarray([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)

        diagnostics = _task_shift_diagnostics(
            rq,
            embeddings,
            embeddings,
            [[0, 1, 0, 1]],
            [(0, [0], 2)],
            1,
        )

        self.assertEqual(diagnostics["delta_task_tv_weighted"], 1.0)
        self.assertEqual(diagnostics["delta_task_context_overlap"], 1.0)
        self.assertEqual(diagnostics["delta_task_contexts_common"], 1)


if __name__ == "__main__":
    unittest.main()
