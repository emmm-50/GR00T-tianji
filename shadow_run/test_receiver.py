"""Regression for receiver failures during a partially processed block."""
import unittest
from unittest.mock import Mock

from shadow_run.run import receive_block


class TestReceiverFailure(unittest.TestCase):
    def test_failure_does_not_count_unprocessed_tail(self):
        block = {"id": "x", "superseded_tail": 0, "timing": {}}
        request = {"mode": "virtual", "targets": [1, 2, 3]}
        receiver = Mock()
        received = [{"index": 0, "mock_receive_ns": 123}]
        receiver.run.return_value = {"samples": received, "failure": "bad IK"}
        with self.assertRaisesRegex(RuntimeError, "bad IK"):
            receive_block(receiver, (block, request))
        self.assertEqual(block["receiver"]["samples"], [{"index": 0, "mock_receive_ns": 123}])
        self.assertEqual(block["receiver"]["failure"], "bad IK")
        self.assertEqual(block["superseded_tail"], 0)


if __name__ == "__main__":
    unittest.main()
