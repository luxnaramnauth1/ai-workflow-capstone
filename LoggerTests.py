import os
import tempfile
import unittest

import logger
from unittests.helpers import remove


class LoggerTests(unittest.TestCase):
    def setUp(self):
        # isolate from production logs: point the logger at a throw-away directory
        self._orig = logger.LOG_DIR
        logger.LOG_DIR = tempfile.mkdtemp(prefix="capstone-logs-")

    def tearDown(self):
        remove(logger.LOG_DIR)
        logger.LOG_DIR = self._orig

    def test_train_log_is_created_with_header(self):
        path = logger.update_train_log("all", "ridge", "1.0", 1.0, 2.0, 3.0, 0.5, test=True)
        self.assertTrue(os.path.exists(path))
        rows = logger.read_log(os.path.basename(path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_name"], "ridge")

    def test_predict_log_appends_rows(self):
        for i in range(3):
            logger.update_predict_log("all", "2018-06-0{}".format(i + 1), 100.0 + i, "ridge", "1.0", 0.01, test=True)
        rows = logger.read_log("predict-test.log")
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["unique_id"] for r in rows], ["1", "2", "3"])

    def test_test_mode_never_writes_production_file(self):
        logger.update_predict_log("all", "2018-06-01", 1.0, "ridge", "1.0", 0.01, test=True)
        self.assertEqual(logger.list_logs(), ["predict-test.log"])

    def test_missing_log_raises(self):
        with self.assertRaises(FileNotFoundError):
            logger.read_log("nope.log")

    def test_path_traversal_is_blocked(self):
        with self.assertRaises(FileNotFoundError):
            logger.read_log("../../etc/passwd")


if __name__ == "__main__":
    unittest.main()
