import os
import unittest

import app as app_module
import model
from unittests.helpers import make_tmp_dataset, remove, clean_test_artifacts


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = make_tmp_dataset()
        cls.client = app_module.app.test_client()
        r = cls.client.post("/train", json={"mode": "test", "data_dir": cls.data_dir})
        assert r.status_code == 200, r.get_data(as_text=True)

    @classmethod
    def tearDownClass(cls):
        remove(cls.data_dir)
        clean_test_artifacts(model.MODEL_DIR, os.environ.get("LOG_DIR", "logs"))

    def _predict(self, **overrides):
        body = {"country": "all", "year": 2018, "month": 9, "day": 1, "mode": "test",
                "data_dirs": [self.data_dir]}
        body.update(overrides)
        return self.client.post("/predict", json=body)

    def test_index(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_train_reports_trained_countries(self):
        r = self.client.post("/train", json={"mode": "test", "data_dir": self.data_dir})
        self.assertIn("all", r.get_json()["trained"])

    def test_train_missing_data_dir_is_404(self):
        r = self.client.post("/train", json={"mode": "test", "data_dir": "/no/such/dir"})
        self.assertEqual(r.status_code, 404)

    def test_predict_all_countries(self):
        r = self._predict()
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.get_json()["y_pred"], 0)

    def test_predict_specific_country(self):
        r = self._predict(country="United Kingdom")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["country"], "United Kingdom")

    def test_predict_unknown_country_is_404(self):
        self.assertEqual(self._predict(country="Atlantis").status_code, 404)

    def test_predict_missing_field_is_400(self):
        r = self.client.post("/predict", json={"country": "all", "year": 2018, "mode": "test"})
        self.assertEqual(r.status_code, 400)

    def test_predict_no_body_is_400(self):
        self.assertEqual(self.client.post("/predict").status_code, 400)

    def test_predict_bad_date_is_400(self):
        self.assertEqual(self._predict(month=13).status_code, 400)

    def test_logs_endpoint_lists_and_returns_test_logs(self):
        self._predict()
        listing = self.client.get("/logs").get_json()["files"]
        self.assertIn("predict-test.log", listing)
        rows = self.client.get("/logs/predict-test.log").get_json()["rows"]
        self.assertGreaterEqual(len(rows), 1)

    def test_unknown_log_is_404(self):
        self.assertEqual(self.client.get("/logs/nope.log").status_code, 404)

    def test_monitor_endpoint(self):
        r = self.client.get("/monitor?mode=test&data_dirs={}".format(self.data_dir))
        self.assertEqual(r.status_code, 200)
        report = r.get_json()
        self.assertIn("all", report)
        for key in ["recent_rmse", "baseline_recent_rmse", "performance_drift", "novelty_rate"]:
            self.assertIn(key, report["all"])


if __name__ == "__main__":
    unittest.main()
