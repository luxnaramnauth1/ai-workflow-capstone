import os
import unittest

import joblib

import model
from unittests.helpers import make_tmp_dataset, remove, clean_test_artifacts


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = make_tmp_dataset()
        cls.summaries = model.model_train(cls.data_dir, test=True)   # test=True -> test-* files only

    @classmethod
    def tearDownClass(cls):
        remove(cls.data_dir)
        clean_test_artifacts(model.MODEL_DIR, os.environ.get("LOG_DIR", "logs"))

    def test_trains_all_plus_top_countries(self):
        names = [s["country"] for s in self.summaries]
        self.assertEqual(names[0], "all")
        self.assertEqual(len(names), 3)   # all + 2 countries in test mode

    def test_multiple_models_were_compared(self):
        scores = self.summaries[0]["scores"]
        for name in ["baseline", "ridge", "random_forest", "gradient_boosting"]:
            self.assertIn(name, scores)
            self.assertGreaterEqual(scores[name]["holdout_rmse"], 0)

    def test_model_files_are_saved_as_test_files(self):
        files = os.listdir(model.MODEL_DIR)
        self.assertTrue(any(f.startswith("test-sl-all") for f in files))

    def test_load_only_returns_test_models_in_test_mode(self):
        models = model.model_load(test=True)
        self.assertIn("all", models)
        self.assertIn("model", models["all"])

    def test_predict_all_countries(self):
        r = model.model_predict("all", 2018, 9, 1, data_dirs=[self.data_dir], test=True)
        self.assertIsInstance(r["y_pred"], float)
        self.assertGreater(r["y_pred"], 0)
        self.assertIsNotNone(r["actual"])

    def test_predict_single_country(self):
        country = [s["country"] for s in self.summaries if s["country"] != "all"][0]
        r = model.model_predict(country, 2018, 9, 1, data_dirs=[self.data_dir], test=True)
        self.assertEqual(r["country"], country)

    def test_predict_unknown_country_raises(self):
        with self.assertRaises(KeyError):
            model.model_predict("Atlantis", 2018, 9, 1, data_dirs=[self.data_dir], test=True)

    def test_predict_date_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            model.model_predict("all", 2010, 1, 1, data_dirs=[self.data_dir], test=True)

    def test_predict_invalid_date_raises(self):
        with self.assertRaises(ValueError):
            model.model_predict("all", 2018, 13, 45, data_dirs=[self.data_dir], test=True)

    def test_baseline_predict_scales_last_28_days(self):
        import pandas as pd
        X = pd.DataFrame({"revenue_prev_28": [28.0]})
        self.assertAlmostEqual(model.NaiveBaseline().predict(X)[0], 30.0)


if __name__ == "__main__":
    unittest.main()
