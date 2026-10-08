import unittest

import pandas as pd

import ingestion
from unittests.helpers import make_tmp_dataset, remove


class IngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = make_tmp_dataset()
        cls.df = ingestion.fetch_data(cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        remove(cls.data_dir)

    def test_expected_columns(self):
        for col in ["country", "customer_id", "invoice", "price", "stream_id", "times_viewed", "date"]:
            self.assertIn(col, self.df.columns)

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            ingestion.fetch_data("/definitely/not/here")

    def test_time_series_has_no_gaps(self):
        ts = ingestion.convert_to_ts(self.df)
        self.assertEqual(len(ts), (ts.index.max() - ts.index.min()).days + 1)

    def test_unknown_country_raises(self):
        with self.assertRaises(KeyError):
            ingestion.convert_to_ts(self.df, "Atlantis")

    def test_target_is_next_30_days_revenue(self):
        ts = ingestion.convert_to_ts(self.df)
        X, y, dates = ingestion.engineer_features(ts)
        d = dates[10]
        expected = ts.loc[d: d + pd.Timedelta(days=29), "revenue"].sum()
        self.assertAlmostEqual(y[10], expected)

    def test_features_use_only_past_data(self):
        ts = ingestion.convert_to_ts(self.df)
        d = ts.index[120]
        base = ingestion.features_for_date(ts, d)
        altered = ts.copy()
        altered.loc[d:, "revenue"] = 1e9      # change today + future
        again = ingestion.features_for_date(altered, d)
        self.assertTrue(base.equals(again))


if __name__ == "__main__":
    unittest.main()
