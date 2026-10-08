"""Run every unit test (API, model, logging, ingestion) with ONE command:  python run-tests.py"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    suite = unittest.TestLoader().discover("unittests", pattern="*Tests.py", top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
