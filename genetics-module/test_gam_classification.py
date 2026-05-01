import unittest

from interpretation import InterpretationEngine


class TestGamClassificationThresholds(unittest.TestCase):
    def test_gam_threshold_boundaries(self):
        self.assertEqual(InterpretationEngine.classify_gam(4.9), "Low")
        self.assertEqual(InterpretationEngine.classify_gam(5.0), "Medium")
        self.assertEqual(InterpretationEngine.classify_gam(9.82), "Medium")
        self.assertEqual(InterpretationEngine.classify_gam(10.0), "Medium")
        self.assertEqual(InterpretationEngine.classify_gam(10.01), "High")


if __name__ == "__main__":
    unittest.main()