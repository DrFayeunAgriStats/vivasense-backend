import unittest

from interpretation import InterpretationEngine


class TestGamClassificationThresholds(unittest.TestCase):
    def test_gam_threshold_boundaries(self):
        self.assertEqual(InterpretationEngine.classify_gam(4.9), "low")
        self.assertEqual(InterpretationEngine.classify_gam(5.0), "moderate")
        self.assertEqual(InterpretationEngine.classify_gam(9.82), "moderate")
        self.assertEqual(InterpretationEngine.classify_gam(10.0), "moderate")
        self.assertEqual(InterpretationEngine.classify_gam(10.01), "high")


if __name__ == "__main__":
    unittest.main()