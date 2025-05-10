import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

# Assuming MK8DXAutoRecorder and related classes are in src.auto_recorder
# Adjust the import path if your project structure is different
from src.auto_recorder.MK8DXAutoRecorder import (
    MK8DXAutoRecorder,
    MatchInfo,
    ResultInfo,
    Player,
)

from utils.cv2_util import imread_safe


class TestMK8DXAutoRecorder(unittest.TestCase):
    def test_detect_match_info(self):
        recorder = MK8DXAutoRecorder(Path("data/mk8dx/battle"))

        all_image_paths = Path("tests/auto_recorder/data/match_info").glob("*.png")
        for image_path in all_image_paths:
            img = imread_safe(str(image_path))
            rule, course = image_path.stem.split("_")
            ret, match_info = recorder.detect_match_info(img)
            print(f"rule: {match_info.rule}, course: {match_info.course}, ret: {ret}")
            self.assertTrue(ret)
            self.assertEqual(match_info.rule, rule)
            self.assertEqual(match_info.course, course)

    def test_detect_result(self):
        pass


if __name__ == "__main__":
    unittest.main()
