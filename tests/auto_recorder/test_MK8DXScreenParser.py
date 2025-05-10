import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

# Assuming MK8DXAutoRecorder and related classes are in src.auto_recorder
# Adjust the import path if your project structure is different
from auto_recorder.MK8DXScreenParser import (
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
            gt_rule, gt_course = image_path.stem.split("_")
            ret, match_info = recorder.detect_match_info(img)
            self.assertTrue(ret)
            self.assertEqual(match_info.rule, gt_rule)
            self.assertEqual(match_info.course, gt_course)

    def test_detect_result(self):
        recorder = MK8DXAutoRecorder(Path("data/mk8dx/battle"))

        all_image_paths = Path("tests/auto_recorder/data/result").glob("*.png")
        for image_path in all_image_paths:
            img = imread_safe(str(image_path))

            rule, gt_n_players, gt_my_place, gt_my_rate = image_path.stem.split("_")
            gt_n_players = int(gt_n_players)
            gt_my_place = int(gt_my_place)
            gt_my_rate = int(gt_my_rate)
            gt_ret = gt_n_players > 0

            match_info = MatchInfo(players=[], course="", rule=rule)
            ret, result_info = recorder.detect_result(img, match_info)
            self.assertEqual(ret, gt_ret)
            if gt_ret:
                self.assertEqual(result_info.my_place, gt_my_place)
                self.assertEqual(result_info.my_rate, gt_my_rate)


if __name__ == "__main__":
    unittest.main()
