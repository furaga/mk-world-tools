from .AutoRecorder import AutoRecorder, MatchInfo, ResultInfo, Player
import numpy as np
from typing import Tuple
from pathlib import Path
import mk8dx_digit_ocr

from utils.cv2_util import imread_safe, crop_img
import cv2

race_type_roi = [0.16, 0.85, 0.24, 0.98]
course_roi = [0.73, 0.87, 0.82, 0.96]


players_roi_base = [
    93 / 1920,
    84 / 1080,
    1827 / 1920,
    870 / 1080,
]

result_rates_rois = [
    [
        1120 / 1280,
        (50 + 52 * i) / 720,
        1224 / 1280,
        (95 + 52 * i) / 720,
    ]
    for i in range(12)
]


# テスト用にマリオカート8DXのオートレコーダーを作成
class MK8DXAutoRecorder(AutoRecorder):
    def __init__(self):
        self.course_dict = {}
        self.race_type_dict = {}

        for d in Path("data/battle_courses").glob("*"):
            for img_path in d.glob("*.png"):
                tmpl = imread_safe(str(img_path))
                tmpl = cv2.resize(tmpl, (173, 97))
                self.course_dict.setdefault(d.stem, []).append(tmpl)

        for d in Path("data/battle_race_type").glob("*"):
            for img_path in d.glob("*.png"):
                tmpl = imread_safe(str(img_path))
                tmpl = cv2.resize(tmpl, (103, 93))
                self.race_type_dict.setdefault(d.stem, []).append(tmpl)

    def detect_match_info(self, img: np.ndarray) -> Tuple[bool, MatchInfo]:
        course, race_type = self.detect_course(img)
        if course == "" or race_type == "":
            return False, None

        rates = self.detect_rates_in_match_info(img)
        n_valid = len([x for x in rates if x > 0])
        if n_valid < 3:
            return False, None

        match_info = MatchInfo(
            players=[Player("", rate) for rate in rates],
            course=course,
            rule=race_type,
        )

        return True, match_info

    def detect_result(
        self, img: np.ndarray, match_info: MatchInfo
    ) -> Tuple[bool, ResultInfo]:
        ret, my_rate, place, rates = self.detect_rates_after(img, match_info.rule)
        if not ret:
            return False, None

        n_valid = len([x for x in rates if x > 0])
        if n_valid < 3:
            return False, None

        result_info = ResultInfo(
            players=[Player("", rate) for rate in rates],
            my_rate=my_rate,
            my_place=place,
        )
        return True, result_info

    def detect_course(self, img, threshold=0.8):
        best_score = 0
        best_course = ""
        course_img = crop_img(img, course_roi)
        course_img = cv2.resize(course_img, (173, 97))
        for k, v in self.course_dict.items():
            for i, template in enumerate(v):
                result = cv2.matchTemplate(course_img, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                # print("  ", k, max_val)
                if best_score < max_val:
                    best_score = max_val
                    best_course = k
        # print(best_course, "| score =", best_score)
        if best_score < threshold:
            return "", ""

        best_score = 0
        best_race_type = ""
        race_type_img = crop_img(img, race_type_roi)
        race_type_img = cv2.resize(race_type_img, (103, 93))
        for k, v in self.race_type_dict.items():
            for i, template in enumerate(v):
                result = cv2.matchTemplate(
                    race_type_img, template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if best_score < max_val:
                    best_score = max_val
                    best_race_type = k

        if best_score < threshold:
            return "", ""

        return best_course, best_race_type

    def detect_rates_after(self, img, rule, min_my_rate=0, max_my_rate=99999):
        inv_img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i, roi in enumerate(result_rates_rois):
            crop = crop_img(inv_img, roi)
            # レースタイプがパックンVSスパイなら自分の色が反転してる
            if rule == "パックンVSスパイ":
                ret, my_rate = mk8dx_digit_ocr.digit_ocr.detect_black_digit(
                    crop
                )  # , verbose=True)
            else:
                ret, my_rate = mk8dx_digit_ocr.digit_ocr.detect_white_digit(
                    crop
                )  # , verbose=True)
            if ret and min_my_rate <= my_rate <= max_my_rate:
                rates_after = []
                for roi in result_rates_rois:
                    crop = crop_img(inv_img, roi)
                    ret, rate = mk8dx_digit_ocr.digit_ocr.detect_digit(crop)
                    if not ret:
                        rate = 0
                    if not (500 <= rate <= 99999):
                        rate = 0
                    rates_after.append(rate)

                return True, my_rate, i + 1, rates_after

        return False, 0, 0, []
