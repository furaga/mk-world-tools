from .ScreenParser import ScreenParser, MatchInfo, ResultInfo, Player
import numpy as np
from typing import Tuple, Union
from pathlib import Path
import mk8dx_digit_ocr
import cv2

from utils.cv2_util import imread_safe, crop_img, imwrite_safe
from utils.logger import setup_logger


logger = setup_logger(__name__)

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


class MKWorldScreenParser(ScreenParser):
    def __init__(
        self,
        template_images_dir: Union[Path, str],
        min_my_rate: int = 1000,
        max_my_rate: int = 99999,
        debug=False,
    ):
        self.course_dict = {}
        self.race_type_dict = {}
        self.debug = debug
        self.min_my_rate = min_my_rate
        self.max_my_rate = max_my_rate

        # テンプレート画像を読み込む
        template_images_dir = Path(template_images_dir)
        for d in (template_images_dir / "courses").glob("*"):
            for img_path in d.glob("*.png"):
                tmpl = imread_safe(str(img_path))
                tmpl = cv2.resize(tmpl, (173, 97))
                self.course_dict.setdefault(d.stem, []).append(tmpl)

        for d in (template_images_dir / "rules").glob("*"):
            for img_path in d.glob("*.png"):
                tmpl = imread_safe(str(img_path))
                tmpl = cv2.resize(tmpl, (103, 93))
                self.race_type_dict.setdefault(d.stem, []).append(tmpl)

        logger.info(
            f"Loaded {len(self.course_dict)} courses and {len(self.race_type_dict)} race types"
        )

    def detect_match_info(self, img: np.ndarray) -> Tuple[bool, MatchInfo]:
        course, race_type = self._course(img)
        if course == "" or race_type == "":
            return False, None

        rates = self._rates_in_match_info(img)
        n_valid = len([x for x in rates if x > 0])
        if n_valid < 3:
            return False, None

        match_info = MatchInfo(
            players=[
                Player(name=f"player{i}", rate=rate) for i, rate in enumerate(rates)
            ],
            course=course,
            rule=race_type,
        )

        return True, match_info

    def detect_result(
        self, img: np.ndarray, match_info: MatchInfo
    ) -> Tuple[bool, ResultInfo]:
        ret, my_rate, place, rates = self._rates_in_result(
            img, match_info.rule, self.min_my_rate, self.max_my_rate
        )
        if not ret:
            return False, None

        n_valid = len([x for x in rates if x > 0])
        if n_valid < 3:
            return False, None

        result_info = ResultInfo(
            players=[Player(name="", rate=rate) for rate in rates],
            my_rate=my_rate,
            my_place=place,
        )
        return True, result_info

    def _course(self, img, threshold=0.8):
        best_score = 0
        best_course = ""
        course_img = crop_img(img, course_roi)
        course_img = cv2.resize(course_img, (173, 97))
        for k, v in self.course_dict.items():
            for i, template in enumerate(v):
                result = cv2.matchTemplate(course_img, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if best_score < max_val:
                    best_score = max_val
                    best_course = k
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

    def _rates_in_match_info(self, img):
        players_roi = players_roi_base

        players_img = crop_img(img, players_roi)

        players = []
        for x in range(2):
            for y in range(6):
                players.append(
                    crop_img(players_img, [x / 2, y / 6, (x + 1) / 2, (y + 1) / 6])
                )

        rates = []
        for i, p in enumerate(players):
            rate_img = crop_img(p, [0.75, 0.5, 0.995, 0.995])
            rate_img = cv2.cvtColor(rate_img, cv2.COLOR_BGR2GRAY)
            ret, rate = mk8dx_digit_ocr.detect_digit(rate_img)
            if not ret:
                rate = 0
            if not (500 <= rate <= 99999):
                rate = 0
            rates.append(rate)

        return rates

    def _rates_in_result(self, img, rule, min_my_rate=100, max_my_rate=99999):
        inv_img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i, roi in enumerate(result_rates_rois):
            crop = crop_img(inv_img, roi)
            # レースタイプがパックンVSスパイなら自分の色が反転してる
            if rule == "パックンVSスパイ":
                if self.debug:
                    imwrite_safe(f"a_crop_{i}.png", crop)
                ret, my_rate = mk8dx_digit_ocr.digit_ocr.detect_black_digit(
                    crop
                )  # , verbose=True)
            else:
                if self.debug:
                    imwrite_safe(f"crop_{i}.png", crop)
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
