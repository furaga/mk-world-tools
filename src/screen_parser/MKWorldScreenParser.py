from .ScreenParser import ScreenParser, MatchInfo, ResultInfo, Player
import numpy as np
from typing import Tuple, Union
from pathlib import Path
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
        self.digit_templates = {}
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

        # 数字テンプレート画像を読み込む
        digits_dir = template_images_dir / "myrate_digits"
        if digits_dir.exists():
            for i in range(10):
                digit_path = digits_dir / f"{i}.png"
                if digit_path.exists():
                    template = imread_safe(str(digit_path))
                    self.digit_templates[str(i)] = template

        logger.info(
            f"Loaded {len(self.course_dict)} courses, {len(self.race_type_dict)} race types, and {len(self.digit_templates)} digit templates"
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
        """
        リザルト画面から自分のレートを検出する
        マリオカートワールド専用実装（黄色ハイライト部分のレートを検出）
        """
        # 画像から黄色の領域を検出して自分のレートを見つける
        my_rate, my_place = self._detect_my_rate_from_yellow_highlight(img)

        if my_rate is None or not (self.min_my_rate <= my_rate <= self.max_my_rate):
            return False, None

        # 他のプレイヤーのレートは0で初期化（要求に従って）
        # 実際に検出したい場合は後で実装可能
        rates = [0] * 12  # 最大12人のプレイヤー
        if my_place and 1 <= my_place <= 12:
            rates[my_place - 1] = my_rate

        result_info = ResultInfo(
            players=[Player(name="", rate=rate) for rate in rates],
            my_rate=my_rate,
            my_place=my_place or 1,  # プレース検出に失敗した場合は1位とする
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

    def _detect_digits_in_image(self, img: np.ndarray, threshold=0.5):
        """
        画像内の数字をテンプレートマッチングで検出する
        Returns: (detected_rate, confidence)
        """
        detections = []

        for digit, template in self.digit_templates.items():
            # テンプレートマッチング
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):  # (x, y)の順
                confidence = result[pt[1], pt[0]]
                detections.append({
                    'digit': digit,
                    'x': pt[0],
                    'y': pt[1],
                    'confidence': confidence
                })

        if not detections:
            return None, 0.0

        # X座標でソート（左から右へ）
        detections.sort(key=lambda x: x['x'])

        # 重複した検出を除去（同じ位置に複数の数字が検出された場合）
        filtered_detections = []
        for detection in detections:
            # 既存の検出と位置が近い場合は、より高い信頼度のものを採用
            is_duplicate = False
            for i, existing in enumerate(filtered_detections):
                if abs(detection['x'] - existing['x']) < 10:  # 10ピクセル以内なら重複とみなす
                    if detection['confidence'] > existing['confidence']:
                        filtered_detections[i] = detection
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_detections.append(detection)

        # 再度X座標でソート
        filtered_detections.sort(key=lambda x: x['x'])

        # 数字文字列を構築
        digit_string = ''.join([d['digit'] for d in filtered_detections])
        avg_confidence = sum([d['confidence'] for d in filtered_detections]) / len(filtered_detections)

        if digit_string.isdigit() and 4 <= len(digit_string) <= 5:
            return int(digit_string), avg_confidence

        return None, 0.0

    def _detect_my_rate_from_yellow_highlight(self, img: np.ndarray):
        """
        黄色のハイライト部分から自分のレートを検出する
        Returns: (my_rate, my_place)
        """
        # HSVカラー空間に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 複数の色範囲を定義（異なる照明条件やUI状態に対応）
        # オレンジ色の範囲（より暗い/濃いハイライト）
        lower_orange = np.array([8, 80, 80])
        upper_orange = np.array([25, 255, 255])

        # 黄色の範囲（明るいハイライト）
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        # より明るい黄色の範囲
        lower_bright_yellow = np.array([20, 50, 150])
        upper_bright_yellow = np.array([40, 255, 255])

        # 白色の範囲（明度が高く彩度が低い）
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        # 各色範囲でマスクを作成
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        bright_yellow_mask = cv2.inRange(hsv, lower_bright_yellow, upper_bright_yellow)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 全てのマスクを結合
        combined_mask = cv2.bitwise_or(orange_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, bright_yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, white_mask)

        # 最終的なマスクを使用
        yellow_mask = combined_mask

        # モルフォロジー処理でノイズを除去
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        # 輪郭を検出
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # 最大の黄色領域を見つける
        largest_contour = max(contours, key=cv2.contourArea)

        # 面積が小さすぎる場合は無視（複数の色範囲を使用するため少し緩和）
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 500:
            return None, None

        # バウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 黄色領域の右側部分（レートが表示されている部分）を抽出
        # レートは行の右端に表示されているので、右側70%の部分を取る
        rate_x = x + int(w * 0.3)
        rate_region = img[y:y+h, rate_x:x+w]

        if self.debug:
            # デバッグ用に画像を保存
            imwrite_safe("debug_yellow_mask.png", yellow_mask)
            imwrite_safe("debug_rate_region.png", rate_region)

            # 元画像に検出領域を描画
            debug_img = img.copy()
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(debug_img, (rate_x, y), (x+w, y+h), (255, 0, 0), 2)

            # 輪郭面積をテキストで表示
            cv2.putText(debug_img, f"Area: {contour_area}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            imwrite_safe("debug_detection.png", debug_img)

        # レート領域の前処理（背景除去と数字強調）
        # 一時的に前処理を無効化してテスト
        # processed_region = self._preprocess_rate_region(rate_region)

        if self.debug:
            # imwrite_safe("debug_processed_region.png", processed_region)
            pass

        # レート領域から数字を検出
        detected_rate, confidence = self._detect_digits_in_image(rate_region)

        if detected_rate is not None and confidence > 0.3:
            # プレースを推定（Y座標から行数を計算）
            # 各行の高さは大体52ピクセル程度と想定
            row_height = 52
            estimated_place = max(1, int((y + h/2) / row_height))
            estimated_place = min(12, estimated_place)  # 最大12位まで

            return detected_rate, estimated_place

        return None, None

    def _preprocess_rate_region(self, rate_region: np.ndarray):
        """
        レート領域の前処理（背景除去と数字強調）
        """
        if rate_region.size == 0:
            return rate_region

        # グレースケール化
        if len(rate_region.shape) == 3:
            gray = cv2.cvtColor(rate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = rate_region.copy()

        # ヒストグラム平均化でコントラストを向上
        enhanced = cv2.equalizeHist(gray)

        # ガウシアンぼかしを適用してノイズを除去
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # 適応的二値化で数字と背景を分離
        # 白い数字を抽出するため、閾値処理を適用
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # モルフォロジー処理でノイズを除去
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        return cleaned

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
