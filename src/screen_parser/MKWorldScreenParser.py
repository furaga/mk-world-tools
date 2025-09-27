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

        # 数字テンプレート画像を読み込む（複数バリエーション対応）
        digits_dir = template_images_dir / "myrate_digits"
        if digits_dir.exists():
            for i in range(10):
                digit_templates = []
                # 基本テンプレート（N.png）
                digit_path = digits_dir / f"{i}.png"
                if digit_path.exists():
                    template = imread_safe(str(digit_path))
                    if template is not None:
                        digit_templates.append(template)

                # バリエーションテンプレート（N_*.png）
                for variant_path in digits_dir.glob(f"{i}_*.png"):
                    template = imread_safe(str(variant_path))
                    if template is not None:
                        digit_templates.append(template)

                if digit_templates:
                    self.digit_templates[str(i)] = digit_templates

        # マッチ画面用の数字テンプレート画像を読み込む
        self.match_digit_templates = {}
        match_digits_dir = template_images_dir / "match_digits"
        if match_digits_dir.exists():
            for i in range(10):
                digit_templates = []
                # 基本テンプレート（N.png）
                digit_path = match_digits_dir / f"{i}.png"
                if digit_path.exists():
                    template = imread_safe(str(digit_path))
                    if template is not None:
                        digit_templates.append(template)

                # バリエーションテンプレート（N_*.png）
                for variant_path in match_digits_dir.glob(f"{i}_*.png"):
                    template = imread_safe(str(variant_path))
                    if template is not None:
                        digit_templates.append(template)

                if digit_templates:
                    self.match_digit_templates[str(i)] = digit_templates

        # 順位テンプレート画像を読み込む（白色二値化版）
        self.place_templates = {}
        places_dir = template_images_dir / "places"
        if places_dir.exists():
            for place_path in places_dir.glob("*.png"):
                place_num = int(place_path.stem)
                template = imread_safe(str(place_path))
                if template is not None:
                    # 白色領域を抽出して保存
                    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                    self.place_templates[place_num] = binary

        # タイマー用の数字テンプレート画像を読み込む
        self.timer_digit_templates = {}
        timer_digits_dir = template_images_dir / "ta"
        if timer_digits_dir.exists():
            for i in range(10):
                digit_path = timer_digits_dir / f"{i}.png"
                if digit_path.exists():
                    template = imread_safe(str(digit_path))
                    if template is not None:
                        self.timer_digit_templates[str(i)] = [template]

        # テンプレート画像の総数を計算
        total_digit_templates = sum(
            len(templates) for templates in self.digit_templates.values()
        )
        total_match_digit_templates = sum(
            len(templates) for templates in self.match_digit_templates.values()
        )

        logger.info(
            f"Loaded {len(self.course_dict)} courses, {len(self.race_type_dict)} race types, {total_digit_templates} result digit templates ({len(self.digit_templates)} digits), {total_match_digit_templates} match digit templates ({len(self.match_digit_templates)} digits), {len(self.place_templates)} place templates, and {len(self.timer_digit_templates)} timer digit templates"
        )

    def detect_match_info(self, img: np.ndarray) -> Tuple[bool, MatchInfo]:
        course, race_type = self._course(img)
        rates, my_rate = self._rates_in_match_info(img)

        n_valid = len([x for x in rates if x > 0])
        if n_valid < 1:
            return False, None

        players = []
        for i, rate in enumerate(rates):
            if rate == my_rate and my_rate is not None:
                # 自分のプレイヤー名は自分のレート
                players.append(Player(name=str(my_rate), rate=rate))
            else:
                # 他のプレイヤーは空文字
                players.append(Player(name="", rate=rate))

        match_info = MatchInfo(
            players=players,
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
        # 黄色ハイライトから自分のレートを検出
        detections = self._detect_result_rates(img)
        my_rate = None
        my_place = None

        # 最も黄色の割合が高い行を自分のレートとして選択
        my_rate_candidates = [det for det in detections if det["is_my_rate"]]
        if my_rate_candidates:
            best_candidate = max(my_rate_candidates, key=lambda x: x["yellow_ratio"])
            my_rate, my_place = best_candidate["rate"], best_candidate["place"]

        if self.debug:
            print(f"detections: {detections}")
            print(f"my_rate: {my_rate}, my_place: {my_place}")

        if my_rate is None or not (self.min_my_rate <= my_rate <= self.max_my_rate):
            return False, None

        # 他のプレイヤーのレートは0で初期化（要求に従って）
        # 実際に検出したい場合は後で実装可能
        # 最大24人のプレイヤーをサポート
        max_place = max((det["place"] for det in detections), default=13)
        rates = [0] * max(max_place, 13)

        # 他のプレイヤーのレートを設定（自分以外）
        for det in detections:
            if det["is_my_rate"]:
                continue
            if 1 <= det["place"] <= len(rates):
                rates[det["place"] - 1] = det["rate"]

        # 自分のレートを最後に設定（上書きを防ぐ）
        if 1 <= my_place <= len(rates):
            rates[my_place - 1] = my_rate

        result_info = ResultInfo(
            players=[
                Player(name=f"player{i}", rate=rate) for i, rate in enumerate(rates)
            ],
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

    def _detect_digits_in_image(
        self,
        img: np.ndarray,
        threshold=0.4,
        use_match_templates=False,
        x_overlap_threshold=9,
        y_overlap_threshold=14,
    ):
        """
        画像内の数字をテンプレートマッチングで検出する
        Returns: (detected_rate, confidence)
        """
        detections = []

        # テンプレート選択（match_templatesが空の場合はresult_templatesを使用）
        if use_match_templates and self.match_digit_templates:
            digit_templates = self.match_digit_templates
        else:
            digit_templates = self.digit_templates

        for digit, templates in digit_templates.items():
            # 複数のテンプレートに対してマッチング
            for template in templates:
                # テンプレートが画像より大きい場合はスキップ
                if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
                    continue

                # テンプレートマッチング
                result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):  # (x, y)の順
                    confidence = result[pt[1], pt[0]]
                    detections.append(
                        {
                            "digit": digit,
                            "x": pt[0],
                            "y": pt[1],
                            "confidence": confidence,
                        }
                    )

        if not detections:
            return None, 0.0

        # X座標でソート（左から右へ）
        detections.sort(key=lambda x: x["x"])

        # 重複した検出を除去（同じ位置に複数の数字が検出された場合、信頼度が高い方を採用）
        # X座標とY座標の両方を考慮
        filtered_detections = []
        for detection in detections:
            is_duplicate = False
            for i, existing in enumerate(filtered_detections):
                x_overlap = abs(detection["x"] - existing["x"]) < x_overlap_threshold
                y_overlap = abs(detection["y"] - existing["y"]) < y_overlap_threshold
                if x_overlap and y_overlap:  # 位置が近い場合は重複とみなす
                    # より高い信頼度のものを採用
                    if detection["confidence"] > existing["confidence"]:
                        filtered_detections[i] = detection
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_detections.append(detection)

        # 再度X座標でソート
        filtered_detections.sort(key=lambda x: x["x"])

        # 数字文字列を構築
        digit_string = "".join([d["digit"] for d in filtered_detections])
        avg_confidence = sum([d["confidence"] for d in filtered_detections]) / len(
            filtered_detections
        )

        if digit_string.isdigit() and 4 <= len(digit_string) <= 5:
            return int(digit_string), avg_confidence

        return None, 0.0

    def _is_yellow_background(self, region: np.ndarray) -> float:
        """
        領域の背景が黄色かどうかを判定し、黄色ピクセルの割合を返す
        """
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 黄色ピクセルの割合を返す
        yellow_ratio = np.count_nonzero(yellow_mask) / yellow_mask.size
        return yellow_ratio

    def _extract_white_digits(self, region: np.ndarray) -> np.ndarray:
        """
        白色の数字部分を抽出して二値化
        """
        # グレースケール化
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # 白色領域を抽出（閾値200以上を白とする）
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        return binary

    def _detect_place(self, region: np.ndarray, threshold=0.4) -> int:
        """
        順位領域から順位をテンプレートマッチングで検出（白色二値化版）
        Returns: 検出された順位（1-24）、検出失敗時はNone
        """
        # グレースケール化
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # 白色領域を抽出（閾値200以上を白とする）
        _, region_binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        best_score = 0
        best_place = None
        scores = []

        for place_num, template_binary in self.place_templates.items():
            if (
                template_binary.shape[0] > region_binary.shape[0]
                or template_binary.shape[1] > region_binary.shape[1]
            ):
                continue

            result = cv2.matchTemplate(
                region_binary, template_binary, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            scores.append((place_num, max_val))

            if max_val > best_score:
                best_score = max_val
                best_place = place_num

        if best_score >= threshold:
            return best_place
        return None

    def _detect_result_rates(self, img: np.ndarray):
        """
        固定座標で全プレイヤー行をチェックし、黄色背景の行からレートを検出
        Returns: list of detections with rate, place, is_my_rate, and yellow_ratio
        """
        h, w = img.shape[:2]
        scale_x = w / 1920.0
        scale_y = h / 1080.0

        # レート表示領域の固定座標（1920x1080基準）
        rate_x1 = int(1730 * scale_x)
        rate_x2 = int(1865 * scale_x)

        # 順位表示領域の固定座標（1920x1080基準）
        place_x1 = int(1070 * scale_x)
        place_x2 = int(1145 * scale_x)

        # 全13行をチェック
        detections = []
        for i in range(13):
            y1 = int((40 + 77 * i) * scale_y)
            y2 = int((110 + 77 * i) * scale_y)

            if y2 >= h or rate_x2 >= w or place_x2 >= w:
                print(f"Skip: y2: {y2}, rate_x2: {rate_x2}, place_x2: {place_x2}")
                continue

            # プレイヤー行全体を取得（黄色判定用）
            player_row = img[y1:y2, w // 2 :]  # 右半分のみ

            # 黄色背景の割合を取得
            yellow_ratio = self._is_yellow_background(player_row)

            # レート領域を抽出
            rate_region = img[y1:y2, rate_x1:rate_x2]

            # 順位領域を抽出
            place_region = img[y1:y2, place_x1:place_x2]

            if self.debug:
                imwrite_safe(f"debug_rate_region_{i}.png", rate_region)
                imwrite_safe(f"debug_place_region_{i}.png", place_region)

            # レートをテンプレートマッチングで検出
            detected_rate, confidence = self._detect_digits_in_image(
                rate_region,
                threshold=0.5,
                x_overlap_threshold=25 / 2 * scale_x,
                y_overlap_threshold=35 / 2 * scale_y,
            )

            # 順位をテンプレートマッチングで検出（二値化版）
            if yellow_ratio > 0.3:
                # 黄色の割合が30%以上の行のみ順位を検出
                detected_place = self._detect_place(place_region, threshold=0.4)
            else:
                detected_place = None

            # レートが検出された場合のみ追加（順位が検出できない場合はデフォルト値を使用）
            if detected_rate:
                # 順位が検出できなかった場合は行番号を使用（フォールバック）
                place = detected_place if detected_place else i + 1

                if self.debug:
                    debug_img = img.copy()
                    cv2.rectangle(
                        debug_img, (rate_x1, y1), (rate_x2, y2), (0, 255, 0), 2
                    )
                    cv2.rectangle(
                        debug_img, (place_x1, y1), (place_x2, y2), (255, 0, 0), 2
                    )
                    cv2.putText(
                        debug_img,
                        f"Rate: {detected_rate}, Place: {place}",
                        (rate_x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    imwrite_safe("debug_detection.png", debug_img)

                detections.append(
                    {
                        "rate": detected_rate,
                        "place": place,
                        "is_my_rate": yellow_ratio > 0.3,
                        "yellow_ratio": yellow_ratio,
                    }
                )
        return detections

    def _rates_in_match_info(self, img: np.ndarray):
        """
        マッチ画面からレート情報を検出する
        Returns: (list of rates, my_rate)
        """
        h, w = img.shape[:2]
        scale_x = w / 1280.0
        scale_y = h / 720.0

        rates = []
        my_rate = None

        # 左列と右列の座標（1280x720基準）
        # 各列はプレイヤー行全体を含み、レートは右端に表示される
        left_col_x1 = int(27 * scale_x)
        left_col_x2 = int(360 * scale_x)
        right_col_x1 = int(363 * scale_x)
        right_col_x2 = int(693 * scale_x)

        # 各列のプレイヤー行をチェック（最大13行ずつ）
        for col_x1, col_x2 in [
            (left_col_x1, left_col_x2),
            (right_col_x1, right_col_x2),
        ]:
            for i in range(13):
                y1 = int((59 + 51 * i) * scale_y)
                y2 = int((103 + 51 * i) * scale_y)
                if y2 >= h or col_x2 >= w:
                    continue

                player_row = img[y1:y2, col_x1:col_x2]

                # レート部分は右端25%の領域（レート数字は右端に表示される）
                rate_width = int(player_row.shape[1] * 0.25)
                rate_region = player_row[:, -rate_width:]

                # 白文字のレートを検出（自分のレート）
                is_white = self._is_white_text(rate_region)
                detected_rate, confidence = self._detect_digits_in_image(
                    rate_region,
                    threshold=0.8,
                    use_match_templates=True,
                    x_overlap_threshold=9 * scale_x,
                    y_overlap_threshold=14 * scale_y,
                )

                if detected_rate is not None and 1000 <= detected_rate <= 99999:
                    rates.append(detected_rate)
                    if is_white and my_rate is None:
                        my_rate = detected_rate
                else:
                    rates.append(0)

        return rates, my_rate

    def _is_white_text(self, region: np.ndarray) -> bool:
        """
        領域のテキストが白色かどうかを判定
        白色：RGB値が高い（200以上）
        灰色：RGB値が中程度（100-200）
        """
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 200)
        total_pixels = gray.size

        # 白色ピクセルの割合が3%以上なら白色テキスト
        white_ratio = white_pixels / total_pixels
        return white_ratio > 0.03

    def detect_timer(self, img: np.ndarray, threshold: float = 0.65) -> tuple[bool, float]:
        h, w = img.shape[:2]
        scale_x = w / 1920.0
        scale_y = h / 1080.0

        x1 = int(1531 * scale_x)
        y1 = int(38 * scale_y)
        x2 = int(1871 * scale_x)
        y2 = int(120 * scale_y)

        if y2 >= h or x2 >= w:
            return False, 0.0

        timer_region = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(timer_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

        detections = []
        for digit, templates in self.timer_digit_templates.items():
            for template in templates:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                _, template_binary = cv2.threshold(template_gray, 150, 255, cv2.THRESH_BINARY)

                if template_binary.shape[0] > binary.shape[0] or template_binary.shape[1] > binary.shape[1]:
                    continue

                result = cv2.matchTemplate(binary, template_binary, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    detections.append({
                        "digit": digit,
                        "x": pt[0],
                        "y": pt[1],
                        "confidence": confidence,
                    })

        if not detections:
            return False, 0.0

        detections.sort(key=lambda x: x["x"])

        filtered_detections = []
        for detection in detections:
            is_duplicate = False
            for i, existing in enumerate(filtered_detections):
                x_overlap = abs(detection["x"] - existing["x"]) < 15
                y_overlap = abs(detection["y"] - existing["y"]) < 15
                if x_overlap and y_overlap:
                    if detection["confidence"] > existing["confidence"]:
                        filtered_detections[i] = detection
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_detections.append(detection)

        filtered_detections.sort(key=lambda x: x["x"])
        digit_string = "".join([d["digit"] for d in filtered_detections])

        time_seconds = self._parse_timer_string(digit_string)
        if time_seconds is not None:
            return True, time_seconds
        return False, 0.0

    def _parse_timer_string(self, digit_string: str) -> float:
        if len(digit_string) < 6:
            return None

        try:
            if len(digit_string) == 6:
                minutes = int(digit_string[0])
                seconds = int(digit_string[1:3])
                milliseconds = int(digit_string[3:6])
            elif len(digit_string) == 7:
                minutes = int(digit_string[0:2])
                seconds = int(digit_string[2:4])
                milliseconds = int(digit_string[4:7])
            else:
                return None

            if seconds >= 60 or milliseconds >= 1000:
                return None

            total_seconds = minutes * 60 + seconds + milliseconds / 1000.0
            return total_seconds
        except ValueError:
            return None
