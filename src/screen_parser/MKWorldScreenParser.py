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

        # テンプレート画像の総数を計算
        total_digit_templates = sum(len(templates) for templates in self.digit_templates.values())

        logger.info(
            f"Loaded {len(self.course_dict)} courses, {len(self.race_type_dict)} race types, and {total_digit_templates} digit templates ({len(self.digit_templates)} digits)"
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
        # 黄色ハイライトから自分のレートを検出
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

    def _detect_digits_in_image(self, img: np.ndarray, threshold=0.4):
        """
        画像内の数字をテンプレートマッチングで検出する
        Returns: (detected_rate, confidence)
        """
        detections = []

        for digit, templates in self.digit_templates.items():
            # 複数のテンプレートに対してマッチング
            for template in templates:
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
                if abs(detection['x'] - existing['x']) < 5:  # 5ピクセル以内なら重複とみなす
                    # 特別ケース: 9と0の混同を処理
                    if (detection['digit'] == '0' and existing['digit'] == '9') or \
                       (detection['digit'] == '9' and existing['digit'] == '0'):
                        # 9と0が同じ位置で競合している場合、信頼度差が小さければ9を優先
                        if detection['digit'] == '9':
                            filtered_detections[i] = detection
                        elif existing['digit'] == '9':
                            # 既存が9なら保持（何もしない）
                            pass
                        else:
                            # 通常の信頼度比較
                            if detection['confidence'] > existing['confidence']:
                                filtered_detections[i] = detection
                    else:
                        # 通常の重複処理
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

    def _detect_my_rate_from_fixed_positions(self, img: np.ndarray):
        """
        固定座標を使って各プレイヤー行から自分のレートを検出する
        Returns: (my_rate, my_place)
        """
        h, w = img.shape[:2]

        # 1920x1080の場合のプレイヤーボックス座標
        # x1=1008, y1=40+70i, x2=1865, y2=110+70i
        # 画像のサイズに応じてスケール調整
        scale_x = w / 1920.0
        scale_y = h / 1080.0

        x1 = int(1008 * scale_x)
        x2 = int(1865 * scale_x)

        # 各プレイヤー行をチェック（13行）
        for i in range(13):
            y1 = int((40 + 70 * i) * scale_y)
            y2 = int((110 + 70 * i) * scale_y)

            # プレイヤー行の領域を抽出
            if y2 >= h or x2 >= w:
                continue

            player_row = img[y1:y2, x1:x2]

            # レート部分は右端付近にあると仮定して右端30%の領域を使用
            rate_width = int(player_row.shape[1] * 0.3)
            rate_region = player_row[:, -rate_width:]

            # この領域で数字認識を実行
            detected_rate, confidence = self._detect_digits_in_image(rate_region, threshold=0.6)

            if detected_rate is not None and self.min_my_rate <= detected_rate <= self.max_my_rate:
                return detected_rate, i + 1

        return None, None

    def _detect_my_rate_from_yellow_highlight(self, img: np.ndarray):
        """
        黄色のハイライト部分から自分のレートを検出する（バックアップ方法）
        Returns: (my_rate, my_place)
        """
        # 検索領域を右側のプレイヤーリスト部分に限定
        # 画像の右半分のみを対象とする
        h, w = img.shape[:2]
        search_region = img[:, w//2:]  # 右半分のみ

        # HSVカラー空間に変換
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)

        # より厳密な黄色/オレンジ色の範囲を定義
        # 実際の1位ハイライトの色に焦点を当てる
        lower_yellow = np.array([15, 120, 150])  # より厳密な黄色
        upper_yellow = np.array([35, 255, 255])

        # 黄色のマスクを作成
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # モルフォロジー処理でノイズを除去
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        # 輪郭を検出
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # 適切な黄色領域を見つける（面積と位置を考慮）
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # 面積が小さすぎる場合は除外
                continue

            # バウンディングボックスを取得
            temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(contour)

            # プレイヤー行らしい形状（横長）であることを確認
            aspect_ratio = temp_w / temp_h if temp_h > 0 else 0
            if aspect_ratio < 3:  # 横幅が高さの3倍以上でない場合は除外
                continue

            valid_contours.append((contour, area, temp_y))

        if not valid_contours:
            return None, None

        # 面積が大きく、かつ上位にある輪郭を優先選択
        # Y座標が小さいほど（上にあるほど）高得点
        def score_contour(contour_info):
            contour, area, y = contour_info
            # 面積スコア（正規化）+ 位置スコア（Y座標が小さいほど高得点）
            area_score = area / 10000  # 面積を正規化
            position_score = (search_region.shape[0] - y) / search_region.shape[0] * 100  # 上にあるほど高得点
            return area_score + position_score

        best_contour_info = max(valid_contours, key=score_contour)
        largest_contour = best_contour_info[0]
        contour_area = best_contour_info[1]

        # バウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 座標を元の画像座標系に戻す（右半分を検索していたため）
        original_w = img.shape[1]
        x = x + original_w // 2

        # レート部分の固定座標を使用（1920x1080基準）
        # レート数字部分のみを抽出（装飾を除外）
        h_img, w_img = img.shape[:2]
        scale_x = w_img / 1920.0
        rate_x1 = int(1730 * scale_x)  # 装飾を除外しつつ数字全体を含める
        rate_x2 = int(1865 * scale_x)

        # Y座標は検出された黄色領域のものを使用
        rate_region = img[y:y+h, rate_x1:rate_x2]

        # この領域から数字を検出
        detected_rate, confidence = self._detect_digits_in_image(rate_region, threshold=0.6)

        if self.debug:
            # デバッグ用に画像を保存
            # yellow_maskは検索領域のサイズなので、元画像サイズに拡張
            full_yellow_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            full_yellow_mask[:, original_w//2:] = yellow_mask
            imwrite_safe("debug_yellow_mask.png", full_yellow_mask)
            imwrite_safe("debug_rate_region.png", rate_region)

            # 元画像に検出領域を描画
            debug_img = img.copy()
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 輪郭面積をテキストで表示
            cv2.putText(debug_img, f"Area: {contour_area}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            imwrite_safe("debug_detection.png", debug_img)

        if detected_rate is not None and confidence > 0.4:
            # プレースを推定（Y座標から行数を計算）
            # 各行: y1=40+70i, y2=110+70i (i=0,1,2,...)
            row_height = 70
            first_row_y = 40
            y_center = y + h/2
            estimated_place = int((y_center - first_row_y) / row_height) + 1
            estimated_place = max(1, min(13, estimated_place))  # 1-13位の範囲

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
