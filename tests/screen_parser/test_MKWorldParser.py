import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

# MKWorldScreenParserをインポート
from screen_parser.MKWorldScreenParser import MKWorldScreenParser
from screen_parser.ScreenParser import MatchInfo, Player

from utils.cv2_util import imread_safe


class TestMKWorldScreenParser(unittest.TestCase):
    def setUp(self):
        """テスト用のパーサーを初期化"""
        self.parser = MKWorldScreenParser(
            template_images_dir=Path("data/mkworld"),
            min_my_rate=1000,
            max_my_rate=99999,
            debug=False,
        )

    def test_detect_result_with_mkw_result_images(self):
        """
        tests/screen_parser/data/result/内の全画像をテスト
        ファイル名形式: mkw_result_<期待レート>.png
        """
        test_data_dir = Path("tests/screen_parser/data/result")

        # mkw_result_*.png/.jpg ファイルを全て取得
        image_files = list(test_data_dir.glob("mkw_result_*.png"))
        image_files.extend(list(test_data_dir.glob("mkw_result_*.jpg")))

        if not image_files:
            self.skipTest(f"テスト画像が見つかりません: {test_data_dir}")

        for image_path in image_files:
            with self.subTest(image=image_path.name):
                # ファイル名から期待レートを抽出
                # mkw_result_9112.png -> 9112
                expected_rate = int(image_path.stem.split("_")[-2])
                expected_place = int(image_path.stem.split("_")[-1])

                # 画像を読み込み
                img = imread_safe(str(image_path))
                self.assertIsNotNone(img, f"画像の読み込みに失敗しました: {image_path}")

                # ダミーのMatchInfoを作成
                match_info = MatchInfo(
                    players=[Player(name=f"player{i}", rate=0) for i in range(12)],
                    course="test_course",
                    rule="vs",
                )

                # detect_result()を実行
                success, result_info = self.parser.detect_result(img, match_info)

                # 結果を検証
                self.assertTrue(
                    success, f"detect_result()が失敗しました: {image_path.name}"
                )
                self.assertIsNotNone(
                    result_info, f"ResultInfoがNoneです: {image_path.name}"
                )

                # レートの検証
                self.assertEqual(
                    result_info.my_rate,
                    expected_rate,
                    f"レートが期待値と異なります ({image_path.name}): 期待値={expected_rate}, 実際={result_info.my_rate}",
                )

                # 順位が妥当な範囲内であることを確認
                self.assertEqual(
                    result_info.my_place,
                    expected_place,
                    f"順位が期待値と異なります: {image_path.name}",
                )

                # 自分のレートが正しい位置に設定されているか確認
                if 1 <= result_info.my_place <= 13:
                    self.assertEqual(
                        result_info.players[result_info.my_place - 1].rate,
                        expected_rate,
                        f"自分のレートが正しい位置に設定されていません: {image_path.name}",
                    )

                # 3人以上のレートが検出されていることを確認
                self.assertGreaterEqual(
                    len([x for x in result_info.players if x.rate > 0]),
                    3,
                    "3人以上のレートが検出されていません",
                )

    def test_detect_result_with_invalid_image(self):
        """
        無効な画像でのdetect_result()のテスト
        """
        # 黒い画像を作成（レートが検出されないはず）
        black_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # ダミーのMatchInfoを作成
        match_info = MatchInfo(
            players=[Player(name=f"player{i}", rate=0) for i in range(12)],
            course="test_course",
            rule="vs",
        )

        # detect_result()を実行
        success, result_info = self.parser.detect_result(black_img, match_info)

        # レート検出に失敗することを確認
        self.assertFalse(success, "黒い画像でレート検出が成功してしまいました")
        self.assertIsNone(result_info, "失敗時のResultInfoがNoneではありません")

    def test_detect_digits_functionality(self):
        """
        数字テンプレートの読み込みテスト
        """
        # 数字テンプレートが正しく読み込まれているか確認
        self.assertEqual(
            len(self.parser.digit_templates),
            10,
            "数字テンプレートが10個読み込まれていません",
        )

        # 各数字（0-9）のテンプレートが存在するか確認
        for i in range(10):
            self.assertIn(
                str(i),
                self.parser.digit_templates,
                f"数字{i}のテンプレートが読み込まれていません",
            )
            self.assertIsNotNone(
                self.parser.digit_templates[str(i)],
                f"数字{i}のテンプレート画像がNoneです",
            )

    def test_detect_match_info_with_mkw_match_images(self):
        """
        tests/screen_parser/data/match_info/内のマッチ画像をテスト
        ファイル名形式: mkw_match_<期待レート>.png
        """
        test_data_dir = Path("tests/screen_parser/data/match_info")

        # mkw_match_*.png ファイルを全て取得
        image_files = list(test_data_dir.glob("mkw_match_*.png"))
        image_files += list(test_data_dir.glob("mkw_match_*.jpg"))

        if not image_files:
            self.skipTest(
                f"テスト画像が見つかりません: {test_data_dir}/mkw_match_*.png"
            )

        for image_path in image_files:
            with self.subTest(image=image_path.name):
                # ファイル名から期待レート（自分のレート）を抽出
                # mkw_match_9087.png -> 9087
                expected_my_rate = int(image_path.stem.split("_")[-1])

                # 画像を読み込み
                img = imread_safe(str(image_path))
                self.assertIsNotNone(img, f"画像の読み込みに失敗しました: {image_path}")

                # detect_match_info()を実行
                success, match_info = self.parser.detect_match_info(img)

                # 結果を検証
                self.assertTrue(
                    success, f"detect_match_info()が失敗しました: {image_path.name}"
                )
                self.assertIsNotNone(
                    match_info, f"MatchInfoがNoneです: {image_path.name}"
                )

                # プレイヤー数の確認（最大26人: 左列13人 + 右列13人）
                self.assertGreater(
                    len(match_info.players),
                    0,
                    f"プレイヤーが検出されていません: {image_path.name}",
                )

                # 自分のレートが検出されたか確認
                my_player = None
                for player in match_info.players:
                    if player.name == str(expected_my_rate):
                        my_player = player
                        break

                self.assertIsNotNone(
                    my_player,
                    f"自分のレート（{expected_my_rate}）がプレイヤー名として設定されていません: {image_path.name}",
                )

                # 自分のレートが正しく設定されているか確認
                self.assertEqual(
                    my_player.rate,
                    expected_my_rate,
                    f"自分のレートが期待値と異なります: 期待値={expected_my_rate}, 実際={my_player.rate}",
                )

                # 他のプレイヤーのレートも検出されているか確認（少なくとも3人以上）
                valid_rates = [p.rate for p in match_info.players if p.rate > 0]
                self.assertGreaterEqual(
                    len(valid_rates),
                    3,
                    f"有効なレートが3つ未満です: {image_path.name}",
                )

    def test_detect_match_info_with_invalid_image(self):
        """
        無効な画像でのdetect_match_info()のテスト
        """
        # 黒い画像を作成（レートが検出されないはず）
        black_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # detect_match_info()を実行
        success, match_info = self.parser.detect_match_info(black_img)

        # レート検出に失敗することを確認（有効なレートが3つ未満）
        self.assertFalse(success, "黒い画像でレート検出が成功してしまいました")
        self.assertIsNone(match_info, "失敗時のMatchInfoがNoneではありません")

    def test_detect_timer_with_ta_images(self):
        """
        tests/screen_parser/data/ta/内のタイムアタック画像をテスト
        ファイル名形式: mkw_ta_<分>m<秒>s<ミリ秒>.png
        """
        test_data_dir = Path("tests/screen_parser/data/ta")

        image_files = list(test_data_dir.glob("mkw_ta_*.png"))
        image_files.extend(list(test_data_dir.glob("mkw_ta_*.jpg")))

        if not image_files:
            self.skipTest(f"テスト画像が見つかりません: {test_data_dir}")

        for image_path in image_files:
            with self.subTest(image=image_path.name):
                time_str = image_path.stem.replace("mkw_ta_", "")
                parts = time_str.replace("m", "_").replace("s", "_").split("_")
                if len(parts) >= 3:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    milliseconds = int(parts[2])
                    expected_time = minutes * 60 + seconds + milliseconds / 1000.0
                else:
                    self.fail(f"ファイル名のフォーマットが不正です: {image_path.name}")

                img = imread_safe(str(image_path))
                self.assertIsNotNone(img, f"画像の読み込みに失敗しました: {image_path}")

                detected, timer_value = self.parser.detect_timer(img)

                self.assertTrue(
                    detected, f"detect_timer()が失敗しました: {image_path.name}"
                )

                self.assertAlmostEqual(
                    timer_value,
                    expected_time,
                    places=1,
                    msg=f"タイマー値が期待値と異なります ({image_path.name}): 期待値={expected_time:.3f}, 実際={timer_value:.3f}",
                )

    def test_detect_timer_with_invalid_image(self):
        """
        無効な画像でのdetect_timer()のテスト
        """
        black_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        detected, timer_value = self.parser.detect_timer(black_img)

        self.assertFalse(detected, "黒い画像でタイマー検出が成功してしまいました")


if __name__ == "__main__":
    unittest.main()
