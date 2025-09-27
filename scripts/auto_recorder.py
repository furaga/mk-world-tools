import argparse
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from enum import Enum, auto

import cv2
import numpy as np
import time
import datetime
import yaml
from screen_parser.ScreenParser import ScreenParser, MatchInfo, ResultInfo
from OBS.OBSController import OBSController
from utils.logger import setup_logger

logger = setup_logger(__name__)

with open("data/config.yml", "r", encoding="utf8") as f:
    config = yaml.safe_load(f)


supported_games = config["games"].keys()
logger.info(f"Supported games: {supported_games}")


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--game", type=str, choices=supported_games, required=True)
    parser.add_argument(
        "--video_path", type=Path, default=None, help="指定がなければOBSから画像取得"
    )
    parser.add_argument("--out_csv_path", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--imshow", action="store_true")
    args = parser.parse_args()
    return args


class GameStatus(Enum):
    NO_CHANGE = auto()
    LOBBY = auto()
    RACE = auto()
    RESULT = auto()


class GameInfo(BaseModel):
    status: GameStatus = GameStatus.NO_CHANGE
    result_info: Optional[ResultInfo] = None
    match_info: Optional[MatchInfo] = None
    consecutive_fail_count: int = 0  # 連続でret=Falseになった回数

    @property
    def course(self) -> str:
        if self.match_info:
            return self.match_info.course
        else:
            return ""

    @property
    def rule(self) -> str:
        if self.match_info:
            return self.match_info.rule
        else:
            return ""

    @property
    def my_place(self) -> int:
        if self.result_info:
            return self.result_info.my_place
        else:
            return 0

    @property
    def my_rate(self) -> int:
        if self.result_info:
            return self.result_info.my_rate
        else:
            return 0

    @property
    def rates_in_match_info(self) -> List[int]:
        if self.match_info:
            return [p.rate for p in self.match_info.players]
        else:
            return [0 for _ in range(12)]

    @property
    def rates_in_result(self) -> List[int]:
        if self.result_info:
            return [p.rate for p in self.result_info.players]
        else:
            return [0 for _ in range(12)]


def OBS_apply_rate(obs: OBSController, game_info: GameInfo):
    obs.set_text(obs.config["my_rate"], f"{game_info.my_rate}")
    obs.set_text(obs.config["my_place"], f"前回{game_info.my_place}位")

    if game_info.my_place <= 3:
        obs.set_color(obs.config["my_place"], (100, 255, 100))
    elif game_info.my_place >= 9:
        obs.set_color(obs.config["my_place"], (255, 100, 100))
    else:
        obs.set_color(obs.config["my_place"], (255, 255, 255))


def count_valid_rates(rates: List[int]):
    return len([rate for rate in rates if rate > 0])


def update_match_info(
    match_info: MatchInfo, game_info: GameInfo, obs: Optional[OBSController]
):
    rates_in_match_info = [p.rate for p in match_info.players]
    n_valid = count_valid_rates(rates_in_match_info)
    prev_n_valid = count_valid_rates(game_info.rates_in_match_info)

    # 前のフレームより多くのプレイヤーを認識できた場合、
    # より正確に画像認識できている可能性が高いので、そのフレームの認識結果を採用する
    # （ただし、最低でも3人は認識できていないと誤検知の可能性が高いので採用しない）
    if n_valid >= max(prev_n_valid, 3):
        game_info.match_info = match_info
        if obs:
            # obs.set_text(obs.config["course"], f"{game_info.course}, {game_info.rule}")
            # 部屋の平均・最小・最大レート
            valid_rates = [v for v in game_info.rates_in_match_info if v > 0]
            avg_rate = np.mean(valid_rates)
            min_rate = np.min(valid_rates)
            max_rate = np.max(valid_rates)
            obs.set_text(
                obs.config["course"],
                #                f"平均レート  {avg_rate:.1f} ({min_rate}-{max_rate})",
                f"Room Avg. {avg_rate:.1f}",  # ({min_rate}-{max_rate})",
            )
        return True
    return False


def update_results(
    result_info: ResultInfo, game_info: GameInfo, obs: Optional[OBSController]
):
    rates_in_result = [p.rate for p in result_info.players]
    n_valid = count_valid_rates(rates_in_result)
    prev_n_valid = count_valid_rates(game_info.rates_in_result)

    # 前のフレームより多くのプレイヤーを認識できた場合、
    # より正確に画像認識できている可能性が高いので、そのフレームの認識結果を採用する
    # （ただし、最低でも3人は認識できていないと誤検知の可能性が高いので採用しない）
    if n_valid >= max(prev_n_valid, 3):
        game_info.result_info = result_info
        if obs and prev_n_valid == n_valid:
            OBS_apply_rate(obs, game_info)
        return True
    return False


def update_game_info(
    img: np.ndarray,
    game_info: GameInfo,
    parser: ScreenParser,
    obs: Optional[OBSController],
) -> tuple[bool, GameInfo]:
    # ロビー中(=レース外)：マッチング情報画面を待つ
    if game_info.status == GameStatus.LOBBY:
        ret, match_info = parser.detect_match_info(img)
        if ret and update_match_info(match_info, game_info, obs):
            game_info.status = GameStatus.RACE
            game_info.consecutive_fail_count = 0  # 状態変更時にカウンターをリセット
            return True, game_info
        else:
            return False, game_info

    # レース中：リザルトが出るのを待つ
    if game_info.status == GameStatus.RACE:
        ret, result_info = parser.detect_result(img, game_info.match_info)
        if ret and update_results(result_info, game_info, obs):
            game_info.status = GameStatus.RESULT
            game_info.consecutive_fail_count = 0  # 状態変更時にカウンターをリセット
            return True, game_info
        else:
            return False, game_info

    # リザルト表示中：リザルトが消えるまで待つ
    if game_info.status == GameStatus.RESULT:
        ret, result_info = parser.detect_result(img, game_info.match_info)
        if ret and update_results(result_info, game_info, obs):
            game_info.consecutive_fail_count = 0  # 成功したらカウンターをリセット
            return False, game_info
        if not ret:
            # 8フレーム以上連続してret=FalseならLOBBYに切り替え
            game_info.consecutive_fail_count += 1
            if game_info.consecutive_fail_count >= 6:
                game_info.status = GameStatus.LOBBY
                game_info.consecutive_fail_count = 0  # カウンターをリセット
                return True, game_info

    return False, game_info


def save_game_info(out_csv_path, game_info):
    header = ["ts", "course", "race_type", "place", "my_rate"]
    header += [f"rates_{i}" for i in range(12)]
    header += [f"rates_after_{i}" for i in range(12)]

    if not out_csv_path.exists():
        # フォルダがなければ作成
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", encoding="utf8") as f:
            f.write(",".join(header) + "\n")

    ts = datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S")
    with open(out_csv_path, "a", encoding="utf8") as f:
        text = str(ts) + ","
        text += game_info.course + ","
        text += game_info.rule + ","
        text += str(game_info.my_place) + ","
        text += str(game_info.my_rate) + ","
        text += ",".join([str(r) for r in game_info.rates_in_match_info]) + ","
        text += ",".join([str(r) for r in game_info.rates_in_result]) + "\n"
        f.write(text)
        f.flush()

    valid_rates = [v for v in game_info.rates_in_match_info if v > 0]
    mid_rate = np.median(valid_rates)
    min_rate = np.min(valid_rates)
    max_rate = np.max(valid_rates)
    logger.info(
        f"[{game_info.course} ({game_info.rule})] Place={game_info.my_place}, BR={game_info.my_rate}, {len(valid_rates)} players, {mid_rate} ({min_rate}-{max_rate})",
    )


def show_chart_browser(
    obs: Optional[OBSController] = None,
):
    if obs:
        obs.set_visible(obs.config["chart_browser"], True)
    return True, time.time()


def update_chart_browser(
    chart_visible: bool,
    chart_appear_time: float,
    obs: Optional[OBSController] = None,
):
    lifetime = config["chart_browser_lifetime_sec"]
    if chart_visible and time.time() - chart_appear_time > lifetime:
        if obs:
            obs.set_visible(obs.config["chart_browser"], False)
        chart_visible = False
    return chart_visible


def capture(
    obs: Optional[OBSController], cap: Optional[cv2.VideoCapture]
) -> Optional[np.ndarray]:
    if obs:
        return obs.capture_game_screen()
    else:
        for _ in range(100):
            ret, frame = cap.read()
        if not ret:
            return None
        return frame


def create_screen_parser(game: str, debug: bool = False, **kwargs):
    game_title, game_mode = game.split("-")
    if game_title == "mk8dx":
        from screen_parser.MK8DXScreenParser import MK8DXScreenParser

        parser = MK8DXScreenParser(
            **kwargs,
            debug=debug,
        )
        return parser
    elif game_title == "mkworld":
        from screen_parser.MKWorldScreenParser import MKWorldScreenParser

        parser = MKWorldScreenParser(
            **kwargs,
            debug=debug,
        )
        return parser
    else:
        raise ValueError(f"Invalid game: {game}")


def main(args):
    logger.info("================================================")
    logger.info(f"Auto Recorder {args.game}")
    logger.info("================================================")

    parser_config = config["games"][args.game]["parser"]
    obs_config = config["obs"] | config["games"][args.game]["obs"]
    logger.info(f"OBS config: {obs_config}")

    # 画像認識のためのrecorderを作成
    parser = create_screen_parser(args.game, args.debug, **parser_config)
    logger.info(f"Created a screen parser for {args.game}")

    # 画像入力のためのOBSまたはVideoCapture
    obs, cap = None, None
    if args.video_path is None:
        logger.info("Use OBS as input")
        obs = OBSController(**obs_config)
    else:
        logger.info(f"Use video file {args.video_path} as input")
        cap = cv2.VideoCapture(str(args.video_path))

    # ゲーム情報を管理するオブジェクト
    game_info = GameInfo(status=GameStatus.LOBBY)

    # レートの推移表の表示・非表示状態を管理する変数
    chart_visible = True
    chart_appear_time = -10000

    # FPS調整用のタイマー
    since = time.time()

    while True:
        # レートの推移表の表示・非表示状態を更新
        chart_visible = update_chart_browser(chart_visible, chart_appear_time, obs)

        # フレームを取得
        frame = capture(obs, cap)
        if frame is None:
            logger.info("End of video")
            break

        # フレームをパースしてゲーム情報を更新
        status_changed, game_info = update_game_info(frame, game_info, parser, obs)
        if args.debug:
            logger.info(f"{status_changed=}, {game_info=}")

        # 状態変化した場合の処理
        if status_changed:
            logger.info(f"Status changed: {game_info.status.name}")
            if game_info.status == GameStatus.LOBBY:
                # レース終了したので情報保存・推移表表示・リセットする
                save_game_info(args.out_csv_path, game_info)
                chart_visible, chart_appear_time = show_chart_browser(obs)
                game_info = GameInfo(status=GameStatus.LOBBY, consecutive_fail_count=0)

        # デバッグ用に画面を表示
        if args.imshow:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        # 指定されたFPSに合わせてsleep
        if config["fps"] > 0:
            time_to_sleep = 1.0 / config["fps"] - (time.time() - since)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            since = time.time()


if __name__ == "__main__":
    main(parse_args())
