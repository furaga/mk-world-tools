import argparse
from pathlib import Path
from typing import List, Optional, Any, Dict
from enum import Enum, auto

import cv2
import numpy as np
import time
import datetime

from auto_recorder.MK8DXAutoRecorder import MK8DXAutoRecorder
from OBS.OBSController import OBSController
from utils.logger import get_logger

logger = get_logger(__name__)


class GameStatus(Enum):
    NO_CHANGE = auto()
    LOBBY = auto()
    RACE = auto()
    RESULT = auto()


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--obs_pass", type=str, default="")
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--out_csv_path", type=Path, required=True)
    parser.add_argument("--imshow", action="store_true")
    parser.add_argument("--min_my_rate", type=int, default=1600)
    parser.add_argument("--max_my_rate", type=int, default=5000)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    return args


class GameInfo:
    def __init__(self):
        self.status: GameStatus = GameStatus.LOBBY
        self.course: str = ""
        self.race_type: str = ""
        self.rates_in_match_info: List[int] = [0 for _ in range(12)]
        self.rates_in_result: List[int] = [0 for _ in range(12)]
        self.my_place: int = 0
        self.my_rate: int = 0

    def __repr__(self) -> str:
        return (
            self.course
            + ","
            + self.race_type
            + ","
            + str(self.my_place)
            + ","
            + str(self.my_rate)
            + " | "
            + ",".join([str(r) for r in self.rates_in_match_info])
        )


def OBS_apply_rate(obs: OBSController, race_info):
    obs.set_text("バトル_現在レート", f"{race_info.my_rate}")
    obs.set_text("バトル_前回順位", f"前回{race_info.place}位")

    if race_info.place <= 3:
        obs.set_color("バトル_前回順位", (100, 255, 100))
    elif race_info.place >= 9:
        obs.set_color("バトル_前回順位", (255, 100, 100))
    else:
        obs.set_color("バトル_前回順位", (255, 255, 255))


def count_valid_rates(rates: List[int]):
    return len([rate for rate in rates if rate > 0])


def update_match_info(match_info, game_info, obs):
    rates_in_match_info = [p.rate for p in match_info.players]
    n_valid = count_valid_rates(rates_in_match_info)
    prev_n_valid = count_valid_rates(game_info.rates_in_match_info)

    # 前のフレームより多くのプレイヤーを認識できた場合、
    # より正確に画像認識できている可能性が高いので、そのフレームの認識結果を採用する
    # （ただし、最低でも3人は認識できていないと誤検知の可能性が高いので採用しない）
    if n_valid >= prev_n_valid and prev_n_valid >= 3:
        game_info.rates_in_match_info = rates_in_match_info
        game_info.course = match_info.course
        game_info.race_type = match_info.rule
        if obs:
            obs.set_text(
                "バトル_コース情報",
                f"{game_info.course}, {game_info.race_type}",
            )
        return True
    return False


def update_results(result_info, game_info, obs):
    rates_in_result = [p.rate for p in result_info.players]
    n_valid = count_valid_rates(rates_in_result)
    prev_n_valid = count_valid_rates(game_info.rates_in_result)

    # 前のフレームより多くのプレイヤーを認識できた場合、
    # より正確に画像認識できている可能性が高いので、そのフレームの認識結果を採用する
    # （ただし、最低でも3人は認識できていないと誤検知の可能性が高いので採用しない）
    if n_valid >= prev_n_valid and prev_n_valid >= 3:
        game_info.my_rate = result_info.my_rate
        game_info.my_place = result_info.my_place
        game_info.rates_in_result = rates_in_result
        if obs and prev_n_valid == n_valid:
            # 認識数が安定したらOBSに反映する
            OBS_apply_rate(obs, game_info)

        # 結果が更新されたらTrueを返す
        return True
    return False


def parse_frame(
    img: np.ndarray,
    game_info: GameInfo,
    recorder: MK8DXAutoRecorder,
    obs: Optional[OBSController],
) -> tuple[GameStatus, GameInfo]:
    # ロビー中(=レース外)：マッチング情報画面を待つ
    if game_info.status == GameStatus.LOBBY:
        ret, match_info = recorder.detect_match_info(img)
        if ret and update_match_info(match_info, game_info, obs):
            return GameStatus.RACE, game_info
        else:
            return GameStatus.NO_CHANGE, game_info

    # レース中：リザルトが出るのを待つ
    if game_info.status == GameStatus.RACE:
        ret, result_info = recorder.detect_result(
            img,
            match_info,
        )
        if ret and update_results(result_info, game_info, obs):
            return GameStatus.RESULT, game_info
        else:
            return GameStatus.NO_CHANGE, game_info

    # リザルト表示中：リザルトが消えるまで待つ
    if game_info.status == GameStatus.RESULT:
        ret, result_info = recorder.detect_result(img, match_info)
        if ret and update_results(result_info, game_info, obs):
            return GameStatus.NO_CHANGE, game_info
        else:
            return GameStatus.LOBBY, game_info

    return GameStatus.NO_CHANGE, game_info


def save_race_info(out_csv_path, race_info):
    header = ["ts", "course", "race_type", "place", "my_rate"]
    header += [f"rates_{i}" for i in range(12)]
    header += [f"rates_after_{i}" for i in range(12)]

    if not out_csv_path.exists():
        with open(out_csv_path, "w", encoding="utf8") as f:
            f.write(",".join(header) + "\n")

    ts = datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S")
    with open(out_csv_path, "a", encoding="utf8") as f:
        text = str(ts) + ","
        text += race_info.course + ","
        text += race_info.race_type + ","
        text += str(race_info.place) + ","
        text += str(race_info.my_rate) + ","
        text += ",".join([str(r) for r in race_info.rates]) + ","
        text += ",".join([str(r) for r in race_info.rates_after]) + "\n"
        f.write(text)
        f.flush()

    valid_rates = [v for v in race_info.rates if v > 0]
    mid_rate = np.median(valid_rates)
    min_rate = np.min(valid_rates)
    max_rate = np.max(valid_rates)
    print(
        f"[{race_info.course} ({race_info.race_type})] Place={race_info.place}, VR={race_info.my_rate}, {len(valid_rates)} players, {mid_rate} ({min_rate}-{max_rate})",
        flush=True,
    )


def show_chart(
    obs: Optional[OBSController] = None,
):
    if obs:
        obs.set_visible("ブラウザ_レート遷移", True)
    return True, time.time()


def update_chart_visible(
    chart_visible: bool,
    chart_appear_time: float,
    obs: Optional[OBSController] = None,
):
    if chart_visible and time.time() - chart_appear_time > 10:
        if obs:
            obs.set_visible("ブラウザ_レート遷移", False)
        chart_visible = False
    return chart_visible


def main(args):
    logger.info("BATTLE MODE")

    if len(args.obs_pass) > 0:
        logger.info("Use OBS as input")
        obs = OBSController(host="localhost", port=4444, password=args.obs_pass)
        cap = None
    else:
        logger.info(f"Use video file {args.video_path} as input")
        obs = None
        cap = cv2.VideoCapture(str(args.video_path))

    def capture() -> Optional[np.ndarray]:
        if obs:
            return obs.capture_game_screen()
        else:
            ret, frame = cap.read()
            if not ret:
                return None
            return frame

    recorder = MK8DXAutoRecorder(
        Path("data/mk8dx/battle"), args.min_my_rate, args.max_my_rate
    )
    game_info = GameInfo()
    chart_visible = True
    chart_appear_time = -10000
    since = time.time()
    while True:
        # レートの推移表の表示・非表示状態を更新
        chart_visible = update_chart_visible(chart_visible, chart_appear_time, obs)

        # フレームを取得
        frame = capture()
        if frame is None:
            logger.info("End of video")
            break

        # フレームをパース
        next_status, game_info = parse_frame(frame, game_info, recorder, obs)

        # 状態変化した場合の処理
        if next_status != GameStatus.NO_CHANGE:
            logger.info(f"Status: {game_info.status.name} -> {next_status.name}")
            game_info.status = next_status  # Update to the suggested state

            if game_info.status == GameStatus.LOBBY:
                # レース終了したので情報保存・推移表表示・リセットする
                save_race_info(args.out_csv_path, game_info)
                chart_visible, chart_appear_time = show_chart(obs)
                game_info = GameInfo()

        # デバッグ用に画面を表示
        if args.imshow:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        # 指定されたFPSに合わせてsleep
        if args.fps > 0:
            time_to_sleep = 1000 / args.fps - int((time.time() - since) * 1000)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep / 1000)
            since = time.time()


if __name__ == "__main__":
    main(parse_args())
