import argparse
from pathlib import Path
from typing import List, Optional, Any, Dict

import cv2
import numpy as np
import time
import datetime

from auto_recorder.MK8DXAutoRecorder import MK8DXAutoRecorder
from OBS.OBSController import OBSController
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--obs_pass", type=str, default="")
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--out_csv_path", type=Path, required=True)
    parser.add_argument("--imshow", action="store_true")
    parser.add_argument("--max_my_rate", type=int, default=5000)
    parser.add_argument("--min_my_rate", type=int, default=1600)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()
    return args


class RaceInfo:
    def __init__(self):
        self.course: str = ""
        self.race_type: str = ""
        self.place: int = 0
        self.rates: List[int] = [0 for _ in range(12)]
        self.rates_after: List[int] = [0 for _ in range(12)]
        self.my_rate: int = 0
        self.delta_rate: Optional[int] = None

    def __repr__(self) -> str:
        return (
            self.course
            + ","
            + self.race_type
            + ","
            + str(self.place)
            + ","
            + str(self.my_rate)
            + " | "
            + ",".join([str(r) for r in self.rates])
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


def parse_frame(
    img,
    ts,
    status,
    race_info: RaceInfo,
    recorder: MK8DXAutoRecorder,
    obs: Optional[OBSController],
    history: List[Dict[str, Any]],
):
    history.append({"ts": ts, "status": status, "visible_coin_lap": False})
    while len(history) > 10:
        history.pop(0)

    ret, match_info = recorder.detect_match_info(img)
    if ret:
        rates = [p.rate for p in match_info.players]
        n_valid = len([x for x in rates if x > 0])
        if n_valid >= 3:
            history[-1].update({"rates": rates})
            prev_n_valid = len([x for x in race_info.rates if x > 0])
            if prev_n_valid <= n_valid:
                race_info.rates = rates
            course, race_type = match_info.course, match_info.rule
            race_info.course = course
            race_info.race_type = race_type
            if obs:
                obs.set_text("バトル_コース情報", f"{course}, {race_type}")
            return "race", race_info

    if status == "race":
        # 結果表のパース
        ret, my_rate, place, rates_after = recorder.detect_result(img, match_info)
        if ret:
            history[-1].update({"my_rate": my_rate})
            race_info.my_rate = my_rate
            if len(history) >= 2 and "my_rate" in history[-2]:
                race_info.delta_rate = my_rate - history[-2]["my_rate"]
            else:
                race_info.delta_rate = None
            race_info.place = place
            n_valid = len([x for x in rates_after if x > 0])
            if n_valid >= 3:
                prev_n_valid = len([x for x in race_info.rates_after if x > 0])
                if prev_n_valid <= n_valid:
                    race_info.rates_after = rates_after
                if obs:
                    OBS_apply_rate(obs, race_info)
                return "result", race_info

    if status == "result":
        # 結果表のパース
        ret, result_info = recorder.detect_result(img, match_info)
        if ret:
            history[-1].update({"my_rate": result_info.my_rate})
            race_info.my_rate = result_info.my_rate
            race_info.place = result_info.place
            n_valid = len([p.rate for p in result_info.players if p.rate > 0])
            prev_n_valid = len([rate for rate in race_info.rates_after if rate > 0])
            if prev_n_valid <= n_valid:
                race_info.rates_after = [p.rate for p in result_info.players]
            if obs:
                OBS_apply_rate(obs, race_info)
            return "", race_info
        else:
            return "none", race_info

    return "", race_info


def save_race_info(out_csv_path, ts, race_info):
    header = ["ts", "course", "race_type", "place", "my_rate"]
    header += [f"rates_{i}" for i in range(12)]
    header += [f"rates_after_{i}" for i in range(12)]

    if not out_csv_path.exists():
        with open(out_csv_path, "w", encoding="utf8") as f:
            f.write(",".join(header) + "\n")

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

    recorder = MK8DXAutoRecorder(Path("data/mk8dx/battle"))

    status = "none"
    race_info = RaceInfo()
    chart_visible = True
    chart_appear_time = -10000
    since = time.time()
    history: List[Dict[str, Any]] = []
    while True:
        # レートの推移表の表示・非表示状態を更新
        chart_visible = update_chart_visible(chart_visible, chart_appear_time, obs)

        # フレームを取得
        frame = capture()
        if frame is None:
            logger.info("End of video")
            break

        # 現在時刻を取得
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d@%H:%M:%S")

        # フレームをパース
        next_status, race_info = parse_frame(
            frame, now_str, status, race_info, recorder, obs, history
        )

        # 状態変化した場合の処理
        if next_status != status:
            race_finished = next_status == "none"
            if race_finished:
                # noneに変化（=レース終了）したら、レース情報を保存して、レート遷移の表示をONにする
                save_race_info(args.out_csv_path, now_str, race_info)
                chart_visible, chart_appear_time = show_chart(obs)
                race_info = RaceInfo()

            status_changed = next_status != ""
            if status_changed:
                status = next_status
                logger.info(f"Status changed: {status} -> {next_status}")

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
