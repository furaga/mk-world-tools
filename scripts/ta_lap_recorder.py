import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import time
import datetime
import yaml
from screen_parser.MKWorldScreenParser import MKWorldScreenParser
from OBS.OBSController import OBSController
from utils.logger import setup_logger

logger = setup_logger(__name__)

with open("data/config.yml", "r", encoding="utf8") as f:
    config = yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Time Attack Lap Timer")
    parser.add_argument(
        "--video_path", type=Path, default=None, help="指定がなければOBSから画像取得"
    )
    parser.add_argument(
        "--out_csv_path",
        type=Path,
        default=Path(".cache/mkworld-ta.csv"),
        help="ラップタイムを保存するCSVファイルのパス",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--imshow", action="store_true")
    args = parser.parse_args()
    return args


class LapTimer:
    def __init__(self, freeze_threshold: float = 2.0, detection_threshold: int = 10):
        self.current_time: Optional[float] = None
        self.last_time: Optional[float] = None
        self.last_detected_value: Optional[float] = None
        self.freeze_start_time: Optional[float] = None
        self.freeze_threshold = freeze_threshold
        self.lap_times = []
        self.detection_threshold = detection_threshold
        self.same_time_count = 0
        self.same_value_buffer = []
        self.reset_detected = False

    def reset(self):
        logger.info("Time Attack Reset detected")
        self.current_time = None
        self.last_time = None
        self.last_detected_value = None
        self.freeze_start_time = None
        self.lap_times = []
        self.same_time_count = 0
        self.same_value_buffer = []
        self.reset_detected = True

    def update(self, detected: bool, timer_value: float) -> Optional[float]:
        if not detected:
            self.same_time_count = 0
            self.same_value_buffer = []
            self.freeze_start_time = None
            return None

        if timer_value < 0.1:
            if len(self.lap_times) > 0:
                self.reset()
            return None

        current_wall_time = time.time()
        self.last_detected_value = timer_value

        if self.current_time is None:
            self.current_time = timer_value
            self.last_time = timer_value
            self.same_value_buffer = [timer_value]
            self.freeze_start_time = current_wall_time
            self.reset_detected = False
            return None

        time_diff = abs(timer_value - self.current_time)

        if time_diff < 0.1:
            self.same_time_count += 1
            self.same_value_buffer.append(timer_value)

            if len(self.same_value_buffer) > 20:
                self.same_value_buffer.pop(0)

            if self.freeze_start_time is None:
                self.freeze_start_time = current_wall_time

            freeze_duration = current_wall_time - self.freeze_start_time

            if (
                freeze_duration >= self.freeze_threshold
                and self.same_time_count >= self.detection_threshold
            ):
                avg_frozen_value = sum(self.same_value_buffer) / len(self.same_value_buffer)

                cumulative_lap_time = sum(self.lap_times)
                actual_lap_time = avg_frozen_value - cumulative_lap_time

                if actual_lap_time > 1.0:
                    self.lap_times.append(actual_lap_time)
                    logger.info(
                        f"Lap {len(self.lap_times)} detected: {self._format_time(actual_lap_time)} (cumulative: {self._format_time(avg_frozen_value)})"
                    )
                    self.current_time = avg_frozen_value
                    self.freeze_start_time = None
                    self.same_time_count = 0
                    self.same_value_buffer = []
                    return actual_lap_time
                else:
                    self.freeze_start_time = None
                    self.same_time_count = 0
                    self.same_value_buffer = []
        else:
            self.last_time = self.current_time
            self.current_time = timer_value
            self.freeze_start_time = None
            self.same_time_count = 0
            self.same_value_buffer = []

        return None

    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:06.3f}"

    def get_lap_times(self):
        return [self._format_time(t) for t in self.lap_times]


def save_lap_times(out_csv_path: Path, lap_times: list[float]):
    if not lap_times:
        return

    header = ["ts"] + [f"lap{i+1}" for i in range(len(lap_times))]

    if not out_csv_path.exists():
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", encoding="utf8") as f:
            f.write(",".join(header) + "\n")

    ts = datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S")
    with open(out_csv_path, "a", encoding="utf8") as f:
        text = ts + "," + ",".join([str(t) for t in lap_times]) + "\n"
        f.write(text)
        f.flush()

    logger.info(f"Lap times saved to {out_csv_path}")


def OBS_update_lap_times(obs: OBSController, lap_times: list[str]):
    if not lap_times:
        return

    lap_text = "/".join([f"{i + 1}: {t}" for i, t in enumerate(lap_times)])
    obs.set_text(obs.config.get("lap_times", "lap_times"), lap_text)


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


def main(args):
    logger.info("================================================")
    logger.info("Time Attack Lap Timer")
    logger.info("================================================")

    template_images_dir = Path("data/mkworld")
    parser = MKWorldScreenParser(
        template_images_dir=template_images_dir,
        debug=args.debug,
    )
    logger.info("Created MKWorldScreenParser")

    obs, cap = None, None
    if args.video_path is None:
        logger.info("Use OBS as input")
        obs_config = config["obs"]
        obs = OBSController(**obs_config)
    else:
        logger.info(f"Use video file {args.video_path} as input")
        cap = cv2.VideoCapture(str(args.video_path))

    lap_timer = LapTimer(freeze_threshold=2.0, detection_threshold=10)

    since = time.time()
    last_lap_count = 0

    while True:
        frame = capture(obs, cap)
        if frame is None:
            logger.info("End of video")
            break

        detected, timer_value = parser.detect_timer(frame)

        if detected and args.debug:
            logger.debug(f"Timer: {lap_timer._format_time(timer_value)}")

        lap_time = lap_timer.update(detected, timer_value)

        if lap_time is not None and obs:
            OBS_update_lap_times(obs, lap_timer.get_lap_times())

        current_lap_count = len(lap_timer.lap_times)
        if current_lap_count > last_lap_count:
            save_lap_times(args.out_csv_path, lap_timer.lap_times)
            last_lap_count = current_lap_count

        if lap_timer.reset_detected:
            lap_timer.reset_detected = False
            last_lap_count = 0
            if obs:
                OBS_update_lap_times(obs, [])

        if config.get("fps", 0) > 0:
            time_to_sleep = 1.0 / config["fps"] - (time.time() - since)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            since = time.time()


if __name__ == "__main__":
    main(parse_args())
