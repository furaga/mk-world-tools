import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import time
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--imshow", action="store_true")
    args = parser.parse_args()
    return args


class LapTimer:
    def __init__(self, freeze_threshold: float = 2.0, detection_threshold: int = 5):
        self.current_time: Optional[float] = None
        self.last_time: Optional[float] = None
        self.freeze_start_time: Optional[float] = None
        self.freeze_threshold = freeze_threshold
        self.lap_times = []
        self.detection_threshold = detection_threshold
        self.same_time_count = 0

    def update(self, detected: bool, timer_value: float) -> Optional[float]:
        if not detected:
            return None

        current_wall_time = time.time()

        if self.current_time is None:
            self.current_time = timer_value
            self.last_time = timer_value
            self.freeze_start_time = current_wall_time
            return None

        time_diff = abs(timer_value - self.current_time)

        if time_diff < 0.05:
            self.same_time_count += 1
            if self.freeze_start_time is None:
                self.freeze_start_time = current_wall_time

            freeze_duration = current_wall_time - self.freeze_start_time

            if freeze_duration >= self.freeze_threshold and self.same_time_count >= self.detection_threshold:
                lap_time = self.current_time
                if len(self.lap_times) == 0 or abs(lap_time - self.lap_times[-1]) > 0.5:
                    self.lap_times.append(lap_time)
                    logger.info(f"Lap detected: {self._format_time(lap_time)}")
                    self.freeze_start_time = current_wall_time
                    self.same_time_count = 0
                    return lap_time
        else:
            self.freeze_start_time = None
            self.same_time_count = 0
            self.last_time = self.current_time
            self.current_time = timer_value

        return None

    def _format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:06.3f}"

    def get_lap_times(self):
        return [self._format_time(t) for t in self.lap_times]


def OBS_update_lap_times(obs: OBSController, lap_times: list[str]):
    if not lap_times:
        return

    lap_text = "\n".join([f"Lap {i+1}: {t}" for i, t in enumerate(lap_times)])
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

    lap_timer = LapTimer(freeze_threshold=1.5, detection_threshold=5)

    since = time.time()

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

        if args.imshow:
            display_frame = frame.copy()
            if detected:
                cv2.putText(
                    display_frame,
                    f"Timer: {lap_timer._format_time(timer_value)}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
            lap_times = lap_timer.get_lap_times()
            for i, lap_time_str in enumerate(lap_times):
                cv2.putText(
                    display_frame,
                    f"Lap {i+1}: {lap_time_str}",
                    (50, 100 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
            cv2.imshow("frame", display_frame)
            if cv2.waitKey(1) == ord("q"):
                break

        if config.get("fps", 0) > 0:
            time_to_sleep = 1.0 / config["fps"] - (time.time() - since)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            since = time.time()

    logger.info("Final lap times:")
    for i, lap_time in enumerate(lap_timer.get_lap_times()):
        logger.info(f"  Lap {i+1}: {lap_time}")


if __name__ == "__main__":
    main(parse_args())