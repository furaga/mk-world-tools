import unittest
import sys
from pathlib import Path
from unittest.mock import patch
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ta_lap_recorder import LapTimer


class TestLapTimer(unittest.TestCase):
    def setUp(self):
        self.lap_timer = LapTimer(freeze_threshold=0.5, detection_threshold=10)

    def test_initial_state(self):
        self.assertIsNone(self.lap_timer.current_time)
        self.assertEqual(len(self.lap_timer.lap_times), 0)
        self.assertEqual(self.lap_timer.same_time_count, 0)

    def test_first_detection(self):
        result = self.lap_timer.update(True, 5.0)
        self.assertIsNone(result)
        self.assertEqual(self.lap_timer.current_time, 5.0)

    def test_lap_detection_after_freeze(self):
        self.lap_timer.update(True, 5.0)

        for i in range(40):
            self.lap_timer.update(True, 5.0 + i * 0.4)
            time.sleep(0.02)

        time.sleep(0.6)
        result = None
        for _ in range(11):
            result = self.lap_timer.update(True, 20.6)
            time.sleep(0.05)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 20.6, places=1)
        self.assertEqual(len(self.lap_timer.lap_times), 1)

    def test_multiple_laps(self):
        self.lap_timer.update(True, 5.0)

        for i in range(210):
            self.lap_timer.update(True, 5.0 + i * 0.1)
            time.sleep(0.005)

        time.sleep(0.6)
        for _ in range(11):
            self.lap_timer.update(True, 25.95)
            time.sleep(0.05)

        self.assertEqual(len(self.lap_timer.lap_times), 1)
        self.assertAlmostEqual(self.lap_timer.lap_times[0], 26.0, places=0)

        for i in range(250):
            self.lap_timer.update(True, 26.2 + i * 0.1)
            time.sleep(0.005)

        time.sleep(0.6)
        for _ in range(11):
            self.lap_timer.update(True, 51.15)
            time.sleep(0.05)

        self.assertEqual(len(self.lap_timer.lap_times), 2)
        self.assertAlmostEqual(self.lap_timer.lap_times[1], 25.2, places=0)

    def test_reset_detection(self):
        self.lap_timer.update(True, 5.0)

        for i in range(210):
            self.lap_timer.update(True, 5.0 + i * 0.1)
            time.sleep(0.005)

        time.sleep(0.6)
        for _ in range(11):
            self.lap_timer.update(True, 25.95)
            time.sleep(0.05)

        self.assertEqual(len(self.lap_timer.lap_times), 1)

        self.lap_timer.update(True, 0.05)

        self.assertTrue(self.lap_timer.reset_detected)
        self.assertEqual(len(self.lap_timer.lap_times), 0)

    def test_no_detection_without_freeze(self):
        self.lap_timer.update(True, 5.0)

        for i in range(20):
            result = self.lap_timer.update(True, 5.0 + i * 0.5)
            self.assertIsNone(result)

        self.assertEqual(len(self.lap_timer.lap_times), 0)

    def test_ignore_short_laps(self):
        self.lap_timer.update(True, 5.0)

        for _ in range(11):
            self.lap_timer.update(True, 5.5)

        self.assertEqual(len(self.lap_timer.lap_times), 0)

    def test_detection_failure_resets_counter(self):
        self.lap_timer.update(True, 5.0)

        for _ in range(5):
            self.lap_timer.update(True, 20.5)
            time.sleep(0.05)

        self.assertGreater(self.lap_timer.same_time_count, 0)

        self.lap_timer.update(False, 0.0)

        self.assertEqual(self.lap_timer.same_time_count, 0)

    def test_format_time(self):
        self.assertEqual(self.lap_timer._format_time(0.0), "0:00.000")
        self.assertEqual(self.lap_timer._format_time(25.5), "0:25.500")
        self.assertEqual(self.lap_timer._format_time(65.123), "1:05.123")
        self.assertEqual(self.lap_timer._format_time(125.789), "2:05.789")

    def test_get_lap_times(self):
        self.lap_timer.lap_times = [25.5, 24.8, 26.1]
        lap_times_str = self.lap_timer.get_lap_times()

        self.assertEqual(len(lap_times_str), 3)
        self.assertEqual(lap_times_str[0], "0:25.500")
        self.assertEqual(lap_times_str[1], "0:24.800")
        self.assertEqual(lap_times_str[2], "0:26.100")


if __name__ == "__main__":
    unittest.main()