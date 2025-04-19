import subprocess
import random
import argparse
from pathlib import Path
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_video_duration(url: str, yt_dlp_path: Path) -> float:
    """Gets the duration of a video using yt-dlp.

    Args:
        url: The URL of the video.
        yt_dlp_path: The path to the yt-dlp executable.

    Returns:
        The duration of the video in seconds.

    Raises:
        RuntimeError: If yt-dlp fails to get the duration.
    """
    command = [str(yt_dlp_path), "--print", "duration", url]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        duration_str = result.stdout.strip()
        if not duration_str:
            raise ValueError("yt-dlp returned empty duration.")
        return float(duration_str)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        raise RuntimeError(f"Failed to get video duration for {url}: {e}") from e


def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS.ms format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 100)
    return f"{minutes:02d}:{secs:02d}.{millis:02d}"


def extract_random_frames(
    url: str,
    num_frames: int = 10,
    output_dir: Path = Path("screenshots"),
    yt_dlp_path: Path = Path("tools/third_party/yt-dlp.exe"),
) -> None:
    """Extracts random frames from a video URL using yt-dlp and OpenCV.

    Downloads small segments around random timestamps using yt-dlp
    and then extracts a single frame from the middle of each segment using OpenCV.

    Args:
        url: The URL of the video.
        num_frames: The number of random frames to extract.
        output_dir: The directory to save the extracted frames.
        yt_dlp_path: Path to the yt-dlp executable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        duration = get_video_duration(url, yt_dlp_path)
        logger.info(f"Video duration: {int(duration // 60)}m {int(duration % 60)}s")
    except RuntimeError as e:
        logger.error(f"Error: {e}")
        return

    if duration < 1:
        logger.error("Video duration is too short.")
        return

    # Generate random timestamps (ensure they are at least 0.5s from start/end)
    # Use max(1, int(duration)-1) to handle very short videos safely
    valid_duration_range = max(1.0, duration - 1.0)
    num_possible_frames = int(valid_duration_range)

    if num_possible_frames == 0:
        logger.error("Not enough duration range to pick frames safely.")
        return

    actual_num_frames = min(num_frames, num_possible_frames)
    if actual_num_frames < num_frames:
        logger.warning(
            f"Warning: Video duration only allows extracting {actual_num_frames} frames."
        )

    random_times = sorted(
        random.sample(
            [
                t / 10 for t in range(5, int(duration * 10) - 5)
            ],  # Sample time points with 0.1s precision, avoiding edges
            actual_num_frames,
        )
    )

    temp_dir = Path(".cache/screenshots")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for i, time_sec in enumerate(random_times):
        frame_num = i + 1
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        timestamp_str = f"{minutes:02d}m{seconds:02d}s"
        logger.info(
            f"\nProcessing frame {frame_num}/{actual_num_frames} at ~{timestamp_str}..."
        )

        # Calculate a small segment around the desired timestamp (e.g., 2 seconds for robustness)
        segment_start = max(0, time_sec)
        segment_end = min(duration, time_sec + 1)

        segment_start_fmt = format_timestamp(segment_start)
        segment_end_fmt = format_timestamp(segment_end)

        # Temporary video segment path
        # Use a more robust naming scheme in case yt-dlp adds extra info
        temp_video_output_template = temp_dir / f"segment_{frame_num}.%(ext)s"

        # Download just a small segment
        download_command = [
            str(yt_dlp_path),
            url,
            # "--quiet",
            "--no-warnings",
            # "--force-keyframes-at-cuts", # Can help with accuracy but might slow down
            "--download-sections",
            f"*{segment_start_fmt}-{segment_end_fmt}",
            "-o",
            str(temp_video_output_template),
            # Request best video quality/resolution, prefer mp4/webm container
            "-f",
            "bestvideo[ext=mp4]/bestvideo[ext=webm]/bestvideo/best",
        ]

        logger.info(f"Downloading segment: {segment_start_fmt} - {segment_end_fmt}")
        try:
            subprocess.run(download_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Failed to download segment for timestamp {timestamp_str}. Error: {e.stderr.decode()}"
            )
            continue

        logger.info(f"Downloaded segment in {time.time() - since:.2f} seconds")

        # Find the downloaded segment file (yt-dlp might choose extension)
        downloaded_files = list(temp_dir.glob(f"segment_{frame_num}.*"))
        if not downloaded_files:
            logger.error(
                f"Could not find downloaded segment file for timestamp {timestamp_str}, skipping..."
            )
            continue
        temp_video_path = downloaded_files[0]

        # Output image path
        output_path = output_dir / f"frame_{frame_num:02d}_{timestamp_str}.jpg"

        # Use OpenCV to extract the frame closest to the target time
        logger.info(f"Extracting frame using OpenCV to {output_path}...")
        cap = cv2.VideoCapture(str(temp_video_path))
        if not cap.isOpened():
            logger.error(
                f"Error: OpenCV could not open video segment: {temp_video_path}"
            )
            continue

        ret, frame = cap.read()
        # If reading the exact frame fails, try the next one (seeking isn't always precise)
        if not ret:
            logger.warning(f"Warning: Could not read frame {0}. Trying next frame.")
            ret, frame = cap.read()

        cap.release()  # Release the video capture object

        if ret and frame is not None:
            # Save the frame with high JPEG quality
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Successfully saved frame {frame_num} to {output_path}")
        else:
            logger.error(
                f"Error: Failed to read frame from video segment {temp_video_path}"
            )


def main():
    """Parses command-line arguments and initiates frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract random frames from a video URL."
    )
    parser.add_argument("--url", help="The URL of the video (e.g., YouTube link).")
    parser.add_argument(
        "-n",
        "--num-frames",
        type=int,
        default=10,
        help="Number of random frames to extract (default: 10).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("screenshots"),
        help="Directory to save the screenshots (default: screenshots).",
    )
    parser.add_argument(
        "--yt-dlp-path",
        type=Path,
        default=Path("tools/third_party/yt-dlp.exe"),
        help="Path to the yt-dlp executable.",
    )

    args = parser.parse_args()

    if not args.yt_dlp_path.is_file():
        logger.error(f"Error: yt-dlp executable not found at {args.yt_dlp_path}")
        return

    logger.info(f"Starting frame extraction for: {args.url}")
    logger.info(f"Number of frames: {args.num_frames}")
    logger.info(f"Output directory: {args.output_dir}")

    extract_random_frames(
        url=args.url,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        yt_dlp_path=args.yt_dlp_path,
    )
    logger.info("\nExtraction process finished.")


if __name__ == "__main__":
    main()
