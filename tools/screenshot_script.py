import subprocess
import random
import os
import json
import re
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


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
    ffmpeg_path: Path = Path("ffmpeg"),  # Assume ffmpeg is in PATH by default
) -> None:
    """Extracts random frames from a video URL.

    Downloads small segments around random timestamps using yt-dlp
    and then extracts a single frame from the middle of each segment using ffmpeg.

    Args:
        url: The URL of the video.
        num_frames: The number of random frames to extract.
        output_dir: The directory to save the extracted frames.
        yt_dlp_path: Path to the yt-dlp executable.
        ffmpeg_path: Path to the ffmpeg executable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        duration = get_video_duration(url, yt_dlp_path)
        print(f"Video duration: {int(duration // 60)}m {int(duration % 60)}s")
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    if duration < 1:
        print("Video duration is too short.")
        return

    # Generate random timestamps (ensure they are at least 0.5s from start/end)
    # Use max(1, int(duration)-1) to handle very short videos safely
    valid_duration_range = max(1.0, duration - 1.0)
    num_possible_frames = int(valid_duration_range)

    if num_possible_frames == 0:
        print("Not enough duration range to pick frames safely.")
        return

    actual_num_frames = min(num_frames, num_possible_frames)
    if actual_num_frames < num_frames:
        print(
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

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for i, time_sec in enumerate(random_times):
            frame_num = i + 1
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            timestamp_str = f"{minutes:02d}m{seconds:02d}s"
            print(
                f"Processing frame {frame_num}/{actual_num_frames} at ~{timestamp_str}..."
            )

            # Calculate a small segment around the desired timestamp (e.g., 1 second)
            segment_start = max(0, time_sec - 0.5)
            segment_end = min(
                duration, time_sec + 0.5
            )  # Ensure segment_end doesn't exceed duration

            segment_start_fmt = format_timestamp(segment_start)
            segment_end_fmt = format_timestamp(segment_end)

            # Temporary video segment path
            # Use a more robust naming scheme in case yt-dlp adds extra info
            temp_video_pattern = temp_dir / f"segment_{frame_num}.*"
            temp_video_output_template = temp_dir / f"segment_{frame_num}.%(ext)s"

            # Download just a small segment
            download_command = [
                str(yt_dlp_path),
                url,
                "--quiet",
                "--no-warnings",
                # "--force-keyframes-at-cuts", # Can help with accuracy but might slow down
                "--download-sections",
                f"*{segment_start_fmt}-{segment_end_fmt}",
                "-o",
                str(temp_video_output_template),
                # Try to get a format ffmpeg easily understands
                "-f",
                "best[ext=mp4]/best[ext=webm]/best",
            ]

            print(f"Downloading segment: {segment_start_fmt} - {segment_end_fmt}")
            try:
                subprocess.run(download_command, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(
                    f"Failed to download segment for timestamp {timestamp_str}. Error: {e.stderr.decode()}"
                )
                continue

            # Find the downloaded segment file (yt-dlp might choose extension)
            downloaded_files = list(temp_dir.glob(f"segment_{frame_num}.*"))
            if not downloaded_files:
                print(
                    f"Could not find downloaded segment file for timestamp {timestamp_str}, skipping..."
                )
                continue
            temp_video_path = downloaded_files[0]

            # Output image path
            output_path = output_dir / f"frame_{frame_num:02d}_{timestamp_str}.jpg"

            # Use FFmpeg to extract a single frame near the middle of the segment
            # Calculate the precise time within the segment to extract the frame
            extract_time_in_segment = time_sec - segment_start
            extract_time_in_segment = max(
                0.01, extract_time_in_segment
            )  # Ensure positive time for -ss

            ffmpeg_command = [
                str(ffmpeg_path),
                "-y",  # Overwrite output files without asking
                "-loglevel",
                "error",  # Suppress verbose ffmpeg output
                "-i",
                str(temp_video_path),
                "-ss",
                str(
                    extract_time_in_segment
                ),  # Seek to the target time within the segment
                "-frames:v",
                "1",
                "-q:v",
                "2",  # High quality JPEG
                str(output_path),
            ]

            print(f"Extracting frame to {output_path}...")
            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                print(f"Successfully saved frame {frame_num} to {output_path}")
            except subprocess.CalledProcessError as e:
                print(
                    f"Failed to extract frame {frame_num} using ffmpeg. Error: {e.stderr.decode()}"
                )
                print(f"FFmpeg command was: {' '.join(ffmpeg_command)}")
            except FileNotFoundError:
                print(
                    f"Error: '{ffmpeg_path}' command not found. Make sure FFmpeg is installed and in your PATH."
                )
                return  # Stop processing if ffmpeg is missing


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
    parser.add_argument(
        "--ffmpeg-path",
        type=Path,
        default=Path("ffmpeg"),  # Assume ffmpeg is in PATH by default
        help="Path to the ffmpeg executable.",
    )

    args = parser.parse_args()

    if not args.yt_dlp_path.is_file():
        print(f"Error: yt-dlp executable not found at {args.yt_dlp_path}")
        return

    # Check for ffmpeg presence (basic check)
    try:
        subprocess.run(
            [str(args.ffmpeg_path), "-version"], capture_output=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            f"Warning: ffmpeg not found at '{args.ffmpeg_path}' or failed to run. Frame extraction will likely fail."
        )
        # Proceed anyway, extract_random_frames will handle the error more specifically

    print(f"Starting frame extraction for: {args.url}")
    print(f"Number of frames: {args.num_frames}")
    print(f"Output directory: {args.output_dir}")

    extract_random_frames(
        url=args.url,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        yt_dlp_path=args.yt_dlp_path,
        ffmpeg_path=args.ffmpeg_path,
    )
    print("Extraction process finished.")


if __name__ == "__main__":
    main()
