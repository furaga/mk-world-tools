import subprocess
import random
import argparse
from pathlib import Path
import cv2
import logging
import concurrent.futures
import time
import urllib.parse
import re

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


def parse_url_and_time(url_str: str) -> tuple[str | None, float | None]:
    """Parses a URL to extract the base URL and time in seconds from '?t=...'."""
    try:
        parsed_url = urllib.parse.urlparse(url_str)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        # Handle 't=' or potentially 'start='
        time_str = query_params.get("t", query_params.get("start", [None]))[0]

        # Remove query and fragment for the base URL
        base_url = urllib.parse.urlunparse(parsed_url._replace(query="", fragment=""))

        if time_str:
            # Handle potential suffixes like 's' (though youtube uses integers)
            time_sec = float(
                re.sub(r"[^\d.]", "", time_str)
            )  # Strip non-numeric chars except '.'
            return base_url, time_sec
        else:
            return url_str, None  # Return original URL if no time
    except Exception as e:
        logger.warning(f"Could not parse URL or time from '{url_str}': {e}")
        return None, None


def _extract_frame_at_time(
    task_num: int,
    total_tasks: int,
    original_url: str,
    base_url: str,
    time_sec: float,
    output_dir: Path,
    temp_dir: Path,
    yt_dlp_path: Path,
):
    """Downloads a segment around a specific time and extracts the first frame."""
    timestamp_str_file = (
        format_timestamp(time_sec).replace(":", "").replace(".", "-")
    )  # For filename e.g., 0534-00
    timestamp_str_log = format_timestamp(time_sec)  # For logging MM:SS.ms
    log_prefix = f"[{task_num}/{total_tasks} @{timestamp_str_log}]"
    logger.info(f"{log_prefix} Processing URL: {original_url}")

    # Calculate a small segment starting exactly at the desired timestamp
    segment_start = time_sec
    segment_end = (
        time_sec + 1.5
    )  # Download 1.5 second segment to increase chance of getting the frame

    segment_start_fmt = format_timestamp(segment_start)
    segment_end_fmt = format_timestamp(segment_end)

    # Attempt to create a more descriptive filename part from URL
    try:
        url_path_part = urllib.parse.urlparse(original_url).path.strip("/")
        url_safe_part = re.sub(
            r'[\\/:*?"<>|\s]+', "_", url_path_part
        )  # Basic sanitization
        if not url_safe_part:  # Handle cases like root path only
            url_safe_part = re.sub(
                r'[\\/:*?"<>|\s]+', "_", urllib.parse.urlparse(original_url).netloc
            )
        max_len = 50  # Limit length
        url_safe_part = (
            url_safe_part[-max_len:] if len(url_safe_part) > max_len else url_safe_part
        )
    except Exception:
        url_safe_part = f"url_{task_num}"  # Fallback

    output_filename = f"{url_safe_part}_{timestamp_str_file}.jpg"
    output_path = output_dir / output_filename
    temp_video_output_template = (
        temp_dir / f"segment_{task_num}_{timestamp_str_file}.%(ext)s"
    )
    temp_video_path = None  # Initialize

    try:
        # Download just a small segment
        download_command = [
            str(yt_dlp_path),
            base_url,  # Use base_url for download
            # "--quiet", # Keep output for debugging potential download issues
            "--no-warnings",
            # "--force-keyframes-at-cuts", # Might help but can be slower
            "--download-sections",
            f"*{segment_start_fmt}-{segment_end_fmt}",
            "-o",
            str(temp_video_output_template),
            "-f",
            "bestvideo[ext=mp4]/bestvideo[ext=webm]/bestvideo/best",  # Prefer common formats
            # Add headers to potentially mimic browser request
            "--add-header",
            "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36",
            "--add-header",
            "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "--add-header",
            "Accept-Language:en-US,en;q=0.9",
        ]

        logger.info(
            f"{log_prefix} Downloading segment: {segment_start_fmt} - {segment_end_fmt}"
        )
        result = subprocess.run(
            download_command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Handle potential decoding errors in output
        )
        logger.debug(f"{log_prefix} yt-dlp stdout: {result.stdout}")
        logger.debug(f"{log_prefix} yt-dlp stderr: {result.stderr}")

        # Find the downloaded segment file more reliably
        potential_files = list(
            temp_dir.glob(f"segment_{task_num}_{timestamp_str_file}.*")
        )
        if not potential_files:
            logger.error(
                f"{log_prefix} Downloaded segment file not found for {timestamp_str_log}."
                f"\nCheck yt-dlp output:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            return
        temp_video_path = potential_files[0]
        if temp_video_path.stat().st_size == 0:
            logger.error(
                f"{log_prefix} Downloaded segment file {temp_video_path} is empty for {timestamp_str_log}."
            )
            temp_video_path.unlink(missing_ok=True)  # Clean up empty file
            return

        # Use OpenCV to extract the frame
        logger.info(f"{log_prefix} Extracting frame using OpenCV to {output_path}...")
        cap = cv2.VideoCapture(str(temp_video_path))
        if not cap.isOpened():
            logger.error(
                f"{log_prefix} Error: OpenCV could not open video segment: {temp_video_path}"
            )
            return

        # Read the first available frame
        ret, frame = cap.read()
        cap.release()  # Release early

        if ret and frame is not None:
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"{log_prefix} Successfully saved frame to {output_path}")
        else:
            logger.error(
                f"{log_prefix} Error: Failed to read frame from video segment {temp_video_path}"
            )

    except subprocess.CalledProcessError as e:
        logger.error(
            f"{log_prefix} Failed to download segment for {timestamp_str_log}. URL: {base_url}"
            f"\nError: {e.stderr}"
        )
    except Exception as e:  # Catch other potential errors (OpenCV, filesystem, etc.)
        logger.error(
            f"{log_prefix} An unexpected error occurred processing {original_url} at {timestamp_str_log}: {e}",
            exc_info=True,  # Log traceback for unexpected errors
        )
    finally:
        # Clean up the temporary video segment
        if temp_video_path and temp_video_path.exists():
            try:
                temp_video_path.unlink()
                logger.debug(f"{log_prefix} Deleted temp file: {temp_video_path}")
            except OSError as e:
                logger.warning(
                    f"{log_prefix} Could not delete temp file {temp_video_path}: {e}"
                )


def process_urls_from_file(
    url_file: Path,
    output_dir: Path = Path("screenshots"),
    yt_dlp_path: Path = Path("tools/third_party/yt-dlp.exe"),
    max_workers: int = 4,
) -> None:
    """
    Reads URLs with timestamps from a file, downloads segments,
    and extracts the specified frame for each URL in parallel.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    urls_to_process = []
    try:
        with open(url_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                url_str = line.strip()
                if not url_str or url_str.startswith("#"):  # Skip empty lines/comments
                    continue
                base_url, time_sec = parse_url_and_time(url_str)
                if base_url and time_sec is not None:
                    urls_to_process.append(
                        {
                            "original_url": url_str,
                            "base_url": base_url,
                            "time_sec": time_sec,
                        }
                    )
                else:
                    logger.warning(
                        f"Skipping line {i + 1}: Invalid URL or missing timestamp: {url_str}"
                    )
    except FileNotFoundError:
        logger.error(f"Error: URL file not found at {url_file}")
        return
    except Exception as e:
        logger.error(f"Error reading URL file {url_file}: {e}")
        return

    if not urls_to_process:
        logger.error("No valid URLs with timestamps found in the file.")
        return

    temp_dir = Path(".cache/screenshots_specific")  # Use a different cache dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using temporary directory: {temp_dir}")

    total_tasks = len(urls_to_process)
    logger.info(
        f"Found {total_tasks} URLs with timestamps. Starting parallel extraction with {max_workers} workers..."
    )

    futures = []
    start_time = time.monotonic()
    # Use ThreadPoolExecutor for I/O-bound tasks (downloading)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, task_data in enumerate(urls_to_process):
            future = executor.submit(
                _extract_frame_at_time,
                task_num=i + 1,
                total_tasks=total_tasks,
                original_url=task_data["original_url"],
                base_url=task_data["base_url"],
                time_sec=task_data["time_sec"],
                output_dir=output_dir,
                temp_dir=temp_dir,
                yt_dlp_path=yt_dlp_path,
            )
            futures.append(future)

        # Wait for all futures to complete and log any exceptions
        completed_tasks = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Raise exception if task failed
                completed_tasks += 1
            except Exception as e:
                # Error already logged inside _extract_frame_at_time
                pass  # logger.error(f"A task failed: {e}", exc_info=True) # Redundant?

    end_time = time.monotonic()
    logger.info(f"Parallel extraction finished in {end_time - start_time:.2f} seconds.")
    logger.info(f"Successfully processed {completed_tasks}/{total_tasks} URLs.")

    # Optional: Clean up the temp directory
    # Consider keeping it for debugging if many errors occur
    # import shutil
    # try:
    #     shutil.rmtree(temp_dir)
    #     logger.info(f"Cleaned up temporary directory: {temp_dir}")
    # except OSError as e:
    #     logger.warning(f"Could not clean up temporary directory {temp_dir}: {e}")


def main():
    """Parses command-line arguments and initiates frame extraction from a URL file."""
    parser = argparse.ArgumentParser(
        description="Extract specific frames from a list of video URLs in a file."
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        required=True,  # Make it required
        help="Path to a text file containing video URLs (one per line, optionally with '?t=...' or '&start=...' timestamp).",
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
        "--max-workers",
        type=int,
        default=4,  # Default to 4 parallel workers
        help="Maximum number of parallel workers (threads) for downloading/processing.",
    )

    args = parser.parse_args()

    if not args.yt_dlp_path.is_file():
        logger.error(f"Error: yt-dlp executable not found at {args.yt_dlp_path}")
        return
    if not args.url_file.is_file():  # Check if url file exists
        logger.error(f"Error: URL file not found at {args.url_file}")
        return

    logger.info(f"Starting frame extraction from URL file: {args.url_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max workers: {args.max_workers}")

    process_urls_from_file(  # Call the refactored function
        url_file=args.url_file,
        output_dir=args.output_dir,
        yt_dlp_path=args.yt_dlp_path,
        max_workers=args.max_workers,
    )
    logger.info("\nExtraction process finished.")


if __name__ == "__main__":
    main()
