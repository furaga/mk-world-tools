import subprocess
import random
import os
import json
import re
import tempfile
import shutil


def get_video_duration(url):
    yt_dlp_path = "tools/third_party/yt-dlp.exe"
    command = [yt_dlp_path, "--print", "duration", url]
    result = subprocess.run(command, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    return duration


def extract_random_frames(url, num_frames=10, output_dir="screenshots"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get video duration in seconds
    duration = get_video_duration(url)
    print(f"Video duration: {int(duration // 60)}m {int(duration % 60)}s")

    # Generate random timestamps
    random_times = sorted(
        random.sample(range(int(duration)), min(num_frames, int(duration)))
    )

    yt_dlp_path = "tools/third_party/yt-dlp.exe"

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, time_sec in enumerate(random_times):
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            # Calculate a small segment around the desired timestamp (1 second)
            segment_start = max(0, time_sec - 0.5)
            segment_end = time_sec + 0.5

            # Format for yt-dlp
            segment_start_fmt = f"{int(segment_start // 60):02d}:{int(segment_start % 60):02d}.{int((segment_start % 1) * 100):02d}"
            segment_end_fmt = f"{int(segment_end // 60):02d}:{int(segment_end % 60):02d}.{int((segment_end % 1) * 100):02d}"

            # Temporary video segment path
            temp_video_path = os.path.join(temp_dir, f"segment_{i + 1}.mp4")

            # Download just a small segment
            download_command = [
                yt_dlp_path,
                url,
                "--quiet",
                "--no-warnings",
                "--download-sections",
                f"*{segment_start_fmt}-{segment_end_fmt}",
                "-o",
                temp_video_path,
            ]

            print(f"Downloading small segment at {timestamp}...")
            subprocess.run(download_command)

            # Check if file exists
            if not os.path.exists(temp_video_path):
                # Look for other formats
                video_files = [
                    f for f in os.listdir(temp_dir) if f.startswith(f"segment_{i + 1}")
                ]
                if video_files:
                    temp_video_path = os.path.join(temp_dir, video_files[0])
                else:
                    print(
                        f"Failed to download segment for timestamp {timestamp}, skipping..."
                    )
                    continue

            # Output image path
            output_path = os.path.join(
                output_dir, f"frame_{i + 1:02d}_{minutes:02d}m{seconds:02d}s.jpg"
            )

            # Use FFmpeg to extract a single frame at the middle of the segment
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                temp_video_path,
                "-ss",
                "0.5",  # Take frame from middle of 1-second segment
                "-frames:v",
                "1",
                "-q:v",
                "2",  # High quality
                output_path,
            ]

            print(f"Extracting frame at {timestamp}...")
            subprocess.run(ffmpeg_command)
            print(
                f"Saved frame {i + 1} (at {minutes:02d}:{seconds:02d}) to {output_path}"
            )


def main():
    video_url = "https://youtu.be/V9tJ471FyM4"

    print("Extracting random frames...")
    extract_random_frames(video_url)


if __name__ == "__main__":
    main()
