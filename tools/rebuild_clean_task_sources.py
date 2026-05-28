import csv
import json
import re
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "behavior-1k/2025-challenge-demos"
REPO_TYPE = "dataset"
TASKS_JSONL = ROOT / "hf_demo_metadata" / "meta" / "tasks.jsonl"
OLD_PAIRS = ROOT / "task_video_pairs_50_official" / "pairs.csv"
OUT_DIR = ROOT / "task_video_sources_50_clean"
RAW_DIR = OUT_DIR / "_raw_hevc"
VIDEO_DIR = OUT_DIR / "videos"


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def convert_to_h264(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "quiet",
        "-y",
        "-i",
        str(src),
        "-vf",
        "scale=720:720,fps=30",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "fast",
        str(tmp),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"ffmpeg failed on {src}")
    tmp.replace(dst)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [json.loads(line) for line in TASKS_JSONL.read_text().splitlines() if line.strip()]
    old_rows = list(csv.DictReader(OLD_PAIRS.open()))
    old_by_task = {row["task_id"]: row for row in old_rows}

    rows = []
    for task in tasks:
        task_index = int(task["task_index"])
        task_id = f"task-{task_index:04d}"
        old = old_by_task[task_id]
        source_video_path = old["source_video_path"]
        source_episode = Path(source_video_path).name
        hf_filename = f"videos/{task_id}/observation.images.rgb.head/{source_episode}"

        raw_path = Path(
            hf_hub_download(
                REPO_ID,
                repo_type=REPO_TYPE,
                filename=hf_filename,
                local_dir=RAW_DIR,
            )
        )

        clean_name = f"{task_id}__{slugify(task['task_name'])}.mp4"
        clean_path = VIDEO_DIR / clean_name
        if not clean_path.exists() or clean_path.stat().st_size == 0:
            print(f"convert {task_id} {source_episode}", flush=True)
            convert_to_h264(raw_path, clean_path)
        else:
            print(f"skip {task_id} {source_episode}", flush=True)

        rows.append(
            {
                "task_id": task_id,
                "task_index": task_index,
                "task_name": task["task_name"],
                "language_description": task["task"],
                "video_path": str(clean_path.relative_to(OUT_DIR)),
                "hf_source_video": hf_filename,
            }
        )

    with (OUT_DIR / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with (OUT_DIR / "pairs.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (OUT_DIR / "README.md").write_text(
        "# Clean 50 Task Source Videos\n\n"
        "Uniform source videos for manual clipping.\n\n"
        "- Codec: H.264\n"
        "- Resolution: 720x720\n"
        "- FPS: 30\n"
        "- Source: official Hugging Face HEVC head-camera videos\n"
    )

    print(f"Wrote clean sources to {OUT_DIR}")


if __name__ == "__main__":
    main()
