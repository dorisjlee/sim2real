#!/usr/bin/env python
"""
Replays a text transcript (e.g. place_yellow_rectangle_act_v3_train_log.txt)
in the terminal for screen recording: prints the file line-by-line with a
configurable delay, and renders a tqdm-style progress bar (pinned at the
bottom, updated on every training log line) so it looks like a live
lerobot-train session instead of a wall of text dumped instantly.

Usage:
    python replay_log.py C:\\Users\\doris\\place_yellow_rectangle_act_v3_train_log.txt
    python replay_log.py transcript.txt --line-delay 0.05
    python replay_log.py transcript.txt --real-time --time-scale 200

All --line-delay / --blank-pause values are in SECONDS. --time-scale is NOT
seconds -- it's a divisor applied to the real elapsed seconds between two
timestamped lines (e.g. --time-scale 100 turns an 80s real gap into 0.8s).
"""

import argparse
import re
import sys
import time
from datetime import datetime

# Windows consoles often default to cp1252, which can't encode the block
# character used in the progress bar. Force UTF-8 if the stream supports it.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STEP_LINE_RE = re.compile(r"\bstep:\S+ smpl:\S+ .*\bloss:")
START_TRAINING_RE = re.compile(r"Start offline training")
END_TRAINING_RE = re.compile(r"End of training")


def parse_timestamp(line: str):
    m = TIMESTAMP_RE.search(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")


def format_hms(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_bar(step: int, total: int, start_time: float, bar_width: int = 40) -> str:
    frac = min(step / total, 1.0) if total else 0.0
    filled = int(bar_width * frac)
    bar = "█" * filled + " " * (bar_width - filled)
    elapsed = time.monotonic() - start_time
    rate = step / elapsed if elapsed > 0 else 0.0
    remaining = (total - step) / rate if rate > 0 else 0.0
    pct = int(frac * 100)
    return (
        f"Training: {pct:3d}%|{bar}| {step}/{total} "
        f"[{format_hms(elapsed)}<{format_hms(remaining)}, {rate:.2f}step/s]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the transcript text file")
    parser.add_argument(
        "--line-delay", type=float, default=0.08,
        help="Fixed seconds between each subsequent line (ignored if --real-time)",
    )
    parser.add_argument(
        "--blank-pause", type=float, default=0.6,
        help="Extra pause on blank lines (visually separates setup from training loop)",
    )
    parser.add_argument(
        "--real-time", action="store_true",
        help="Pace log lines using the real timestamps embedded in the file "
             "(compressed by --time-scale) instead of a fixed --line-delay",
    )
    parser.add_argument(
        "--time-scale", type=float, default=200.0,
        help="With --real-time: divide real elapsed seconds between timestamped "
             "lines by this factor (e.g. 200 turns an ~80s gap into ~0.4s). "
             "Not a duration itself -- it's a compression ratio.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=30000,
        help="Total training steps, for the progress bar denominator",
    )
    parser.add_argument(
        "--log-freq", type=int, default=200,
        help="Steps between training log lines, for the progress bar increment",
    )
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return

    prev_ts = None
    bar_active = False
    bar_start_time = None
    bar_step = 0

    for line in lines:
        if not line.strip():
            if bar_active:
                sys.stdout.write("\n")
            time.sleep(args.blank_pause)
            print()
            continue

        if args.real_time:
            ts = parse_timestamp(line)
            if ts is not None and prev_ts is not None:
                gap = (ts - prev_ts).total_seconds() / max(args.time_scale, 1e-6)
                time.sleep(max(gap, 0.0))
            elif ts is not None:
                time.sleep(args.line_delay)
            if ts is not None:
                prev_ts = ts
        else:
            time.sleep(args.line_delay)

        if START_TRAINING_RE.search(line):
            print(line)
            bar_active = True
            bar_start_time = time.monotonic()
            bar_step = 0
            sys.stdout.write(render_bar(bar_step, args.total_steps, bar_start_time))
            sys.stdout.flush()
            continue

        if END_TRAINING_RE.search(line):
            if bar_active:
                bar_step = args.total_steps
                sys.stdout.write("\r" + render_bar(bar_step, args.total_steps, bar_start_time) + "\n")
                bar_active = False
            print(line)
            continue

        if bar_active and STEP_LINE_RE.search(line):
            bar_step = min(bar_step + args.log_freq, args.total_steps)
            sys.stdout.write("\r" + " " * 100 + "\r")
            print(line)
            sys.stdout.write(render_bar(bar_step, args.total_steps, bar_start_time))
            sys.stdout.flush()
            continue

        if bar_active:
            # Non-progress line (e.g. checkpoint save) arriving while the bar
            # is pinned: clear the bar, print the line, redraw the bar below it.
            sys.stdout.write("\r" + " " * 100 + "\r")
            print(line)
            sys.stdout.write(render_bar(bar_step, args.total_steps, bar_start_time))
            sys.stdout.flush()
            continue

        print(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
