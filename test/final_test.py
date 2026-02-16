# parse_and_plot_log.py
# Usage:
#   python parse_and_plot_log.py --log log/log.txt --sz 124M --out plots/log_plot.png
#   python parse_and_plot_log.py --log log/log.txt --show

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


LOSS_BASELINE = {
    "124M": 3.2924,
}

HELLA2_BASELINE = {  # HellaSwag for GPT-2
    "124M": 0.294463,
    "350M": 0.375224,
    "774M": 0.431986,
    "1558M": 0.488946,
}

HELLA3_BASELINE = {  # HellaSwag for GPT-3
    "124M": 0.337,
    "350M": 0.436,
    "774M": 0.510,
    "1558M": 0.547,
}


def parse_log_file(log_path: str):
    """
    Expected log format per line:
        <step> <stream> <val>
    e.g.
        10 train 3.12
        10 val   3.45
        10 hella 0.28
    """
    streams = {}
    bad_lines = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                bad_lines += 1
                continue

            step_s, stream, val_s = parts
            try:
                step = int(step_s)
                val = float(val_s)
            except ValueError:
                bad_lines += 1
                continue

            streams.setdefault(stream, {})[step] = val

    # convert to sorted xy lists
    streams_xy = {}
    for stream, d in streams.items():
        xy = sorted(d.items())  # list[(step, val)]
        if not xy:
            continue
        xs, ys = zip(*xy)
        streams_xy[stream] = (list(xs), list(ys))

    return streams_xy, bad_lines


def plot_streams(streams_xy, sz: str, out_path: str | None, show: bool):
    loss_baseline = LOSS_BASELINE.get(sz, None)
    hella2_baseline = HELLA2_BASELINE.get(sz, None)
    hella3_baseline = HELLA3_BASELINE.get(sz, None)

    plt.figure(figsize=(16, 6))

    # Panel 1: Losses
    plt.subplot(1, 2, 1)

    if "train" in streams_xy:
        xs, ys = streams_xy["train"]
        ys_np = np.array(ys, dtype=np.float64)
        plt.plot(xs, ys_np, label=f"nanogpt ({sz}) train loss")
        print("Min Train Loss:", float(np.min(ys_np)))
    else:
        print("Warning: 'train' stream not found in log.")

    if "val" in streams_xy:
        xs, ys = streams_xy["val"]
        ys_np = np.array(ys, dtype=np.float64)
        plt.plot(xs, ys_np, label=f"nanogpt ({sz}) val loss")
        print("Min Validation Loss:", float(np.min(ys_np)))
    else:
        print("Warning: 'val' stream not found in log.")

    if loss_baseline is not None:
        plt.axhline(
            y=loss_baseline,
            color="r",
            linestyle="--",
            label=f"OpenAI GPT-2 ({sz}) checkpoint val loss",
        )

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.ylim(top=4.0)
    plt.title("Loss")
    plt.legend()

    # Panel 2: HellaSwag eval
    plt.subplot(1, 2, 2)

    if "hella" in streams_xy:
        xs, ys = streams_xy["hella"]
        ys_np = np.array(ys, dtype=np.float64)
        plt.plot(xs, ys_np, label=f"nanogpt ({sz})")
        print("Max Hellaswag eval:", float(np.max(ys_np)))
    else:
        print("Warning: 'hella' stream not found in log.")

    if hella2_baseline is not None:
        plt.axhline(
            y=hella2_baseline,
            color="r",
            linestyle="--",
            label=f"OpenAI GPT-2 ({sz}) checkpoint",
        )
    if hella3_baseline is not None:
        plt.axhline(
            y=hella3_baseline,
            color="g",
            linestyle="--",
            label=f"OpenAI GPT-3 ({sz}) checkpoint",
        )

    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("HellaSwag eval")
    plt.legend()

    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse and visualize training log.")
    parser.add_argument("--log", type=str, default=None, help="Path to log file (e.g., log/log.txt)")
    parser.add_argument("--sz", type=str, default="124M", help="Model size key (124M, 350M, 774M, 1558M)")
    parser.add_argument("--out", type=str, default=None, help="Output image path (e.g., plots/log_plot.png)")
    parser.add_argument("--show", action="store_true", help="Show plot window")
    args = parser.parse_args()

    # default log path similar to your notebook logic
    if args.log is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        args.log = os.path.join(base_dir, "log", "log.txt")

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"Log file not found: {args.log}")

    streams_xy, bad_lines = parse_log_file(args.log)
    if bad_lines:
        print(f"Skipped {bad_lines} malformed line(s).")

    # quick visibility into what we parsed
    print("Streams found:", ", ".join(sorted(streams_xy.keys())))

    plot_streams(streams_xy, sz=args.sz, out_path=args.out, show=args.show)


if __name__ == "__main__":
    main()