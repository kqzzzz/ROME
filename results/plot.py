"""Plot the archived classification accuracies for Figs. 9 and 16."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


DATA = Path(__file__).resolve().parent


def draw(ax, filename, title):
    with (DATA / filename).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        series = reader.fieldnames[1:]
        rows = list(reader)
    psr = [int(row["psr_db"]) for row in rows]
    colors = ("#65727b", "#d88314", "#41845c", "#3479a6", "#876aa5")
    markers = ("o", "s", "^", "D", "v")
    for index, label in enumerate(series):
        values = [float(row[label]) for row in rows]
        clean = label == "Clean" or label.endswith("(clean)")
        method = label.removesuffix(" (clean)")
        if method in {"ROME", "APG"}:
            color = "#b43222"
        elif method == "No defense":
            color = "#276191"
        else:
            color = colors[index % len(colors)]
        ax.plot(psr, values, label=label, color=color, linestyle="--" if clean else "-",
                marker=None if clean else markers[index % len(markers)], markersize=3, linewidth=1.5)
    ax.set(title=title, xlabel="Test PSR (dB)", ylabel="Classification accuracy",
           ylim=(0, 1), xticks=range(-15, 0, 2))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
    draw(left, "fig9a.csv", "(a) Fixed-power PGs and APG")
    draw(right, "fig9b.csv", "(b) Gaussian jamming and APG")
    fig.savefig(args.output_dir / "fig9.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.8), layout="constrained")
    draw(ax, "fig16.csv", "ROME and the standard receiver under APG")
    fig.savefig(args.output_dir / "fig16.png", dpi=180)
    plt.close(fig)
    print(f"Saved fig9.png and fig16.png to {args.output_dir}")


if __name__ == "__main__":
    main()
