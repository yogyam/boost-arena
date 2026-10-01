"""Draws the pictures in docs/images from the task definitions and the published results.

    python tools/make_figures.py

Needs matplotlib on top of the package. The task figure shows twelve seeded situations per
task from above; the baselines figure shows every task score of every bot in results/.

The other pictures in docs/images are not drawn here: pipeline.svg and observation.svg are
written by hand, and leaderboard.png and replay.gif are screenshots of the built site taken
with headless Chrome. The replay viewer takes `&t=<seconds>&camera=follow` to show one
paused moment, which is how the GIF's frames were captured, ten per second of replay.
"""

import json
import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from boost_arena import keeper  # noqa: E402
from boost_arena.runner import _episode_rngs  # noqa: E402
from boost_arena.sim import Game, ball_path  # noqa: E402
from boost_arena.tasks import FIELD_HALF_LENGTH, FIELD_HALF_WIDTH, GOAL_HALF_WIDTH, TASKS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "images")

PAGE, INK, INK2, MUTED, GRID = "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, BALL, PITCH = "#2a78d6", "#e07a1f", "#6d6b65", "#eef0ea"

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": PAGE, "axes.facecolor": PAGE, "savefig.facecolor": PAGE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.titlesize": 12, "axes.titleweight": "semibold",
})


def draw_pitch(ax, title):
    ax.set_facecolor(PITCH)
    ax.add_patch(Rectangle((-FIELD_HALF_WIDTH, -FIELD_HALF_LENGTH), 2 * FIELD_HALF_WIDTH, 2 * FIELD_HALF_LENGTH,
                           fill=False, edgecolor=MUTED, linewidth=1))
    ax.axhline(0, color=GRID, linewidth=1)
    ax.add_patch(Circle((0, 0), 1000, fill=False, edgecolor=GRID, linewidth=1))
    for y, colour in ((FIELD_HALF_LENGTH, ORANGE), (-FIELD_HALF_LENGTH, BLUE)):
        ax.add_patch(Rectangle((-GOAL_HALF_WIDTH, y if y > 0 else y - 880), 2 * GOAL_HALF_WIDTH, 880,
                               facecolor=colour, alpha=0.18, edgecolor=colour, linewidth=1.2))
    ax.set_xlim(-FIELD_HALF_WIDTH - 200, FIELD_HALF_WIDTH + 200)
    ax.set_ylim(-FIELD_HALF_LENGTH - 1100, FIELD_HALF_LENGTH + 1100)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", pad=6)


def car_marker(ax, pos, forward, colour):
    yaw = math.atan2(forward[1], forward[0])
    ax.add_patch(Rectangle((-60, -42), 120, 84, facecolor=colour, edgecolor="none",
                           transform=matplotlib.transforms.Affine2D().rotate(yaw).translate(pos[0], pos[1]) + ax.transData))
    ax.annotate("", xy=(pos[0] + 420 * math.cos(yaw), pos[1] + 420 * math.sin(yaw)), xytext=(pos[0], pos[1]),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1, mutation_scale=8))


def figure_tasks(seed=0, situations=12):
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.6))
    for ax, (key, task) in zip(axes.flat, TASKS.items()):
        game = Game(task.with_opponent)
        draw_pitch(ax, task.name)
        for episode in range(situations):
            rng, _ = _episode_rngs(task, seed, episode)
            task.setup(game, rng)
            ball = game.ball_info()
            cars = game.car_infos()
            if np.linalg.norm(ball.vel) > 1:
                path = ball_path(ball.pos, ball.vel, min(task.time_limit, 4.0))
                ax.plot(path[:, 0], path[:, 1], color=BALL, linewidth=0.8, alpha=0.5)
            ax.add_patch(Circle((ball.pos[0], ball.pos[1]), 92, facecolor="white", edgecolor=BALL, linewidth=1, zorder=3))
            car_marker(ax, cars[0].pos, cars[0].forward, BLUE)
            if len(cars) > 1:
                car_marker(ax, cars[1].pos, cars[1].forward, ORANGE)
        note = {
            "empty_net": "Still ball; score within 20 s", "pass": "Rolling ball from a wing; 20 s",
            "falling_ball": "Ball dropping from 1,200 to 1,800 uu; 20 s", "cross": "High ball arcing in; 15 s",
            "save": "Shot on the bot's goal; keep it out for 6 s", "penalty": "Scripted keeper on the line; 10 s",
        }[key]
        ax.text(0.0, -0.02, note, transform=ax.transAxes, fontsize=9.5, color=INK2, va="top")
    fig.text(0.01, 0.005, "Twelve seeded situations per task, seen from above. Blue car: the bot, attacking the orange goal at the top. "
             "Grey lines: where the ball goes on its own in the first 4 s.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(OUT, "tasks.png"), dpi=150)
    plt.close(fig)


def figure_baselines():
    folder = os.path.join(ROOT, "results")
    documents = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                document = json.load(f)
            if "results" in document:
                documents.append(document)
    documents.sort(key=lambda d: -d["overall_score"])
    keys = list(TASKS)
    fig, axes = plt.subplots(1, len(keys), figsize=(12, 3.2), sharey=True)
    for ax, key in zip(axes, keys):
        rates = []
        for document in documents:
            result = next(r for r in document["results"] if r["task"] == key)
            rates.append(result["success_rate"] * 100)
        y = np.arange(len(documents))[::-1]
        ax.barh(y, rates, height=0.5, color=BLUE)
        for yi, rate in zip(y, rates):
            ax.text(rate + 2, yi, f"{rate:.0f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 50, 100])
        ax.set_title(TASKS[key].name, loc="left", fontsize=11)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(documents))[::-1])
    axes[0].set_yticklabels([d["manifest"]["name"] for d in documents], fontsize=10)
    fig.text(0.01, 0.01, "Success rate, percent of 1,000 episodes, for every bot on the leaderboard.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "baselines.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_tasks()
    figure_baselines()
    print("wrote", OUT)
