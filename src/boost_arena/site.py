"""Builds the leaderboard website from the result files: plain HTML, no external resources."""

import datetime
import html
import json
import os
import shutil

from .duel_service import duel_points
from .duels import DUEL_KINDS
from .rating import expected_share, ratings
from .submissions import REPLAYS_SUFFIX
from .tasks import TASKS

DISCLAIMER = (
    "Boost Arena is a fan project and is not affiliated with Psyonix or Epic Games. "
    "Portions of the materials used are trademarks and/or copyrighted works of Epic Games, Inc. "
    "All rights reserved by Epic. This material is not official and is not endorsed by Epic."
)

STYLE = """
:root { color-scheme: light dark; --ink: #1b1b1b; --muted: #6b6b6b; --line: #d9d9d9; --bg: #fafaf8; --card: #ffffff; --accent: #2a78d6; }
@media (prefers-color-scheme: dark) { :root { --ink: #ededed; --muted: #a8a8a8; --line: #3a3a3a; --bg: #161616; --card: #1f1f1f; --accent: #5aa0f0; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { margin: 0 0 4px; font-size: 26px; }
h2 { margin: 32px 0 8px; font-size: 18px; }
p.lead { margin: 0 0 20px; color: var(--muted); }
a { color: var(--accent); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 8px 12px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; cursor: pointer; user-select: none; }
th.text, td.text { text-align: left; }
td.overall { font-weight: 700; }
tr:last-child td { border-bottom: none; }
.small { color: var(--muted); font-size: 13px; }
.error { color: #b3261e; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
"""

SCRIPT = """
document.querySelectorAll("th[data-sort]").forEach(function (th) {
  th.addEventListener("click", function () {
    var table = th.closest("table"), body = table.querySelector("tbody");
    var index = Array.prototype.indexOf.call(th.parentNode.children, th);
    var numeric = th.dataset.sort === "number";
    var asc = th.dataset.dir !== "asc";
    th.parentNode.querySelectorAll("th").forEach(function (h) { delete h.dataset.dir; });
    th.dataset.dir = asc ? "asc" : "desc";
    var rows = Array.prototype.slice.call(body.querySelectorAll("tr"));
    rows.sort(function (a, b) {
      var x = a.children[index].dataset.value, y = b.children[index].dataset.value;
      if (numeric) { x = parseFloat(x); y = parseFloat(y); return asc ? x - y : y - x; }
      return asc ? x.localeCompare(y) : y.localeCompare(x);
    });
    rows.forEach(function (r) { body.appendChild(r); });
  });
});
"""


def load_results(results_folder: str) -> list:
    documents = []
    if not os.path.isdir(results_folder):
        return documents
    for name in sorted(os.listdir(results_folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(results_folder, name), "r", encoding="utf-8") as f:
            document = json.load(f)
        document["_slug"] = name[:-5]
        documents.append(document)
    return documents


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _link(url: str, text: str) -> str:
    if url and url.startswith("https://"):
        return f'<a href="{_e(url)}" rel="noopener">{_e(text)}</a>'
    return _e(text)


def render_duels(documents: list, duels: list) -> str:
    """The duel rating table and, for a handful of bots, the grid of every pairing."""
    if not duels:
        return ""
    names = {d["_slug"]: d["manifest"]["name"] for d in documents if "manifest" in d}
    points = {}
    played = {}
    for duel in duels:
        a, b = duel["a"], duel["b"]
        points[(a, b)] = points.get((a, b), 0) + duel["a_points"]
        points[(b, a)] = points.get((b, a), 0) + duel["b_points"]
        played[a] = played.get(a, 0) + 1
        played[b] = played.get(b, 0) + 1
    rating = ratings(points)
    order = sorted(rating, key=lambda slug: -rating[slug])

    rows = []
    for rank, slug in enumerate(order, start=1):
        won = sum(v for (x, _), v in points.items() if x == slug)
        lost = sum(v for (_, y), v in points.items() if y == slug)
        rows.append(
            f'<tr><td data-value="{rank}">{rank}</td>'
            f'<td class="text" data-value="{_e(names.get(slug, slug).lower())}">{_e(names.get(slug, slug))}</td>'
            f'<td class="overall" data-value="{rating[slug]}">{rating[slug]:.0f}</td>'
            f'<td data-value="{played.get(slug, 0)}">{played.get(slug, 0)}</td>'
            f'<td data-value="{won}">{won}</td><td data-value="{lost}">{lost}</td></tr>'
        )
    table = ('<div class="card"><table><thead><tr><th data-sort="number">#</th><th class="text" data-sort="text">Bot</th>'
             '<th data-sort="number">Rating</th><th data-sort="number">Opponents</th><th data-sort="number">Points won</th>'
             f'<th data-sort="number">Points lost</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    grid = ""
    if 2 <= len(order) <= 12:
        by_pair = {(d["a"], d["b"]): d for d in duels}
        head = "".join(f'<th title="{_e(names.get(s, s))}">{rank + 1}</th>' for rank, s in enumerate(order))
        body = []
        for i, a in enumerate(order):
            cells = []
            for b in order:
                if a == b:
                    cells.append('<td class="small">–</td>')
                    continue
                duel = by_pair.get((min(a, b), max(a, b)))
                if duel is None:
                    cells.append('<td class="small">not yet</td>')
                    continue
                mine, theirs = (duel["a_points"], duel["b_points"]) if a == duel["a"] else (duel["b_points"], duel["a_points"])
                pair = duel["a"] + "__" + duel["b"]
                title = ", ".join(f"{DUEL_KINDS[k]}: {v['a_points'] if a == duel['a'] else v['b_points']}-{v['b_points'] if a == duel['a'] else v['a_points']}" for k, v in duel["kinds"].items())
                cells.append(f'<td title="{_e(title)}"><a href="replay.html?duel={_e(pair)}">{mine}-{theirs}</a></td>')
            body.append(f'<tr><td class="text"><strong>{i + 1}</strong> {_e(names.get(a, a))}</td>{"".join(cells)}</tr>')
        grid = (f'<h2>Every pairing</h2><div class="card"><table><thead><tr><th class="text">Points for the row bot against…</th>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>'
                '<p class="small">Each pairing plays 200 episodes per direction of each duel kind. Click a result to watch it.</p>')

    return (f'<h2>Duels</h2><p class="lead">Bots play each other directly: a penalty duel, attacking then defending, and a kickoff duel. '
            f'The rating is fitted to every point won and lost; 400 points of difference means winning about ten points in eleven.</p>'
            f'{table}{grid}')


def render(documents: list, duels: list = None) -> str:
    scored = [d for d in documents if "results" in d and "error" not in d]
    failed = [d for d in documents if "error" in d]
    scored.sort(key=lambda d: (-d["overall_score"], d["manifest"]["name"].lower()))

    task_keys = list(TASKS)
    head = ['<th data-sort="number">#</th>', '<th class="text" data-sort="text">Bot</th>', '<th class="text" data-sort="text">Author</th>',
            '<th data-sort="number">Overall</th>']
    head += [f'<th data-sort="number" title="{_e(TASKS[key].description)}">{_e(TASKS[key].name)}</th>' for key in task_keys]
    head += ['<th data-sort="text">Scored</th>', '<th class="text">Replays</th>']

    rows = []
    for rank, document in enumerate(scored, start=1):
        manifest = document["manifest"]
        by_task = {r["task"]: r for r in document["results"]}
        name = _link(manifest.get("homepage", ""), manifest["name"])
        if manifest.get("public_model_url"):
            name += f' <span class="small">({_link(manifest["public_model_url"], "model")})</span>'
        cells = [
            f'<td data-value="{rank}">{rank}</td>',
            f'<td class="text" data-value="{_e(manifest["name"].lower())}">{name}'
            + (f'<br><span class="small">{_e(manifest["description"])}</span>' if manifest.get("description") else "") + "</td>",
            f'<td class="text" data-value="{_e(manifest["author"].lower())}">{_e(manifest["author"])}</td>',
            f'<td class="overall" data-value="{document["overall_score"]:.4f}">{document["overall_score"]:.1f}</td>',
        ]
        for key in task_keys:
            result = by_task.get(key)
            if result is None:
                cells.append('<td data-value="-1">–</td>')
                continue
            title = f"{result['successes']} of {result['episodes']} episodes, 95% interval {result['success_rate_low']:.1%} to {result['success_rate_high']:.1%}"
            if result["mean_seconds_to_score"]:
                title += f", {result['mean_seconds_to_score']:.1f} s to score on average"
            cells.append(f'<td data-value="{result["success_rate"]:.6f}" title="{_e(title)}">{result["success_rate"]:.1%}</td>')
        cells.append(f'<td data-value="{_e(document["scored_at"])}"><span class="small">{_e(document["scored_at"][:10])}</span></td>')
        if document.get("_has_replays"):
            cells.append(f'<td class="text"><a href="replay.html?bot={_e(document["_slug"])}">Watch</a></td>')
        else:
            cells.append('<td class="text"><span class="small">–</span></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    if rows:
        table = f'<div class="card"><table><thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    else:
        table = '<div class="card"><p>No bots have been scored yet.</p></div>'

    failed_html = ""
    if failed:
        items = "".join(
            f'<li><strong>{_e(d.get("manifest", {}).get("name", d["_slug"]))}</strong>: <span class="error">{_e(d["error"])}</span></li>'
            for d in failed
        )
        failed_html = f"<h2>Not scored</h2><div class=\"card\"><ul>{items}</ul></div>"

    built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boost Arena leaderboard</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>Boost Arena</h1>
<p class="lead">An open benchmark for car-football bots. Every bot faces the same six tasks, 1,000 times each, in the RocketSim simulator.
<a href="https://github.com/yogyam/boost-arena">How to enter</a> · <a href="https://github.com/yogyam/boost-arena/blob/main/docs/TASKS.md">The tasks</a></p>
{table}
<p class="small">Success rates over 1,000 episodes per task. Overall is the average of the six, out of 100. Hover a score for its confidence interval: differences smaller than it are not meaningful. Click a column heading to sort. Replays show the first five episodes of each task, which are the same situations for every bot.</p>
{failed_html}
{render_duels(documents, duels or [])}
<footer><p>{DISCLAIMER}</p><p>Built {built}.</p></footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""


def load_duels(duels_folder: str) -> list:
    duels = []
    if duels_folder and os.path.isdir(duels_folder):
        for name in sorted(os.listdir(duels_folder)):
            if name.endswith(".json"):
                with open(os.path.join(duels_folder, name), "r", encoding="utf-8") as f:
                    duels.append(json.load(f))
    return duels


def build_site(results_folder: str, output_folder: str, replays_folder: str = None, duels_folder: str = None) -> str:
    documents = load_results(results_folder)
    duels = load_duels(duels_folder)
    os.makedirs(os.path.join(output_folder, "replays", "duels"), exist_ok=True)

    for duel in duels:
        source = os.path.join(duels_folder, duel["a"] + "__" + duel["b"] + REPLAYS_SUFFIX)
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(output_folder, "replays", "duels", os.path.basename(source)))

    # Replays are copied next to the page, one compressed file per bot
    for document in documents:
        source = os.path.join(replays_folder, document["_slug"] + REPLAYS_SUFFIX) if replays_folder else None
        document["_has_replays"] = bool(source and os.path.isfile(source))
        if document["_has_replays"]:
            shutil.copyfile(source, os.path.join(output_folder, "replays", document["_slug"] + REPLAYS_SUFFIX))

    path = os.path.join(output_folder, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(documents, duels))

    template = os.path.join(os.path.dirname(__file__), "replay.html")
    with open(template, "r", encoding="utf-8") as f:
        page = f.read().replace("__TASK_NAMES__", json.dumps({key: task.name for key, task in TASKS.items()}))
    with open(os.path.join(output_folder, "replay.html"), "w", encoding="utf-8") as f:
        f.write(page)

    # The raw results are published too, for anyone who wants to make their own charts
    with open(os.path.join(output_folder, "results.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in d.items() if not k.startswith("_")} | {"slug": d["_slug"], "has_replays": d.get("_has_replays", False)} for d in documents], f, indent=1)
    return path
