"""Builds the leaderboard website from the result files: plain HTML, and nothing loaded from other hosts."""

import datetime
import html
import json
import os
import shutil

from .duels import DUEL_KINDS
from .rating import ratings
from .submissions import REPLAYS_SUFFIX
from .tasks import TASKS

REPOSITORY = "https://github.com/yogyam/boost-arena"

DISCLAIMER = (
    "Boost Arena is a fan project and is not affiliated with Psyonix or Epic Games. "
    "Portions of the materials used are trademarks and/or copyrighted works of Epic Games, Inc. "
    "All rights reserved by Epic. This material is not official and is not endorsed by Epic."
)

STYLE = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11, 11, 11, 0.10); --wash: rgba(11, 11, 11, 0.05);
  --series: #2a78d6; --series-soft: #d5e5f9; --good: #006300;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255, 255, 255, 0.10); --wash: rgba(255, 255, 255, 0.07);
    --series: #3987e5; --series-soft: #1c3b63; --good: #0ca30c;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 0 16px 56px; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
a { color: var(--series); }
header { padding: 36px 0 20px; display: flex; flex-wrap: wrap; gap: 16px 32px; align-items: flex-end; justify-content: space-between; }
header h1 { margin: 0; font-size: 34px; font-weight: 700; letter-spacing: -0.01em; }
header p { margin: 6px 0 0; color: var(--ink-2); max-width: 640px; }
nav { display: flex; flex-wrap: wrap; gap: 6px; }
nav a { color: var(--ink); text-decoration: none; padding: 7px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); font-size: 14px; }
nav a:hover { background: var(--wash); }
nav a.primary { background: var(--series); border-color: var(--series); color: #fff; }
h2 { margin: 36px 0 10px; font-size: 20px; }
p.lead { margin: 0 0 14px; color: var(--ink-2); }
.small { color: var(--muted); font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 8px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; overflow-x: auto; }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; margin: 0 0 10px; }
.toolbar input { font: inherit; padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--ink); min-width: 240px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 7px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; vertical-align: middle; }
th { color: var(--ink-2); font-weight: 600; cursor: pointer; user-select: none; font-size: 13px; }
th.text, td.text { text-align: left; }
tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: var(--wash); }
td.rank { color: var(--ink-2); width: 40px; }
td.rank.top { color: var(--ink); font-weight: 700; }
.bot { font-weight: 600; }
.desc { color: var(--ink-2); font-size: 12.5px; white-space: normal; max-width: 300px; }
.bar { display: inline-flex; align-items: center; gap: 6px; justify-content: flex-end; min-width: 72px; }
.bar i { display: inline-block; height: 8px; width: 32px; background: var(--series-soft); border-radius: 0 4px 4px 0; overflow: hidden; position: relative; }
.bar i b { position: absolute; left: 0; top: 0; bottom: 0; background: var(--series); border-radius: 0 4px 4px 0; }
.bar.overall i { width: 72px; height: 10px; }
.bar.overall span { font-weight: 700; }
.error { color: #b3261e; }
.flag { font-size: 11px; font-weight: 600; color: #8a4b00; background: #fff1dc; border-radius: 4px; padding: 1px 6px; text-decoration: none; vertical-align: middle; }
svg.chart { display: block; width: 100%; height: auto; }
.chart text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink-2); }
.chart text.value { fill: var(--ink); font-weight: 600; }
.chart text.name { fill: var(--ink); }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.step { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.step b { display: block; font-size: 13px; color: var(--series); margin-bottom: 4px; }
.step code { font-size: 12.5px; background: var(--wash); padding: 2px 6px; border-radius: 4px; white-space: nowrap; }
footer { margin-top: 44px; color: var(--muted); font-size: 12px; }
@media (max-width: 640px) { header h1 { font-size: 26px; } .desc { max-width: 240px; } }
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
var search = document.getElementById("search");
if (search) {
  search.addEventListener("input", function () {
    var needle = search.value.trim().toLowerCase();
    document.querySelectorAll("#board tbody tr").forEach(function (row) {
      row.style.display = !needle || row.dataset.search.indexOf(needle) !== -1 ? "" : "none";
    });
  });
}
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


def load_duels(duels_folder: str) -> list:
    duels = []
    if duels_folder and os.path.isdir(duels_folder):
        for name in sorted(os.listdir(duels_folder)):
            if name.endswith(".json"):
                with open(os.path.join(duels_folder, name), "r", encoding="utf-8") as f:
                    duels.append(json.load(f))
    return duels


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _link(url: str, text: str) -> str:
    if url and url.startswith("https://"):
        return f'<a href="{_e(url)}" rel="noopener">{_e(text)}</a>'
    return _e(text)


def _bar(fraction: float, label: str, overall: bool = False, title: str = "") -> str:
    """A number with a small inline bar, so a column can be read at a glance."""
    width = max(0.0, min(1.0, fraction)) * 100
    cls = "bar overall" if overall else "bar"
    return f'<span class="{cls}" title="{_e(title)}"><i><b style="width:{width:.1f}%"></b></i><span>{_e(label)}</span></span>'


def _tiles(items: list) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div></div>' for label, value in items
    ) + "</div>"


def _overall_chart(scored: list) -> str:
    """Horizontal bars of overall score, one per bot. The leaderboard table is its table view."""
    if not scored:
        return ""
    width, row_height, left, right = 960, 30, 240, 70
    height = row_height * len(scored) + 36
    plot = width - left - right
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Overall score of every bot">']
    for tick in (0, 25, 50, 75, 100):
        x = left + plot * tick / 100
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="8" y2="{height - 26}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle">{tick}</text>')
    out.append(f'<line x1="{left}" x2="{left}" y1="8" y2="{height - 26}" stroke="var(--axis)" stroke-width="1"/>')
    for i, document in enumerate(scored):
        y = 12 + i * row_height
        value = document["overall_score"]
        bar_width = plot * value / 100
        name = document["manifest"]["name"]
        label = name if len(name) <= 34 else name[:33] + "…"
        out.append(f'<text class="name" x="{left - 10}" y="{y + 14}" text-anchor="end"><title>{_e(name)}</title>{_e(label)}</text>')
        out.append(f'<rect x="{left}" y="{y + 4}" width="{bar_width:.1f}" height="12" rx="4" fill="var(--series)">'
                   f'<title>{_e(name)}: {value:.1f} out of 100</title></rect>')
        # Keep the baseline end square: only the data end of the bar is rounded
        if bar_width > 4:
            out.append(f'<rect x="{left}" y="{y + 4}" width="4" height="12" fill="var(--series)"/>')
        out.append(f'<text class="value" x="{left + bar_width + 8:.1f}" y="{y + 14}">{value:.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def render_board(scored: list, failed: list) -> tuple:
    task_keys = list(TASKS)
    head = ['<th data-sort="number">#</th>', '<th class="text" data-sort="text">Bot</th>', '<th class="text" data-sort="text">Author</th>',
            '<th data-sort="number">Overall</th>']
    head += [f'<th data-sort="number" title="{_e(TASKS[key].description)}">{_e(TASKS[key].name)}</th>' for key in task_keys]
    head += ['<th class="text">Replays</th>']

    rows = []
    for rank, document in enumerate(scored, start=1):
        manifest = document["manifest"]
        by_task = {r["task"]: r for r in document["results"]}
        name = f'<span class="bot">{_link(manifest.get("homepage", ""), manifest["name"])}</span>'
        if manifest.get("public_model_url"):
            name += f' <span class="small">({_link(manifest["public_model_url"], "model")})</span>'
        if document.get("_flag"):
            name += f' <a class="flag" href="{_e(document["_flag"])}" title="This entry is under question; click for the discussion">Flagged</a>'
        if manifest.get("description"):
            name += f'<div class="desc">{_e(manifest["description"])}</div>'
        overall = document["overall_score"]
        cells = [
            f'<td class="rank{" top" if rank <= 3 else ""}" data-value="{rank}">{rank}</td>',
            f'<td class="text" data-value="{_e(manifest["name"].lower())}">{name}</td>',
            f'<td class="text" data-value="{_e(manifest["author"].lower())}">{_e(manifest["author"])}'
            f'<div class="small" title="Date of the official score">{_e(document["scored_at"][:10])}</div></td>',
            f'<td data-value="{overall:.4f}">{_bar(overall / 100, f"{overall:.1f}", True, "Average success rate over the six tasks, out of 100")}</td>',
        ]
        for key in task_keys:
            result = by_task.get(key)
            if result is None:
                cells.append('<td data-value="-1">–</td>')
                continue
            title = f"{result['successes']} of {result['episodes']} episodes. 95% interval {result['success_rate_low']:.1%} to {result['success_rate_high']:.1%}"
            if result["mean_seconds_to_score"]:
                title += f". {result['mean_seconds_to_score']:.1f} s to score on average"
            rate = result["success_rate"]
            cells.append(f'<td data-value="{rate:.6f}">{_bar(rate, f"{rate:.1%}", False, title)}</td>')
        if document.get("_has_replays"):
            cells.append(f'<td class="text"><a href="replay.html?bot={_e(document["_slug"])}">Watch</a></td>')
        else:
            cells.append('<td class="text"><span class="small">–</span></td>')
        search = f'{manifest["name"]} {manifest["author"]} {manifest.get("description", "")}'.lower()
        rows.append(f'<tr data-search="{_e(search)}">' + "".join(cells) + "</tr>")

    if rows:
        table = f'<div class="card"><table id="board"><thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    else:
        table = '<div class="card"><p>No bots have been scored yet.</p></div>'

    failed_html = ""
    if failed:
        items = "".join(
            f'<li><strong>{_e(d.get("manifest", {}).get("name", d["_slug"]))}</strong>: <span class="error">{_e(d["error"])}</span></li>'
            for d in failed
        )
        failed_html = f'<h2>Not scored</h2><div class="card"><ul>{items}</ul></div>'
    return table, failed_html


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
    top = max(rating.values()) if rating else 1

    rows = []
    for rank, slug in enumerate(order, start=1):
        won = sum(v for (x, _), v in points.items() if x == slug)
        lost = sum(v for (_, y), v in points.items() if y == slug)
        share = won / (won + lost) if won + lost else 0
        rows.append(
            f'<tr><td class="rank{" top" if rank <= 3 else ""}" data-value="{rank}">{rank}</td>'
            f'<td class="text" data-value="{_e(names.get(slug, slug).lower())}"><span class="bot">{_e(names.get(slug, slug))}</span></td>'
            f'<td data-value="{rating[slug]}">{_bar(rating[slug] / top, f"{rating[slug]:.0f}", True, "Bradley-Terry rating; the average bot is near 1000")}</td>'
            f'<td data-value="{played.get(slug, 0)}">{played.get(slug, 0)}</td>'
            f'<td data-value="{share:.4f}">{_bar(share, f"{share:.0%}", False, f"{won} points won, {lost} lost")}</td></tr>'
        )
    table = ('<div class="card"><table><thead><tr><th data-sort="number">#</th><th class="text" data-sort="text">Bot</th>'
             '<th data-sort="number">Rating</th><th data-sort="number">Opponents</th><th data-sort="number">Points won</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

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
                first = a == duel["a"]
                mine, theirs = (duel["a_points"], duel["b_points"]) if first else (duel["b_points"], duel["a_points"])
                pair = duel["a"] + "__" + duel["b"]
                title = ", ".join(
                    f"{DUEL_KINDS[k]}: {v['a_points'] if first else v['b_points']}-{v['b_points'] if first else v['a_points']}"
                    for k, v in duel["kinds"].items()
                )
                colour = "var(--good)" if mine > theirs else "var(--ink-2)"
                cells.append(f'<td title="{_e(title)}"><a href="replay.html?duel={_e(pair)}" style="color:{colour}">{mine}-{theirs}</a></td>')
            body.append(f'<tr><td class="text"><strong>{i + 1}</strong> {_e(names.get(a, a))}</td>{"".join(cells)}</tr>')
        grid = (f'<h2>Every pairing</h2><div class="card"><table><thead><tr><th class="text">Points for the row bot against…</th>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>'
                '<p class="small">Each pairing plays 200 episodes per direction of each duel kind. Green means the row bot won the pairing. Click a result to watch it.</p>')

    return (f'<h2 id="duels">Duels</h2><p class="lead">Scored bots play each other directly: a penalty duel, attacking then defending, and a kickoff duel. '
            f'The rating is fitted to every point won and lost; 400 points of difference means winning about ten points in eleven. '
            f'<a href="{REPOSITORY}/blob/main/docs/DUELS.md">How duels work</a></p>{table}{grid}')


def render(documents: list, duels: list = None) -> str:
    duels = duels or []
    scored = [d for d in documents if "results" in d and "error" not in d]
    failed = [d for d in documents if "error" in d]
    scored.sort(key=lambda d: (-d["overall_score"], d["manifest"]["name"].lower()))
    table, failed_html = render_board(scored, failed)

    last = max((d["scored_at"] for d in scored), default="")
    tiles = _tiles([
        ("Bots", str(len(scored))),
        ("Tasks", str(len(TASKS))),
        ("Episodes per bot", f"{len(TASKS) * 1000:,}"),
        ("Duels played", str(len(duels))),
        ("Last scored", last[:10] or "–"),
    ])

    chart = f'<h2>Scores at a glance</h2><div class="card">{_overall_chart(scored)}</div>' if scored else ""

    enter = f'''<h2 id="enter">Enter your bot</h2>
<div class="steps">
  <div class="step"><b>1 · Train</b>Any framework, any method. No bot yet? The <a href="{REPOSITORY}/blob/main/docs/STARTER_KIT.md">starter kit</a> trains one on a laptop in a couple of hours.</div>
  <div class="step"><b>2 · Seal</b>Export the policy as ONNX, then <code>boost-arena submit my_bot.onnx --github you</code>. The model is sealed so only the scoring service can open it; it is never published.</div>
  <div class="step"><b>3 · Submit</b>Host the sealed file and open a pull request with the manifest. Once merged, scoring, duels and replays follow on their own.</div>
</div>
<p class="small">Full instructions: <a href="{REPOSITORY}/blob/main/docs/SUBMITTING.md">SUBMITTING.md</a>. The interface every bot uses: <a href="{REPOSITORY}/blob/main/docs/INTERFACE.md">INTERFACE.md</a>. The rules: <a href="{REPOSITORY}/blob/main/RULES.md">RULES.md</a>.</p>'''

    built = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Boost Arena leaderboard</title>
<meta name="description" content="An open, task-based benchmark for car-football bots, scored in the RocketSim simulator.">
<style>{STYLE}</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>Boost Arena</h1>
    <p>An open benchmark for car-football bots. Every bot faces the same six tasks, 1,000 times each, in the RocketSim simulator, then plays the other bots head to head.</p>
  </div>
  <nav>
    <a class="primary" href="#enter">Enter your bot</a>
    <a href="{REPOSITORY}/blob/main/docs/TASKS.md">The tasks</a>
    <a href="#duels">Duels</a>
    <a href="{REPOSITORY}">GitHub</a>
  </nav>
</header>
{tiles}
<h2 id="leaderboard">Leaderboard</h2>
<div class="toolbar"><p class="lead" style="margin:0">Success rate per task over 1,000 episodes. Overall is the average of the six, out of 100.</p>
<input id="search" type="search" placeholder="Search bots and authors" aria-label="Search bots and authors"></div>
{table}
<p class="small">Hover a score for its confidence interval: differences smaller than it are not meaningful. Click a column heading to sort. Replays show the first five episodes of each task, which are the same situations for every bot.</p>
{failed_html}
{chart}
{render_duels(documents, duels)}
{enter}
<footer><p>{DISCLAIMER}</p><p>Built {built}. Raw data: <a href="results.json">results.json</a>. Code: <a href="{REPOSITORY}">{REPOSITORY}</a>, MIT licence.</p></footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""


def load_flags(path: str) -> dict:
    """`flags.json` maps a slug to the https address of the discussion about that entry (see RULES.md)."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        flags = json.load(f)
    return {slug: url for slug, url in flags.items() if isinstance(url, str) and url.startswith("https://")}


def build_site(results_folder: str, output_folder: str, replays_folder: str = None, duels_folder: str = None,
               flags_path: str = None) -> str:
    documents = load_results(results_folder)
    duels = load_duels(duels_folder)
    flags = load_flags(flags_path)
    for document in documents:
        if document["_slug"] in flags:
            document["_flag"] = flags[document["_slug"]]
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

    # The viewer's one library is served by the site itself, not fetched from another host
    os.makedirs(os.path.join(output_folder, "vendor"), exist_ok=True)
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "vendor", "three.module.js"),
                    os.path.join(output_folder, "vendor", "three.module.js"))

    # The raw results are published too, for anyone who wants to make their own charts
    with open(os.path.join(output_folder, "results.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in d.items() if not k.startswith("_")} | {"slug": d["_slug"], "has_replays": d.get("_has_replays", False)} for d in documents], f, indent=1)
    return path
