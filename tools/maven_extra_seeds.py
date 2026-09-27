"""Generate extra Maven seed artifacts from group directory listings.

Why this exists: the recovered code contains no seed list for the Maven graph, so
the node set has to be reconstructed.  A BFS from a hand-written list of popular
artifacts reaches ~1,150 artifacts, while the published graph has 1,484 nodes and
a higher average out-degree (3.7 vs 2.7 for npm), which is what one expects from
the sibling artifacts of the same groups (``spring-boot-starter-*`` and friends)
being present as well.  This tool therefore extends the seed set with the other
artifacts of the groups that already appear in the crawl.

    python tools/maven_extra_seeds.py --per-group 20 --max-groups 80 --max-total 1200

The output file is consumed by ``fetch_maven.py`` through ``SEEDS_FILE``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_maven"
OUT = ROOT / "data" / "seeds" / "maven_extra.txt"
BASE = "https://repo1.maven.org/maven2/"
UA = "Mozilla/5.0 (compatible; dependency-graph-research/1.0)"
_HREF_RE = re.compile(r'href="(?P<name>[^"/]+)/"')


def known_artifacts() -> tuple[Counter, Counter]:
    """Rank groups by how many of their artifacts we hold and by dep frequency."""
    have = Counter()
    want = Counter()
    for path in RAW.glob("*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        g = rec.get("group")
        if g:
            have[g] += 1
        for dep in (rec.get("deps") or {}):
            if ":" in dep:
                want[dep.split(":", 1)[0]] += 1
    return have, want


def list_group(group: str) -> list[str]:
    url = BASE + group.replace(".", "/") + "/"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []
    names = {m.group("name") for m in _HREF_RE.finditer(html)}
    return sorted(n for n in names if n not in {"..", "maven-metadata.xml"} and not n.startswith("."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--max-groups", type=int, default=80)
    ap.add_argument("--max-total", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    have, want = known_artifacts()
    score = {g: have[g] + min(want[g], 40) for g in set(have) | set(want)}
    groups = [g for g, _ in sorted(score.items(), key=lambda kv: -kv[1])][: args.max_groups]
    print(f"known artifacts: {sum(have.values())} in {len(have)} groups; "
          f"scanning top {len(groups)} groups", flush=True)

    out: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for group, names in zip(groups, ex.map(list_group, groups)):
            added = 0
            for art in names:
                if added >= args.per_group or len(out) >= args.max_total:
                    break
                out.append(f"{group}:{art}")
                added += 1
            print(f"  {group}: {len(names)} artifacts listed, {added} seeded "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # already-held artifacts are cheap (cached), so keep them in the list too
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(sorted(set(out))) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(set(out))} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
