"""Download the demo's browser files (styles, fonts, scripts) once, so the demo also works offline.

Solara fetches its user-interface files from a CDN the first time a page asks for them and keeps
them in ``<venv>/share/solara/cdn``. If that first download is slow or there is no internet, the
page renders without styling. Running this script once (``setup.bat`` / ``setup.sh`` do it, and
``run_demo.bat`` / ``run.sh demo`` check it) puts every file in that cache in advance.

Usage::

    python prepare_demo_assets.py          # download whatever is missing
    python prepare_demo_assets.py --check  # exit 1 if anything is missing, download nothing
"""

from __future__ import annotations

import re
import sys

import solara.settings
from solara.server import settings as server_settings
from solara.server.cdn_helper import fetch_from_cdn, get_from_cache, put_in_cache

# The files Solara 1.62.1 asks for (the versions are pinned in requirements.txt).
VUETIFY = "@widgetti/solara-vuetify3-app@5.2.0/dist"
FILES = [
    f"{VUETIFY}/main8.css",
    f"{VUETIFY}/solara-vuetify-app8.js",
    f"{VUETIFY}/solara-vuetify-app8.min.js",  # used by 'solara run --production'
    f"{VUETIFY}/fonts.css",
    "font-awesome@4.5.0/css/font-awesome.min.css",
    "katex@0.16.9/dist/katex.min.css",
    "katex@0.16.9/dist/katex.min.js",
    "katex@0.16.9/dist/contrib/auto-render.min.js",
    "mermaid@10.8.0/dist/mermaid.min.js",
    "requirejs@2.3.6/require.js",
]


def _cached(path: str) -> bytes | None:
    try:
        return get_from_cache(server_settings.assets.proxy_cache_dir, path)
    except Exception:
        return None


def _fonts(css_path: str, css: bytes) -> list[str]:
    """Font files a stylesheet links to, as cache paths next to the stylesheet."""
    folder = css_path.rsplit("/", 1)[0]
    names = re.findall(r"url\(\s*[\"']?([^)\"'?#]+\.(?:woff2|woff|ttf|eot|otf))", css.decode("utf-8", "replace"))
    out = []
    for name in dict.fromkeys(names):
        parts = folder.split("/")
        for piece in name.split("/"):
            if piece == "..":
                parts.pop()
            elif piece not in ("", "."):
                parts.append(piece)
        out.append("/".join(parts))
    return out


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    cache = server_settings.assets.proxy_cache_dir
    cdn = solara.settings.assets.cdn.rstrip("/") + "/"
    todo = list(FILES)
    missing: list[str] = []
    seen: set[str] = set()
    done = 0
    while todo:
        path = todo.pop(0)
        data = _cached(path)
        if data is None and not check_only:
            try:
                data = fetch_from_cdn(cdn + path)
                put_in_cache(cache, path, data)
                done += 1
            except Exception as exc:  # no internet, CDN down, ...
                print(f"  could not download {path}: {exc}")
        if data is None:
            missing.append(path)
        elif path.endswith(".css"):
            todo += [f for f in _fonts(path, data) if f not in todo and f not in FILES and f not in seen]
        seen.add(path)
    if missing:
        print(
            f"Demo UI files missing ({len(missing)}): connect to the internet and run "
            "'python prepare_demo_assets.py' once, or the demo page may appear unstyled."
        )
        return 1
    print(f"Demo UI files ready in {cache} ({done} downloaded now).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
