"""Build the static UI for Vercel deployment.

Usage:
    python hybrid/3_ui/build.py --api https://<tunnel-url>
    cd hybrid/3_ui/teach-help && vercel deploy --prod --yes

The backend URL can also be provided through the ?api=... URL parameter.
Vercel only serves static files; model inference runs on the backend.
"""

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
UI = HERE.parent.parent / "ui"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--api",
        default="",
        help="Permanent backend URL (optional; usually provided through ?api=...)"
    )
    args = p.parse_args()

    dist = HERE / "teach-help"
    dist.mkdir(exist_ok=True)

    for old in dist.iterdir():
        # Keep .vercel to preserve the existing Vercel project connection.
        if old.name != ".vercel":
            shutil.rmtree(old) if old.is_dir() else old.unlink()

    shutil.copy(UI / "index.html", dist)

    (dist / "config.js").write_text(
        f"window.API_BASE = {json.dumps(args.api.rstrip('/'))};\n",
        encoding="utf-8"
    )

    # Prevent browser caching when the backend URL changes.
    (dist / "vercel.json").write_text(
        json.dumps({
            "headers": [{
                "source": "/config.js",
                "headers": [{
                    "key": "Cache-Control",
                    "value": "no-store"
                }]
            }]
        }, indent=2),
        encoding="utf-8"
    )

    print("Ready:", dist, "| backend:", args.api or "provided through ?api=...")


if __name__ == "__main__":
    main()
