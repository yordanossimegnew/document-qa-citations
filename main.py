"""Entry point.

    uvicorn main:app --reload                         # API (docs at /docs)
    python main.py add report.pdf notes.md            # index files from the terminal
    python main.py add https://example.com/page       # index a web page
    python main.py ask "What does the report conclude?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.api import create_app
from src.utils import load_settings, setup_logging

settings = load_settings()
setup_logging(settings.log_level, settings.log_dir)
app = create_app(settings)

BOLD, DIM, CYAN, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"


def main() -> None:
    from fastapi.testclient import TestClient

    parser = argparse.ArgumentParser(description="Document QA with citations")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Run the FastAPI server (default).")
    serve.add_argument("--port", type=int, default=8000)
    add = sub.add_parser("add", help="Index files or URLs.")
    add.add_argument("items", nargs="+")
    ask = sub.add_parser("ask", help="Ask a question.")
    ask.add_argument("question")
    args = parser.parse_args()

    if args.command in ("add", "ask"):
        client = TestClient(app)
        if args.command == "add":
            for item in args.items:
                if item.startswith(("http://", "https://")):
                    r = client.post("/documents/url", json={"url": item})
                else:
                    r = client.post("/documents", files=[("files", (Path(item).name, Path(item).read_bytes()))])
                body = r.json()
                docs = body if isinstance(body, list) else [body]
                if r.status_code >= 400:
                    print(f"{YELLOW}{item}: {body.get('detail')}{RESET}")
                for d in docs if r.status_code < 400 else []:
                    print(f"{GREEN}✓{RESET} {d['title']} {DIM}({d['source_type']}, {d['chunks']} chunks){RESET}")
        else:
            r = client.post("/ask", json={"question": args.question})
            body = r.json()
            if r.status_code >= 400:
                sys.exit(body.get("detail"))
            print(f"\n{body['markdown']}\n")
            for s in body["sources"]:
                print(f"{CYAN}[{s['n']}]{RESET} {BOLD}{s['label']}{RESET}")
                for q in s["quotes"]:
                    print(f"    {DIM}“{q['text'].strip()[:200]}”{RESET}")
        return

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=getattr(args, "port", 8000))


if __name__ == "__main__":
    main()
