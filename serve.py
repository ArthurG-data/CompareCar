#!/usr/bin/env python3
"""Serve analysis.html on the LAN and keep the household shortlist in data/shortlist.json."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHORTLIST = ROOT / "data" / "shortlist.json"
REFRESH_LOG = ROOT / "data" / "refresh.log"
PIPELINE = [
    ("Collecting listings from AutoTrader", [sys.executable, str(ROOT / "collect_autotrader.py")]),
    ("Fitting the price model", [sys.executable, str(ROOT / "model_prices.py")]),
    ("Rebuilding the page", [sys.executable, str(ROOT / "build_html.py")]),
]
MAX_BODY = 16 * 1024
_LOCK = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Refresher:
    """Runs the data pipeline steps one after another in a background thread, one run at a time."""

    def __init__(self, log_path, steps, cwd=None):
        self.log_path, self.steps, self.cwd = Path(log_path), steps, cwd
        self.lock = threading.Lock()
        self.thread = None
        self.state = {"running": False, "step": None, "step_no": 0, "n_steps": len(steps),
                      "started_at": None, "finished_at": None, "ok": None}

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return False
            self.state.update(running=True, step=self.steps[0][0], step_no=1,
                              started_at=_now(), finished_at=None, ok=None)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return True

    def _run(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        ok = True
        # unbuffered: the child processes write to the same fd, and a buffered wrapper's
        # cached offset would overwrite their output (leaving NUL-padded gaps)
        with open(self.log_path, "wb", buffering=0) as log:
            for i, (name, cmd) in enumerate(self.steps, 1):
                with self.lock:
                    self.state.update(step=name, step_no=i)
                log.write(f"=== {name} ===\n".encode())
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=self.cwd)
                if rc != 0:
                    log.write(f"\n!!! step failed with exit code {rc}\n".encode())
                    ok = False
                    break
        with self.lock:
            self.state.update(running=False, step=None, finished_at=_now(), ok=ok)

    def status(self):
        try:
            tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            tail = ""
        with self.lock:
            return {**self.state, "log_tail": tail}


def load_items(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _save(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def upsert(path, item):
    with _LOCK:
        items = load_items(path)
        for i, cur in enumerate(items):
            if cur["id"] == item["id"]:
                items[i] = {**cur, **item, "saved_at": cur["saved_at"]}
                break
        else:
            items.append({**item, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        _save(path, items)
        return items


def remove(path, item_id):
    with _LOCK:
        items = [i for i in load_items(path) if i["id"] != item_id]
        _save(path, items)
        return items


class Handler(SimpleHTTPRequestHandler):
    shortlist = SHORTLIST
    refresher = None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/analysis.html")
            self.end_headers()
        elif self.path == "/api/shortlist":
            self._json(200, {"items": load_items(self.shortlist)})
        elif self.path == "/api/refresh":
            self._json(200, self.refresher.status())
        elif self.path.startswith("/api/"):
            self._json(404, {"error": "not found"})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/refresh":
            started = self.refresher.start()
            return self._json(202 if started else 409, self.refresher.status())
        if self.path != "/api/shortlist":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n <= 0 or n > MAX_BODY:
            return self._json(400, {"error": "bad body size"})
        try:
            item = json.loads(self.rfile.read(n))
        except ValueError:
            return self._json(400, {"error": "invalid json"})
        if not isinstance(item, dict) or not item.get("id") or not item.get("url"):
            return self._json(400, {"error": "id and url required"})
        item["id"] = str(item["id"])
        self._json(200, {"items": upsert(self.shortlist, item)})

    def do_DELETE(self):
        prefix = "/api/shortlist/"
        if not self.path.startswith(prefix) or len(self.path) <= len(prefix):
            return self._json(404, {"error": "not found"})
        self._json(200, {"items": remove(self.shortlist, self.path[len(prefix):])})

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/"):
            super().log_message(fmt, *args)


def make_server(root, host, port, steps=PIPELINE):
    root = Path(root)
    handler = type("RootHandler", (Handler,), {
        "shortlist": root / "data" / "shortlist.json",
        "refresher": Refresher(root / "data" / "refresh.log", steps, cwd=str(root)),
    })
    return ThreadingHTTPServer((host, port), partial(handler, directory=str(root)))


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = make_server(ROOT, "0.0.0.0", port)
    print(f"Dashboard: http://{lan_ip()}:{port}/analysis.html", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
