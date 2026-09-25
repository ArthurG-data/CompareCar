import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import serve


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "shortlist.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_is_empty(self):
        self.assertEqual(serve.load_items(self.path), [])

    def test_upsert_adds_saved_at_and_persists(self):
        items = serve.upsert(self.path, {"id": "a1", "url": "https://x/a1", "price": 100})
        self.assertEqual(len(items), 1)
        self.assertIn("saved_at", items[0])
        self.assertEqual(json.loads(self.path.read_text())[0]["id"], "a1")
        self.assertFalse(list(self.path.parent.glob("*.tmp*")))

    def test_upsert_same_id_updates_but_keeps_saved_at(self):
        first = serve.upsert(self.path, {"id": "a1", "url": "u", "price": 100})[0]
        items = serve.upsert(self.path, {"id": "a1", "url": "u", "price": 90})
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["price"], 90)
        self.assertEqual(items[0]["saved_at"], first["saved_at"])

    def test_remove(self):
        serve.upsert(self.path, {"id": "a1", "url": "u"})
        serve.upsert(self.path, {"id": "b2", "url": "u"})
        items = serve.remove(self.path, "a1")
        self.assertEqual([i["id"] for i in items], ["b2"])
        self.assertEqual(serve.remove(self.path, "zzz"), items)


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "refresh.log"

    def tearDown(self):
        self.tmp.cleanup()

    def wait(self, r):
        for _ in range(100):
            if not r.status()["running"]:
                return r.status()
            time.sleep(0.02)
        self.fail("refresh did not finish")

    def test_runs_steps_in_order_and_reports_ok(self):
        r = serve.Refresher(self.log, [("one", ["sh", "-c", "echo A"]), ("two", ["sh", "-c", "echo B"])])
        self.assertTrue(r.start())
        st = self.wait(r)
        self.assertTrue(st["ok"])
        self.assertEqual(st["step"], None)
        self.assertTrue(st["finished_at"])
        self.assertNotIn("\x00", st["log_tail"])
        self.assertEqual(st["log_tail"], "=== one ===\nA\n=== two ===\nB\n")

    def test_second_start_rejected_while_running(self):
        r = serve.Refresher(self.log, [("slow", ["sh", "-c", "sleep 0.3"])])
        self.assertTrue(r.start())
        self.assertFalse(r.start())
        self.assertEqual(r.status()["step"], "slow")
        self.wait(r)
        self.assertTrue(r.start())
        self.wait(r)

    def test_failure_stops_pipeline(self):
        r = serve.Refresher(self.log, [("bad", ["sh", "-c", "echo boom; exit 3"]), ("never", ["sh", "-c", "echo NO"])])
        r.start()
        st = self.wait(r)
        self.assertFalse(st["ok"])
        self.assertIn("boom", st["log_tail"])
        self.assertNotIn("NO", st["log_tail"])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "analysis.html").write_text("<title>t</title>")
        cls.server = serve.make_server(root, "127.0.0.1", 0,
                                       steps=[("fake", ["sh", "-c", "sleep 0.2; echo done"])])
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def req(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, r.getheader("Content-Type"), raw

    def test_root_redirects_to_dashboard(self):
        status, _, _ = self.req("GET", "/")
        self.assertIn(status, (301, 302))

    def test_static_is_no_store(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request("GET", "/analysis.html")
        r = c.getresponse(); r.read(); c.close()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Cache-Control"), "no-store")

    def test_round_trip(self):
        status, ctype, raw = self.req("GET", "/api/shortlist")
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("application/json"))
        self.assertEqual(json.loads(raw), {"items": []})

        status, _, raw = self.req("POST", "/api/shortlist",
                                  {"id": "x1", "url": "https://x/1", "price": 100, "saved_by": "Al"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["items"][0]["saved_by"], "Al")

        status, _, raw = self.req("POST", "/api/shortlist", {"id": "x1", "url": "https://x/1", "price": 80})
        items = json.loads(raw)["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["price"], 80)

        status, _, raw = self.req("DELETE", "/api/shortlist/x1")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), {"items": []})

    def test_bad_bodies_are_400(self):
        self.assertEqual(self.req("POST", "/api/shortlist", b"not json")[0], 400)
        self.assertEqual(self.req("POST", "/api/shortlist", {"url": "u"})[0], 400)
        self.assertEqual(self.req("POST", "/api/shortlist", {"id": "q"})[0], 400)
        self.assertEqual(self.req("POST", "/api/shortlist", b"x" * 20000)[0], 400)

    def test_refresh_endpoints(self):
        status, _, raw = self.req("GET", "/api/refresh")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(raw)["running"])
        self.assertEqual(self.req("POST", "/api/refresh")[0], 202)
        self.assertEqual(self.req("POST", "/api/refresh")[0], 409)
        for _ in range(50):
            st = json.loads(self.req("GET", "/api/refresh")[2])
            if not st["running"]:
                break
            time.sleep(0.05)
        self.assertTrue(st["ok"])
        self.assertIn("done", st["log_tail"])

    def test_unknown_api_is_404(self):
        self.assertEqual(self.req("GET", "/api/nope")[0], 404)
        self.assertEqual(self.req("DELETE", "/api/shortlist")[0], 404)


if __name__ == "__main__":
    unittest.main()
