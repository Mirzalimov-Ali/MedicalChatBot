"""Test the API from start to finish: create user -> create key -> /v1 requests -> revoke key.

Usage:
    python hybrid/1_backend/test_api.py http://127.0.0.1:7860
    python hybrid/1_backend/test_api.py https://xxx.trycloudflare.com  (over the internet)

This script performs the actions a user would normally do in the UI (sign up and create an API key)
through HTTP, then sends requests using the key as a client application would.
Standard library only; also tests with the `openai` package if it is installed.
"""

import json
import secrets
import sys
import time
import urllib.error
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860").rstrip("/")
SIGNUP_CODE = sys.argv[2] if len(sys.argv) > 2 else ""


def call(method, path, body=None, token=None):
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if "json" in r.headers.get("Content-Type", "") else raw), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}"), e.headers


def check(name, ok, extra=""):
    print(("  OK   " if ok else "  FAIL ") + name, extra)
    if not ok:
        sys.exit(1)


print("Server:", URL)
email = f"test-{secrets.token_hex(3)}@gmail.com"
s, r, _ = call("POST", "/auth/signup", {"email": email, "password": "password12345", "code": SIGNUP_CODE})
check("1. User registration", s == 200, email)
session = r["token"]
check("   registering the same email again -> 409", call("POST", "/auth/signup", {"email": email, "password": "password12345", "code": SIGNUP_CODE})[0] == 409)
check("   incorrect password -> 401", call("POST", "/auth/login", {"email": email, "password": "wrong"})[0] == 401)

s, r, _ = call("POST", "/keys", {"name": "test"}, session)
check("2. Create API key", s == 200 and r["key"].startswith("sk-th-"), r.get("preview", ""))
key = r["key"]
keys = call("GET", "/keys", token=session)[1]
check("   key list shows only a shortened version (not the full key)", key not in json.dumps(keys) and len(keys) == 1)

check("3. /v1/models without an API key -> 401", call("GET", "/v1/models")[0] == 401)
check("   invalid API key -> 401", call("GET", "/v1/models", token="sk-th-fake")[0] == 401)
s, r, _ = call("GET", "/v1/models", token=key)
check("   /v1/models with a valid API key", s == 200, [m["id"] for m in r["data"]])

model = r["data"][0]["id"]
# model = "smollm2-1.7b-instruct"

t = time.time()
s, r, h = call("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "What can cause neck pain after sitting at a computer for a long time?"}]}, key)
check("4. /v1/chat/completions", s == 200 and r["choices"][0]["message"]["content"].strip() != "",
      f"({time.time() - t:.1f} s, {r.get('usage')}, remaining requests: {h.get('x-ratelimit-remaining-requests')})")
print("       response:", r["choices"][0]["message"]["content"][:150].replace("\n", " "))

t = time.time()
s, raw, _ = call("POST", "/v1/chat/completions", {"model": model, "stream": True, "stream_options": {"include_usage": True},
                                                  "messages": [{"role": "user", "content": "What are some safe ways to improve posture while working at a desk?"}]}, key)
events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
check("5. Streaming response (SSE)", s == 200 and raw.rstrip().endswith("data: [DONE]") and text.strip() != "",
      f"({time.time() - t:.1f} s, {len(events)} chunks, usage: {events[-1].get('usage')})")
print("       response:", text[:150].replace("\n", " "))

try:
    from openai import OpenAI
    client = OpenAI(base_url=URL + "/v1", api_key=key)
    out = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "How can I maintain healthy hydration throughout the day?"}])
    check("6. Request using the OpenAI library", out.choices[0].message.content.strip() != "", out.usage)
except ImportError:
    print("  --   6. openai package is not installed, skipped")

usage = call("GET", "/usage", token=session)[1]
check("7. Usage statistics recorded", usage and usage[0]["requests"] >= 2, usage)

call("DELETE", f"/keys/{keys[0]['id']}", token=session)
check("8. Revoked API key -> 401", call("GET", "/v1/models", token=key)[0] == 401)
print("All tests passed.")
