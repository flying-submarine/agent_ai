import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method)
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode()
        req.add_header("Content-Type", "application/json")
        req.data = data
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main():
    with open("eval-set-starter.json", encoding="utf-8") as f:
        suite = json.load(f)

    passed = 0
    for item in suite["items"]:
        item_id = item["id"]
        try:
            if item["endpoint"].startswith("GET /stats"):
                got = call("GET", "/stats")
                ok = got.get("total_chars") == 495 and got.get("num_chunks") == 4
                detail = str(got)
            elif item["endpoint"].startswith("GET /search"):
                got = call("GET", "/search?q=RAG")
                ok = got.get("total", 0) >= 1
                detail = f"total={got.get('total')}"
            elif item["endpoint"] == "POST /ask_tool":
                got = call("POST", "/ask_tool", {"question": item["input"]})
                answer = got.get("answer", "")
                ok = all(s in answer for s in item["gt_contains"])
                detail = f"rounds={got.get('rounds')} answer={answer[:120]}"
            elif item["endpoint"] == "POST /ask":
                got = call("POST", "/ask", {"q": item["input"]})
                answer = got.get("answer", "")
                ok = all(s in answer for s in item["gt_contains"])
                detail = f"answer={answer[:120]} sources={got.get('sources')}"
            else:
                ok, detail = False, "未知 endpoint"
        except Exception as e:  # 服务没起、key 没 export、网络超时都落这儿
            ok, detail = False, f"调用失败：{type(e).__name__} {e}"
        print(("PASS " if ok else "FAIL ") + item_id + " | " + detail)
        passed += ok

    print(f"\n{passed}/{len(suite['items'])} 通过。FAIL 先看是不是服务没起："
          "cd ~/Desktop/day1 → source .venv/bin/activate → export 两个 key → uvicorn app:app --reload")


if __name__ == "__main__":
    main()

