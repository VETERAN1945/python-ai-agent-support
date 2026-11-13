import json
import time
import httpx
from sentence_transformers import SentenceTransformer, util

AUTO_PASS = 0.75
AUTO_FAIL = 0.25

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

run_id = str(int(time.time()))
hits = 0
total = 0
retries = 0
errors = 0
passed = 0
failed = 0
gray = 0

with open("evals/dataset.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        for attempt in range(3):
            resp = httpx.post(
                "http://127.0.0.1:8000/chat",
                json={
                    "user_id": run_id + "_" + row["id"] + "_" + str(attempt),
                    "message": row["question"],
                },
                timeout=60,
            )
            data = resp.json()

            if data["reply"] != "Произошла ошибка сервера":
                break

        total += 1
        retries += attempt
        if data["reply"] == "Произошла ошибка сервера":
            errors += 1

        hit = row["expected_doc"] == data["retrieved_doc"]
        if hit:
            hits += 1

        sim = util.cos_sim(model.encode(data["reply"]), model.encode(row["reference"])).item()

        if sim >= AUTO_PASS:
            verdict = "PASS"
            passed += 1
        elif sim <= AUTO_FAIL:
            verdict = "FAIL"
            failed += 1
        else:
            verdict = "GRAY"
            gray += 1

        print(row["id"], hit, round(sim, 2), verdict, row["question"], "->", data["reply"][:300].replace("\n", " "))

print("hit@1:", hits, "/", total)
print("лишних попыток:", retries, "| ошибок осталось:", errors)
print("PASS:", passed, "| FAIL:", failed, "| GRAY (судье):", gray)