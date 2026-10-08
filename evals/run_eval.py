import json
import time
import httpx

run_id = str(int(time.time()))

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

        hit = row["expected_doc"] == data["retrieved_doc"]
        print(row["id"], hit, data["reply"][:60])