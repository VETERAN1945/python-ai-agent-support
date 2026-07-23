**English** | [Русский](README.md)

# AI Agent in Plain Python

An affiliate-program support bot: LLM + memory + tool calling + RAG, written
without agent frameworks (no LangChain, no LlamaIndex).

This is a learning project. I had already built the same agent in n8n, where
everything hides behind nodes. Here every layer is written by hand, so I can see
exactly where the model ends and the code begins.

## What's inside

- **LLM calls** — Groq, `llama-3.3-70b-versatile`
- **Memory** — conversation histories in a `sessions` dict, keyed by `user_id`
- **Tool calling** — two tools: `multiply` and `search_docs`
- **RAG** — `paraphrase-multilingual-MiniLM-L12-v2` embeddings, nearest document
  by Euclidean distance (`numpy`), no vector database
- **HTTP layer** — FastAPI + uvicorn, `POST /chat`
- **Failure handling** — `try/except` around both model calls, plus a rollback of
  the conversation history on error

## Files

| File | What it is |
|---|---|
| `api.py` | Final version — FastAPI server, all layers together |
| `first_call.py` | Console version, earlier stage: the same agent in a `while True` loop |
| `test_call.py` | Scratch file: isolated API calls used to check pieces before assembling them |

Earlier versions are kept on purpose: they show how the project was built layer
by layer, and what changed when it moved from console to web.

## Running it

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
```

Get a key at console.groq.com/keys.

```bash
python -m uvicorn api:app --reload
```

The first start is slow — the embedding model (~500 MB) is downloaded once.
Then open http://127.0.0.1:8000/docs

Request body:

```json
{"user_id": "u1", "message": "Сколько вы платите партнёрам?"}
```

## History rollback on failure

Groq regularly produces a malformed tool call (`400 tool_use_failed`). If you
just catch the error and return "server error" to the user, the history is left
holding a tool-call request with no matching tool response — and this user's
*next* request then fails on the corrupted history.

So the length of the history is recorded before the work starts, and everything
appended past that mark is deleted inside `except`. It's a transaction rollback:
`snapshot` is the savepoint, `del history[snapshot:]` is the rollback itself.
`try/except` saves the current request; the rollback saves every future request
from that user.

## Two things found the hard way

**1. Multilingual embeddings aren't a detail — they're a precondition.**
With `all-MiniLM-L6-v2` (`all` = all *English* datasets), Russian documents
collapse into nearly identical vectors: the gap between first and second place
was 0.005, i.e. the winner was picked by noise. Nothing errors out — a response
arrives, and it's wrong. After switching to a multilingual model the gaps became
0.63–1.24.

**2. Search by the user's question, not by the model's phrasing.**
Originally `search_docs` received the argument the model invented — often a
single word. One word produces a vector with almost no direction, equally close
to every document. Feeding it the user's original question instead made the same
query retrieve a different — and correct — document.

The general lesson: anything that has to be exact gets moved out of the model and
into code. What's left for the model is understanding meaning and phrasing text,
where exactness isn't required.

## What's deliberately missing

This is a learning project, not production code. Known gaps:

- **No `system` prompt.** Found on the final run: retrieval returned the correct
  document ("multi-accounts are banned") and the model answered the opposite,
  falling back on its own weights. Nothing in the code obliges it to answer from
  the retrieved document. That's the next step.
- **No distance threshold in search** — the nearest document is always returned,
  even when every document is far away. It needs a "nothing found" path.
- **No retry on `tool_use_failed`** — a frequent error that a repeat request fixes.
- **Sessions live in process memory** — a restart wipes every conversation.
- Documents are hardcoded, there's no authentication, and history grows without
  limit until it hits the token ceiling.
<img width="1867" height="876" alt="image" src="https://github.com/user-attachments/assets/24af8766-bc77-4e06-862f-226cdbe48a26" />
