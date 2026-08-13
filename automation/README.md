# n8n eval regression monitor

An n8n workflow that watches the agent's answer quality over time and alerts when
it regresses — the "monitor, evaluate, and improve model performance" piece,
automated instead of something you'd run by hand.

## What it does

```
Schedule (every 6h) → GET /eval-check → pass? ──no──→ push alert (ntfy.sh)
                                          │
                                         yes → do nothing
```

`GET /eval-check` (added in `api.py`, logic in `eval/health_check.py`) runs a
small random sample of the eval set — fast enough to call on a timer, unlike the
full 54-question `eval/run_eval.py` — and checks the results against thresholds in
`.env` (`EVAL_GROUNDED_THRESHOLD`, `EVAL_FAITHFULNESS_THRESHOLD`,
`EVAL_HIT_RATE_THRESHOLD`). If retrieval, grounding, or faithfulness has dropped
below its threshold, the endpoint returns `"pass": false` and n8n sends a push
notification with the specific failure(s).

Both branches of this workflow were tested live end to end (not just imported and
left untouched): the pass path routes correctly to "All Good — No Alert," and the
failure path was triggered on purpose (by temporarily setting
`EVAL_HIT_RATE_THRESHOLD=1.5`, an impossible bar) and confirmed to deliver a real
push notification via ntfy.sh with the exact failing metric in the message body.

## Run it

1. Start the RAG API: `uvicorn api:app --reload` — reachable at `127.0.0.1:8000`
   if n8n runs on the same machine.
2. Start n8n:
   ```bash
   npx n8n
   ```
   (Docker also works if you have it: `docker run -it --rm -p 5678:5678 n8nio/n8n`)
3. Open <http://localhost:5678>, then **Import from File** →
   `automation/n8n-eval-monitor.json`.
4. Click **Execute Workflow** to test it once, then toggle **Active** to run it on
   the schedule.

## Two gotchas worth knowing about

**Use `127.0.0.1`, not `localhost`, in the HTTP Request node URLs.** Node.js
resolves `localhost` to IPv6 (`::1`) first; if your server is only bound to IPv4
`127.0.0.1` (uvicorn's default), n8n's request fails instantly with "connection
refused" even though the same URL works fine in a browser. Using the literal IP
sidesteps the DNS resolution order entirely.

**n8n blocks `$env.*` expressions in node parameters by default** (a security
setting), so referencing an environment variable for the ntfy topic or API URL
directly in a node field throws "access to env vars denied." The working version
of this workflow hardcodes both URLs directly in the node UI instead of using
`{{$env.RAG_API_BASE_URL}}`-style expressions.

**Timeout matters for local LLMs.** If you're running the RAG agent against a
local model (e.g. Ollama), CPU-bound inference can take minutes per question —
the default 120-second HTTP Request timeout isn't enough. This workflow's
"Run Eval Health Check" node is set to a 300-second timeout and uses
`sample_size=2` (not the default 8) to keep test runs reasonably fast.

## Why ntfy.sh instead of Slack/email

No account, no API key, no SMTP setup — a plain HTTP POST that shows up as a push
notification within seconds. That keeps the demo runnable by anyone in a couple of
minutes. Swapping the alert step for a Slack or email node instead is a one-node
change if you have those credentials set up. Note: ntfy.sh public topics need no
auth, so pick a topic name that isn't easily guessable if you don't want others
subscribing to your alerts.

## Test the failure path yourself

Temporarily lower a threshold so the check fails on purpose, without needing the
model to actually get worse:

```powershell
$env:EVAL_HIT_RATE_THRESHOLD="1.5"; uvicorn api:app --reload
```

Then hit **Execute Workflow** in n8n — the alert branch fires and you'll get a
push notification, confirming the whole loop works end to end. Restart uvicorn
without the override afterward to return to normal thresholds.
