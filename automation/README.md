# n8n eval regression monitor

An n8n workflow that watches the agent's answer quality over time and alerts
when it regresses — the "monitor, evaluate, and improve model performance"
piece, automated instead of something you'd run by hand.

## What it does

```
Schedule (every 6h) → GET /eval-check → pass? ──no──→ push alert (ntfy.sh)
                                          │
                                         yes → do nothing
```

`GET /eval-check` (added in `api.py`, logic in `eval/health_check.py`) runs a
small random sample of the eval set — fast enough to call on a timer, unlike
the full 54-question `eval/run_eval.py` — and checks the results against
thresholds in `.env` (`EVAL_GROUNDED_THRESHOLD`, `EVAL_FAITHFULNESS_THRESHOLD`,
`EVAL_HIT_RATE_THRESHOLD`). If retrieval, grounding, or faithfulness has
dropped below its threshold, the endpoint returns `"pass": false` and n8n
sends a push notification with the specific failure(s).

## Run it

1. Start the RAG API: `uvicorn api:app --reload` (must be reachable from
   wherever n8n runs — `localhost:8000` if n8n is on the same machine).
2. Start n8n:
   ```bash
   docker run -it --rm -p 5678:5678 n8nio/n8n
   ```
3. Open <http://localhost:5678>, then **Import from File** →
   `automation/n8n-eval-monitor.json`.
4. (Optional) In the workflow's environment/variables, set `NTFY_TOPIC` to
   something private — anyone who knows the topic name can read alerts sent
   to it, since [ntfy.sh](https://ntfy.sh) public topics need no auth. Install
   the ntfy app or open `https://ntfy.sh/<your-topic>` in a browser tab to
   receive the push notification.
5. Click **Execute Workflow** to test it once, then toggle **Active** to run
   it on the schedule.

## Why ntfy.sh instead of Slack/email

No account, no API key, no SMTP setup — a plain HTTP POST that shows up as a
push notification within seconds. That keeps the demo runnable by anyone in a
couple of minutes. Swapping the alert step for a Slack or email node instead
is a one-node change if you have those credentials set up.

## Test the failure path

Temporarily lower a threshold so the check fails on purpose, without needing
the model to actually get worse:

```bash
EVAL_HIT_RATE_THRESHOLD=1.5 uvicorn api:app --reload
```

Then hit **Execute Workflow** in n8n — the alert branch fires and you'll get
a push notification, confirming the whole loop works end to end.
