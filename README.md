# Agentic Trust & Friction Orchestrator

A Multi-Agent System (MAS) that evaluates cross-border money transfer risk in real time using Claude (Anthropic). It uses LangGraph to orchestrate specialized AI agents that analyze transactions, assess risk, and apply the right level of friction — from auto-approve to human review.

## Architecture

```
Transaction Request
        │
        ▼
┌─────────────────┐
│ Compliance       │──── Sanctions/AML check (hard rules, no LLM)
│ Firewall         │──── BLOCK if sanctioned country or AML threshold
└────────┬────────┘
         │ PASS
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Investigator     │────▶│ Context Agent    │
│ Agent (Claude)   │     │ (Claude)         │
│                  │     │                  │
│ Analyzes:        │     │ Retrieves:       │
│ • Amount anomaly │     │ • User history   │
│ • Device/IP      │     │ • Life events    │
│ • Corridor risk  │     │ • Past behavior  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │ Risk Scorer      │
          │ Agent (Claude)   │
          │                  │
          │ Outputs:         │
          │ • Risk score 0-1 │
          │ • Action         │
          └────────┬────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   Score < 0.3         Score 0.3-0.6
   ✅ APPROVE          ⚠️ CHALLENGE
                             │
                             ▼
                   ┌─────────────────┐
                   │ Communication    │
                   │ Agent (Claude)   │
                   │                  │
                   │ Generates        │
                   │ friendly 2FA     │
                   │ message          │
                   └─────────────────┘

         Score > 0.6
         🚨 ESCALATE → Human-in-the-Loop review
```

## Agents

| Agent | Role | Engine |
|---|---|---|
| **Investigator** | Analyzes transaction features — amount, IP, device, corridor, time patterns | Claude |
| **Context** | Retrieves user history and life-event context from ChromaDB vector memory | Claude |
| **Risk Scorer** | Synthesizes Investigator + Context signals into a risk score (0–1) and action | Claude |
| **Communication** | Generates personalized, friendly verification messages for challenged transactions | Claude |
| **LLM-as-a-Judge** | Grades the quality of agent reasoning, tool usage, and decision logic | Claude |

## Friction Actions

| Action | Risk Score | What Happens |
|---|---|---|
| ✅ **APPROVE** | < 0.3 | Transaction goes through instantly |
| ⚠️ **CHALLENGE** | 0.3 – 0.6 | Soft verification (2FA code sent to phone) |
| 🚨 **ESCALATE** | > 0.6 | Paused for human analyst review |
| 🛑 **BLOCK** | N/A | Immediately rejected (sanctioned country / AML) |

## Project Structure

```
agentic_orchestrator/
├── agents/                 # AI agents (Investigator, Context, Risk Scorer, Communication)
├── compliance/             # Firewall (sanctions/AML), PII masker, HITL reviewer
├── evaluation/             # LLM-as-a-Judge, baseline model, shadow mode comparison
├── llm/                    # Anthropic Claude client wrapper
├── memory/                 # Short-term (session) and long-term (ChromaDB) memory
├── orchestrator/           # LangGraph engine that coordinates the agent workflow
├── data/                   # Synthetic transaction data generator
├── static/                 # Web UI (single-page app)
├── api.py                  # FastAPI server with REST endpoints
├── config.py               # Thresholds, weights, and constants
└── main.py                 # CLI demo runner
tests/                      # 82 tests covering all components
```

## Security

- **PII Hashing** — Sensitive fields (IP, device ID, recipient name) are SHA-256 hashed before reaching LLM agents
- **API Key Auth** — All endpoints protected via `X-API-Key` header
- **Rate Limiting** — slowapi-based rate limiting to protect Anthropic API credits
- **CORS** — Configurable allowed origins
- **Compliance Firewall** — Hard-coded sanctions (KP, IR, SY, CU) and AML checks run before any LLM call

## Setup

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your values:
#   ANTHROPIC_API_KEY=sk-ant-...
#   API_SECRET_KEY=your-secret-key
#   PII_HASH_SALT=your-unique-salt
```

## Run

```bash
# Start the API server with web UI
uvicorn agentic_orchestrator.api:app --reload

# Open http://localhost:8000 in your browser
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — LLM status, model info |
| `POST` | `/api/v1/evaluate` | Evaluate a transaction through the full agent pipeline |
| `POST` | `/api/v1/users` | Register a user profile for context-aware analysis |
| `GET` | `/api/v1/users` | List registered user IDs |

## Test

```bash
# Run all 82 tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=agentic_orchestrator --cov-report=term-missing
```

## Lint

```bash
ruff check agentic_orchestrator/ tests/
ruff format --check agentic_orchestrator/ tests/
```

## CI/CD

GitHub Actions pipeline runs on every push and PR to `main`:

1. **Lint & Format** — ruff check + format verification
2. **Test** — pytest on Python 3.11 and 3.12
3. **Security Scan** — bandit + secrets check
4. **Build & Verify** — API startup health check

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `ANTHROPIC_MODEL` | No | Model name (default: `claude-sonnet-4-20250514`) |
| `API_SECRET_KEY` | No | API key for endpoint auth (skipped if unset) |
| `PII_HASH_SALT` | No | Salt for PII hashing (default: empty) |
| `CORS_ALLOWED_ORIGINS` | No | Comma-separated allowed origins |

