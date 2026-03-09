# Agentic Trust & Friction Orchestrator

A multi-agent system that evaluates cross-border money transfer risk in real time using Claude (Anthropic).

## Agents

- **Investigator** — Analyzes transaction features (amount, IP, device, corridor)
- **Context** — Retrieves user history and life-event context from vector memory
- **Risk Scorer** — Synthesizes signals into a risk score and action (Approve / Challenge / Escalate / Block)
- **Communication** — Generates personalized verification messages for flagged transactions
- **LLM-as-a-Judge** — Grades agent reasoning quality

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

## Run

```bash
uvicorn agentic_orchestrator.api:app --reload
# Open http://localhost:8000
```

## Test

```bash
python -m pytest tests/ -v
```

## Lint

```bash
ruff check agentic_orchestrator/ tests/
ruff format --check agentic_orchestrator/ tests/
```

