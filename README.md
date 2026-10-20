# ollama-python-agent

A small Python agent that calls tools through your local [Ollama](https://ollama.com) server.

It uses the official `ollama` client and `llama3.2:3b` (already pulled on this machine). Tools:

- `current_time` — local date and time
- `calculate` — arithmetic (`+ - * / // % **`)

## Run

```bash
cd ~/ollama-python-agent
source .venv/bin/activate
python agent.py "What time is it, and what is 17 * 23?"
```

With no arguments it opens a prompt. Type `quit` to leave.

```bash
python agent.py
```

## Overrides

| Variable | Default |
| --- | --- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `llama3.2:3b` |
