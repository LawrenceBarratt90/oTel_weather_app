# Weather AI Demo: AI Observability with OpenTelemetry + Dynatrace

A small chat app: ask about the weather in plain English and get an answer back.
The point of the demo is how **little** code it takes to make an AI app observable.

```
You: Should I take an umbrella in Tokyo?
Bot: It's unlikely you'll need one in Tokyo today, there's no rain forecast...
```

## How the app works (`app.py`, about 60 lines of logic)

A three-step [LangGraph](https://langchain-ai.github.io/langgraph/) workflow:

```
question ─► extract_city ─► get_weather ─► answer ─► reply
              (LLM)          (API call)      (LLM)
```

1. **extract_city**: the LLM pulls the city out of your question.
2. **get_weather**: a plain HTTP call to [Open-Meteo](https://open-meteo.com) (free, no API key).
3. **answer**: the LLM turns the raw numbers into a friendly reply.

The model is a local Ollama model (`llama3.2:1b`), so no AI provider key is needed.

## The OpenTelemetry part (`telemetry.py`, about 15 lines)

```python
provider = TracerProvider(resource=Resource.create({"service.name": "weather-ai-demo"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=..., headers=...)))  # 1. where to send
LangchainInstrumentor().instrument()     # 2. LangChain + LangGraph + LLM calls
HTTPXClientInstrumentor().instrument()   # 3. outgoing HTTP calls
```

That's all. `app.py` calls `setup_telemetry()` once at startup and has **no tracing code**
in the graph nodes. The only manual line is one `weather-chat` span per question, which
groups everything into a single trace and tags it with a session ID.

## What you get in Dynatrace

One trace per question, showing:

| Span | What it tells you |
|---|---|
| `weather-chat` | The whole request, end to end |
| `invoke_agent LangGraph` / `LangGraph.workflow` | The graph run as a whole |
| `execute_task extract_city` / `get_weather` / `answer` | Time spent in each node |
| `ChatOllama.chat` | Prompt and response, input/output token counts, latency (`gen_ai.*` attributes) |
| `GET` / `POST` (HTTP) | The two Open-Meteo calls and the calls to Ollama, with status codes and timing |

Good things to show off:
- **Where did the time go?** The LLM calls usually dominate the trace.
- **What did the model actually see and say?** Open an LLM span and read the prompt and completion.
- **Token usage and cost drivers** per step.
- **A failing step**: ask about a city that doesn't exist and watch the trace.

Find it under **Distributed Tracing** (service `weather-ai-demo`), or in the **AI Observability** app.

Each reply prints its trace ID. Paste it into Distributed Tracing to open that exact run, or query `fetch spans | filter toString(trace.id) == "<id>"`.

Example DQL:
```
fetch spans
| filter service.name == "weather-ai-demo"
| summarize count(), avg(duration), by:{span.name}
```

## Run it

Needs Python 3.10+ and [Ollama](https://ollama.com) with `ollama pull llama3.2:1b`.

```bash
python3 -m venv .venv && source .venv/bin/activate     # or: uv venv
pip install -r requirements.txt                          # or: uv pip install -r requirements.txt
cp .env.example .env                                     # then fill in your Dynatrace values
python app.py        # or just: ./chat
```

`.env` needs:
- `DT_ENDPOINT`: `https://<env-id>.live.dynatrace.com/api/v2/otlp`
- `DT_API_TOKEN`: an API token with the `openTelemetryTrace.ingest` scope

Never commit `.env` (it's in `.gitignore`).

## Dedicated Ollama (optional)

If another app already uses Ollama on port 11434 and slows it down, run a separate instance: `./start-ollama.sh` starts one on port 11435 using the same model files. Set `OLLAMA_URL=http://localhost:11435` in `.env`. It does not survive a reboot, so run the script again.

## Tips

- **Model name shows as `unknown` on the LLM span** when using Ollama (the instrumentation doesn't read it from `ChatOllama`). The root span carries it as `chat.model`.
- The 1B model is fast but not clever: a few example messages in `extract_city` keep it on task. A larger model (change `OLLAMA_MODEL`) needs less prompting.
- Prompts and responses are captured in spans by default. To turn that off for sensitive data, set `TRACELOOP_TRACE_CONTENT=false`.
- To use a hosted model instead of Ollama, swap the `ChatOllama` line; the instrumentation setup stays the same (not tested here).

## Querying from the terminal (dtctl)

```bash
dtctl query 'fetch spans, from:now()-1h | filter service.name == "weather-ai-demo" | summarize count(), by:{span.name}'
```

On a headless Linux box without a keyring, set `DTCTL_TOKEN_STORAGE=file` (needed for `dtctl auth refresh` too) and run `dtctl auth login`. The access token is short-lived: run `dtctl auth refresh` when queries return "Jwt expired".
