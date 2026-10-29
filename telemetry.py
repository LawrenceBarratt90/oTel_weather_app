"""
OpenTelemetry -> Dynatrace setup. This file is ALL the observability code the app needs.

HOW THIS APP IS CONNECTED TO DYNATRACE
--------------------------------------
Data flow:   app (this process) --OTLP over HTTPS--> Dynatrace tenant --> Distributed Tracing / AI Observability

  * Protocol:  OTLP/HTTP (protobuf) - the standard OpenTelemetry wire format. No Dynatrace agent or
               OneAgent is involved; the app sends its own spans straight to Dynatrace.
  * Endpoint:  DT_ENDPOINT in .env, e.g. https://<env-id>.sprint.dynatracelabs.com/api/v2/otlp
               We append /v1/traces because we only send traces (no metrics or logs).
  * Auth:      DT_API_TOKEN in .env, sent as the header  "Authorization: Api-Token <token>".
               The token needs the "openTelemetryTrace.ingest" scope (ingest only; it cannot read data).
  * Identity:  every span is tagged service.name=weather-ai-demo (OTEL_SERVICE_NAME in .env).
               That is the name you filter on in Dynatrace:
                   fetch spans | filter service.name == "weather-ai-demo"
  * Batching:  spans are queued and sent in the background, so chatting is never slowed by the
               network. app.py calls provider.force_flush() on exit so the last spans are not lost.

Nothing here is Dynatrace-specific code; swap the endpoint and header and the same setup works with
any OTLP backend. The values come from .env (loaded by app.py before this module is called).
"""
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_telemetry() -> TracerProvider:
    # STEP 1 - WHO WE ARE: a "resource" describes this app. Dynatrace shows it as the service name.
    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "weather-ai-demo")})

    # STEP 2 - WHERE TO SEND: the exporter is the connection to Dynatrace (endpoint + API token).
    #   os.environ[...] (not .get) on purpose: the app fails fast with a clear KeyError if .env is missing.
    exporter = OTLPSpanExporter(
        endpoint=f"{os.environ['DT_ENDPOINT']}/v1/traces",
        headers={"Authorization": f"Api-Token {os.environ['DT_API_TOKEN']}"},
    )

    # STEP 3 - HOW TO SEND: the provider creates spans; the batch processor ships them to the exporter.
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)  # makes this the global tracer, so app.py's get_tracer() uses it

    # STEP 4 - WHAT TO CAPTURE: auto-instrumentation. These two calls are the whole "AI observability" setup;
    #   the graph nodes in app.py contain no tracing code.
    #   LangchainInstrumentor  -> spans for the LangGraph run and each node (execute_task <node>), plus every
    #                             LLM call (ChatOllama.chat) with prompt, response and token counts
    #                             (gen_ai.* attributes). These are what Dynatrace's AI Observability views read.
    #                             Set TRACELOOP_TRACE_CONTENT=false to stop prompts/responses being captured.
    #   HTTPXClientInstrumentor -> spans for outgoing HTTP calls: Open-Meteo (weather) and Ollama (the model).
    #   Known gap: for Ollama the model name on the LLM span shows as "unknown" (see app.py, chat.model).
    LangchainInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    return provider
