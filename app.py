"""Weather chat: ask a question in plain English, get an answer. LangGraph + Ollama + Open-Meteo."""
import os
import uuid
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

# .env holds the Dynatrace connection (DT_ENDPOINT, DT_API_TOKEN) and the model settings.
# It must be loaded BEFORE telemetry is set up, because setup reads those values.
load_dotenv()

from telemetry import setup_telemetry  # noqa: E402  (after load_dotenv so env vars are set)

# DYNATRACE: this one call connects the app to Dynatrace and switches on auto-instrumentation.
# See telemetry.py for exactly where the data goes and how it is authenticated.
provider = setup_telemetry()

from opentelemetry import trace  # noqa: E402

tracer = trace.get_tracer("weather-ai-demo")
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
                 base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"), temperature=0)


class State(TypedDict, total=False):
    question: str
    city: str
    weather: str
    answer: str


def extract_city(state: State) -> State:
    """LLM step 1: pull the city out of the question."""
    reply = llm.invoke([
        ("system", "You extract the city name from a message. Output only the city name, never anything else."),
        ("user", "Is it raining in London today?"), ("assistant", "London"),
        ("user", "how hot is it in new york"), ("assistant", "New York"),
        ("user", state["question"]),
    ])
    return {"city": reply.content.strip().strip(".")}


def get_weather(state: State) -> State:
    """Tool step: look up live weather (Open-Meteo, free, no API key)."""
    geo = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": state["city"], "count": 1}, timeout=10).json()
    if not geo.get("results"):
        return {"weather": f"No location found for '{state['city']}'."}
    place = geo["results"][0]
    now = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,wind_speed_10m,precipitation"}).json()["current"]
    return {"weather": f"{place['name']}, {place.get('country', '')}: {now['temperature_2m']}°C, "
                       f"wind {now['wind_speed_10m']} km/h, precipitation {now['precipitation']} mm"}


def answer(state: State) -> State:
    """LLM step 2: turn the raw data into a friendly reply."""
    reply = llm.invoke(f"Question: {state['question']}\nWeather data: {state['weather']}\n"
                       "Answer the question in one or two friendly sentences using only this data.")
    return {"answer": reply.content.strip()}


graph = StateGraph(State)
graph.add_node("extract_city", extract_city)
graph.add_node("get_weather", get_weather)
graph.add_node("answer", answer)
graph.add_edge(START, "extract_city")
graph.add_edge("extract_city", "get_weather")
graph.add_edge("get_weather", "answer")
graph.add_edge("answer", END)
app = graph.compile()


def main() -> None:
    session_id = str(uuid.uuid4())[:8]
    print("Weather assistant (type 'quit' or press Ctrl-D to exit). Try: What's the weather like in Paris?")
    try:
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"quit", "exit"}:
                break
            if not question:
                continue
            # DYNATRACE (the only hand-written tracing in the app): one root span per question.
            # Everything the auto-instrumentation creates inside this block (graph nodes, LLM calls,
            # weather HTTP calls) becomes a child of it, so each question is ONE trace in Dynatrace.
            # The attributes are searchable: e.g. filter chat.session_id == "<id>" to see one conversation.
            with tracer.start_as_current_span("weather-chat") as span:
                span.set_attribute("chat.session_id", session_id)
                span.set_attribute("chat.model", llm.model)  # the LLM span itself shows model "unknown" for Ollama
                result = app.invoke({"question": question})
                trace_id = format(span.get_span_context().trace_id, "032x")
            print(f"Bot: {result['answer']}")
            # DYNATRACE: this ID identifies the trace we just sent. Paste it into Distributed Tracing,
            # or query it:  fetch spans | filter toString(trace.id) == "<id>"   (spans take a few seconds to appear)
            print(f"     (trace {trace_id})")
    except (EOFError, KeyboardInterrupt):
        print()
    # DYNATRACE: spans are sent in batches in the background; flush so the final question's
    # spans reach Dynatrace before the process exits.
    provider.force_flush()


if __name__ == "__main__":
    main()
