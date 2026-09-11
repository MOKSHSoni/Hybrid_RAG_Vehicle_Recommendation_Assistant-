"""Phase 13: Streamlit chat UI.

Run with: streamlit run streamlit_app.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.app.orchestrator import Pipeline, answer_query
from src.query.models import ConversationTurn
from src.query.ollama_client import is_reachable
from src.rag.comparison import build_comparison_rows
from src.rag.generation import generate_answer_stream, needs_regeneration, regenerate_long_form

st.set_page_config(page_title="Car Sales Assistant", page_icon="🚗", layout="wide")

MODE_LABELS = {
    "exact": ("Exact match", "green"),
    "relaxed": ("Relaxed match -- some requirements were loosened", "orange"),
    "fallback": ("Semantic fallback -- no requirements verified", "red"),
    "superlative": ("Ranked by exact metadata sort, not relevance", "blue"),
}


@st.cache_resource(show_spinner="Loading knowledge base and models (first run only, ~30s)...")
def get_pipeline() -> Pipeline:
    return Pipeline.build()


@st.cache_data(ttl=30, show_spinner=False)
def cached_ollama_reachable() -> bool:
    return is_reachable()


def render_mode_badge(mode: str) -> None:
    label, color = MODE_LABELS.get(mode, (mode, "gray"))
    st.markdown(f":{color}[**{label.upper() if mode == 'exact' else label}**]")


def _fmt(value, unit: str = "") -> str:
    if value is None or value == "":
        return "N/A"
    return f"{value}{unit}"


def render_vehicle_card(merged) -> None:
    meta = merged.best_chunk.metadata
    with st.container(border=True):
        st.markdown(f"**{meta.get('name', 'Unknown vehicle')}**")
        cols = st.columns(4)
        cols[0].metric("Price", f"Rs {_fmt(meta.get('price_lakhs'))} L")
        cols[1].metric("Seats", _fmt(meta.get("seating_capacity")))
        cols[2].metric("Body type", _fmt(meta.get("body_type")))
        cols[3].metric("Fuel", ", ".join(meta.get("fuel_types") or []) or "N/A")


def render_debug_panel(result) -> None:
    log = result.log
    with st.expander("Debug panel"):
        st.markdown("**Query understanding**")
        st.json(
            {
                "extraction_method": log.extraction_method,
                "standalone_query": log.standalone_query,
                "transformed_query": log.transformed_query,
                "constraints": log.constraints_summary,
            }
        )
        st.markdown("**Retrieval**")
        st.json(
            {
                "mode": log.mode,
                "retrieval_method": log.retrieval_method,
                "relaxation_steps": log.relaxation_steps,
                "candidate_count": log.candidate_count,
                "final_chunk_ids": log.final_chunk_ids,
            }
        )
        st.markdown("**Per-stage timings (ms)**")
        st.table([{"stage": t.stage, "duration_ms": round(t.duration_ms, 1)} for t in log.stage_timings])
        st.caption(f"Total: {log.total_ms():.0f} ms")

        if result.expansion_queries or result.hyde_description:
            st.markdown("**Experimental (informational only -- NOT used in the answer above)**")
            if result.expansion_queries:
                st.write("Expanded queries (Phase 4):")
                for q in result.expansion_queries:
                    st.write(f"- {q}")
            if result.hyde_description:
                st.write("HyDE hypothetical description (Phase 5) -- synthetic, never a real vehicle:")
                st.info(result.hyde_description)


def render_comparison_table(result) -> None:
    if len(result.outcome.results) < 2:
        return
    rows = build_comparison_rows(result.outcome)
    if not rows:
        return
    with st.expander("Compare"):
        st.dataframe(rows, hide_index=True, use_container_width=True)


def render_result(result) -> None:
    render_mode_badge(result.log.mode)
    if result.outcome.results:
        for merged in result.outcome.results:
            render_vehicle_card(merged)
        render_comparison_table(result)
    if st.session_state.get("show_debug"):
        render_debug_panel(result)


def main() -> None:
    st.title("Car Sales Assistant")
    st.caption("Ask about vehicles in natural language -- I'll search a ~150-car dataset and explain my recommendations.")

    if "history" not in st.session_state:
        st.session_state.history = []  # List[ConversationTurn], fed to context resolution
    if "messages" not in st.session_state:
        st.session_state.messages = []  # chat display log: [{"role", "content", "result"}]

    with st.sidebar:
        st.header("Options")
        st.session_state["show_debug"] = st.checkbox("Show debug panel", value=False)
        show_experimental = st.checkbox(
            "Also run query expansion + HyDE (debug-only)",
            value=False,
            help="Runs Phase 4 expansion and Phase 5 HyDE informationally, shown in the debug panel. "
            "Does not change the recommendation itself -- both are experimental per the project spec.",
        )
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.session_state.messages = []
            st.rerun()

    if not cached_ollama_reachable():
        st.error(
            "The local Ollama backend doesn't seem to be reachable right now. Constraint extraction "
            "will fall back to regex-only matching and written recommendations will fall back to a "
            "plain vehicle listing until it's back up -- nothing will crash, but responses will be "
            "less capable. Start it with `ollama serve` (and make sure `qwen3:4b` is pulled), then "
            "refresh this page."
        )

    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Failed to build the knowledge base: {e}")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("result") is not None:
                render_result(msg["result"])

    user_input = st.chat_input('e.g. "affordable 7 seater SUV with good mileage"')
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "result": None})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Retrieval first (fast), then stream generation -- generation is
            # by far the slowest stage, so streaming it drops perceived
            # latency from "minutes of spinner" to "prose within seconds".
            with st.spinner("Understanding your request and searching..."):
                try:
                    result = answer_query(
                        pipeline,
                        user_input,
                        st.session_state.history,
                        include_expansion_hyde_debug=show_experimental,
                        generate=False,
                    )
                except Exception as e:
                    st.error(f"Something went wrong while answering: {e}")
                    st.stop()

            answer_slot = st.empty()
            started = time.time()
            try:
                with answer_slot.container():
                    answer = st.write_stream(
                        generate_answer_stream(user_input, result.outcome, pipeline.kb.chunk_store)
                    )
            except Exception as e:
                st.error(f"Something went wrong while writing the answer: {e}")
                st.stop()

            # The quality checks need the whole text, so they run post-stream;
            # a failed check replaces what was streamed via the slower path.
            if needs_regeneration(answer, result.outcome):
                with st.spinner("Improving that answer..."):
                    better = regenerate_long_form(user_input, result.outcome, pipeline.kb.chunk_store)
                # None means the retry failed -- keep the answer already on
                # screen rather than replacing real content with an error.
                if better:
                    answer = better
                    answer_slot.markdown(answer)

            result.answer = answer
            result.log.add_timing("generation (streamed)", (time.time() - started) * 1000)
            render_result(result)

        st.session_state.messages.append({"role": "assistant", "content": result.answer, "result": result})
        st.session_state.history.append(ConversationTurn(role="user", content=user_input))
        st.session_state.history.append(ConversationTurn(role="assistant", content=result.answer))


if __name__ == "__main__":
    main()
