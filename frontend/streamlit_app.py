"""Streamlit frontend for the Meraki 10-K agent.

State strategy: the app holds NO data state — only the active session_id in
st.session_state. Every interaction re-fetches from FastAPI, so the UI is
refresh-safe and never goes stale.
"""
import os
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get("MERAKI_API_BASE", "http://localhost:8000")


# -- HTTP helpers --------------------------------------------------------------


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=60.0)


def _upload_client() -> httpx.Client:
    """PDF ingest can exceed a minute (parse + many embedding batches)."""
    return httpx.Client(base_url=API_BASE, timeout=httpx.Timeout(600.0))


def create_session() -> str:
    with _client() as c:
        r = c.post("/sessions")
        r.raise_for_status()
        return r.json()["session_id"]


def get_history(session_id: str) -> list[dict]:
    with _client() as c:
        r = c.get(f"/sessions/{session_id}/history")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()["messages"]


def list_documents(session_id: str) -> list[dict]:
    with _client() as c:
        r = c.get(f"/sessions/{session_id}/documents")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()["documents"]


def upload_document(session_id: str, file_name: str, file_bytes: bytes) -> dict:
    with _upload_client() as c:
        files = {"file": (file_name, file_bytes, "application/pdf")}
        r = c.post(f"/sessions/{session_id}/documents", files=files)
        r.raise_for_status()
        return r.json()


def send_message(session_id: str, content: str) -> dict:
    with _client() as c:
        r = c.post(f"/sessions/{session_id}/messages", json={"content": content})
        r.raise_for_status()
        return r.json()


def delete_session(session_id: str) -> None:
    with _client() as c:
        c.delete(f"/sessions/{session_id}")


# -- UI ------------------------------------------------------------------------


st.set_page_config(page_title="Meraki 10-K Agent", layout="wide")
st.title("Meraki 10-K Agent")
st.caption("Conversational Q&A over Indian listed-company FY25 annual reports")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "upload_counter" not in st.session_state:
    # Rotating key for st.file_uploader. Bump after each successful ingest
    # so the next render gets a fresh empty widget — prevents Streamlit from
    # re-ingesting the same file on every subsequent rerun (e.g. chat message).
    st.session_state.upload_counter = 0

# -- Sidebar: session + upload -------------------------------------------------

with st.sidebar:
    st.header("Session")

    col_a, col_b = st.columns(2)
    if col_a.button("New session", use_container_width=True):
        st.session_state.session_id = create_session()
        st.rerun()
    if col_b.button("End session", use_container_width=True, disabled=not st.session_state.session_id):
        delete_session(st.session_state.session_id)
        st.session_state.session_id = None
        st.rerun()

    if st.session_state.session_id:
        st.caption(f"ID: `{st.session_state.session_id[:8]}…`")
    else:
        st.info("Click 'New session' to start.")

    st.divider()
    st.header("Documents")

    if st.session_state.session_id:
        upload = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            accept_multiple_files=False,
            key=f"uploader_{st.session_state.upload_counter}",
        )
        if upload is not None:
            with st.spinner(f"Ingesting {upload.name}…"):
                result = upload_document(st.session_state.session_id, upload.name, upload.getvalue())
            st.success(f"Indexed {result['chunk_count']} chunks from {result['doc_name']}")
            # Rotate the uploader key so the widget renders empty next time —
            # otherwise the same file would re-trigger ingest on every rerun.
            st.session_state.upload_counter += 1
            st.rerun()

        docs = list_documents(st.session_state.session_id)
        if docs:
            for d in docs:
                st.markdown(f"- **{d['doc_name']}** ({d['chunk_count']} chunks)")
        else:
            st.caption("No documents uploaded.")
    else:
        st.caption("Create a session first.")


# -- Main: chat ----------------------------------------------------------------


def _render_tool_payload(label: str, value: Any) -> None:
    """Render a tool-call args/result. Falls back to text for plain strings
    so Streamlit's JSON parser doesn't choke on the synthetic 'final_result'
    tool which returns 'Final result processed.' as plain text."""
    if value is None:
        return
    st.caption(label)
    if isinstance(value, (dict, list)):
        st.json(value)
    elif isinstance(value, str):
        # Try parsing as JSON; if it isn't, render as code/text.
        import json
        try:
            st.json(json.loads(value))
        except (json.JSONDecodeError, ValueError):
            st.code(value, language=None)
    else:
        st.code(repr(value), language="python")


def _render_message(msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"]):
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            with st.expander(f"Tool calls ({len(tool_calls)})"):
                for i, tc in enumerate(tool_calls, start=1):
                    st.markdown(f"**{i}. `{tc.get('tool_name', '?')}`**")
                    _render_tool_payload("args", tc.get("args"))
                    _render_tool_payload("result", tc.get("result"))
        st.markdown(msg["content"])


if st.session_state.session_id:
    history = get_history(st.session_state.session_id)
    for msg in history:
        _render_message(msg)

    user_input = st.chat_input("Ask about the uploaded documents…")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Thinking…"):
            response = send_message(st.session_state.session_id, user_input)
        with st.chat_message("assistant"):
            tool_calls = response.get("tool_calls") or []
            if tool_calls:
                with st.expander(f"Tool calls ({len(tool_calls)})"):
                    for i, tc in enumerate(tool_calls, start=1):
                        st.markdown(f"**{i}. `{tc.get('tool_name', '?')}`**")
                        _render_tool_payload("args", tc.get("args"))
                        _render_tool_payload("result", tc.get("result"))
            st.markdown(response["answer_text"])
            citations = response.get("citations", [])
            if citations:
                with st.expander(f"Citations ({len(citations)})"):
                    for c in citations:
                        if c.get("kind") == "doc":
                            st.markdown(f"- `{c['doc_name']}` — page {c['page_number']}")
                        elif c.get("kind") == "web":
                            st.markdown(f"- [{c.get('title', c['url'])}]({c['url']})")
            usage = response.get("usage", {})
            if usage:
                st.caption(
                    f"tokens: {usage.get('total_tokens', '?')} | "
                    f"retrieval iterations: {response.get('retrieval_iterations', 0)} | "
                    f"prompt: {response.get('prompt_version', '?')}"
                )
        st.rerun()
else:
    st.info("Create a session in the sidebar to start chatting.")
