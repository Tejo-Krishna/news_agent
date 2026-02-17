# streamlit_app.py
# Streamlit UI for your FastAPI backend
# - Uses POST /ask and POST /agent
# - Keeps chat history in Streamlit session_state
# - Shows /health status

import os
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("NEWS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="News Assistant (RAG)", page_icon="🗞️", layout="wide")
st.title("🗞️ News Assistant — RAG (HF Embeddings + GPT Generation)")
st.caption("Streamlit UI → FastAPI backend (/ask, /agent)")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL).strip()

    # Prevent common mistake: user pastes /ask into base url
    if api_url.endswith("/ask") or api_url.endswith("/agent"):
        st.warning("⚠️ Base URL should be like http://localhost:8000 (not /ask or /agent).")
        api_url = api_url.rsplit("/", 1)[0]

    mode = st.radio("Mode", ["RAG Answer (/ask)", "Agent (/agent)"], index=0)
    st.divider()
    st.write("Tip: Start FastAPI first, then run Streamlit.")
    st.write("FastAPI docs:", f"{api_url}/docs")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content":"..."}]

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def call_ask(question: str):
    payload = {
        "question": question,
        "chat_history": st.session_state.messages,
    }
    # ✅ MUST be POST (not GET)
    r = requests.post(f"{api_url}/ask", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def call_agent(user_input: str):
    payload = {
        "input": user_input,
        "chat_history": st.session_state.messages,
    }
    # ✅ MUST be POST (not GET)
    r = requests.post(f"{api_url}/agent", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

prompt = st.chat_input("Ask about the latest news… (e.g., 'What's new in AI?' )")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    assistant_text = ""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if mode.startswith("RAG"):
                    data = call_ask(prompt)

                    answer = data.get("answer", "").strip()
                    takeaways = data.get("takeaways", []) or []
                    sources = data.get("sources", []) or []
                    confidence = data.get("confidence", "").strip()

                    parts = []
                    parts.append(answer if answer else "I couldn't find an answer in the retrieved context.")

                    if confidence:
                        parts.append(f"\n\n**Confidence:** `{confidence}`")

                    if takeaways:
                        parts.append("\n\n**Takeaways**")
                        parts.extend([f"- {t}" for t in takeaways])

                    if sources:
                        parts.append("\n\n**Sources**")
                        for s in sources[:5]:
                            title = (s.get("title") or "Source").strip()
                            link = (s.get("link") or "").strip()
                            published = (s.get("published") or "").strip()
                            if link:
                                parts.append(f"- [{title}]({link})" + (f" ({published})" if published else ""))
                            else:
                                parts.append(f"- {title}" + (f" ({published})" if published else ""))

                    assistant_text = "\n".join(parts).strip()
                    st.markdown(assistant_text)

                else:
                    data = call_agent(prompt)
                    assistant_text = (data.get("output") or "").strip() or "(No output returned)"
                    st.markdown(assistant_text)

            except requests.exceptions.HTTPError as e:
                # Show server response body if available (very helpful)
                try:
                    detail = e.response.text
                except Exception:
                    detail = ""
                assistant_text = f"❌ API error: {e}\n\n**Details:**\n```\n{detail}\n```"
                st.error(assistant_text)

            except requests.exceptions.RequestException as e:
                assistant_text = f"❌ API error: {e}"
                st.error(assistant_text)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    st.write("Backend health check:")
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        if health.status_code == 200:
            st.success(f"✅ {health.json()}")
        else:
            st.warning(f"⚠️ /health returned {health.status_code}")
    except Exception:
        st.warning("⚠️ Backend not reachable. Start FastAPI first.")
