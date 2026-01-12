import streamlit as st
from backend import ask_question
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "data", "bis_logo.png")

st.set_page_config(
    page_title="SOP Chatbot",
    page_icon=IMAGE_PATH,
    layout="centered"
)

# ---- SINGLE LINE HEADER ----
col1, col2 = st.columns([1, 4])

with col1:
    st.image(IMAGE_PATH, width=180)

with col2:
    st.markdown("<h1 style='font-size:36px;'>SOP Chatbot</h1>", unsafe_allow_html=True)


# ---- DESCRIPTION COMES DOWNWARD ----
st.caption("AI-powered chatbot")
st.markdown("---")

# ---- CHATBOT UI AREA ----

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question from the SOP...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Searching document..."):
        answer = ask_question(user_input)

    if isinstance(answer, dict):
        answer = answer.get("answer", "")

    answer = answer.split("Source pages")[0].strip()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
