# app.py
"""
Study Wise Ai Tutor - Streamlit app
Features:
- Modern, bubble-style chat UI (purple theme)
- File upload in message input (pdf/docx/txt/md)
- Automatic parsing & analysis of uploaded files (PyPDF2, python-docx)
- Step-by-step reasoning mode for tutor responses
- "External references" cards suggested by the model
- Lightweight, modular, commented code

Requirements:
- streamlit
- PyPDF2
- python-docx
- markdown2
- google.genai (or adapt get_client() to your LLM provider)
"""

import os
import io
import time
import textwrap
import tempfile
from typing import Tuple, List, Dict, Optional

import streamlit as st
from PyPDF2 import PdfReader
import docx
import markdown2

try:
    from google import genai
except Exception:
    genai = None

MODEL_NAME_DEFAULT = "gemini-2.0-flash"

def get_client():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found.")
        st.stop()
    if genai is None:
        st.error("❌ google.genai package not found.")
        st.stop()
    return genai.Client(api_key=api_key)

def read_pdf_bytes(bytes_data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(bytes_data))
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        return f"[Error reading PDF: {e}]"

def read_docx_bytes(bytes_data: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(bytes_data)
            tmp.flush()
            doc = docx.Document(tmp.name)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"[Error reading DOCX: {e}]"

def read_text_bytes(bytes_data: bytes, encoding="utf-8") -> str:
    try:
        return bytes_data.decode(encoding, errors="replace")
    except Exception:
        return bytes_data.decode("latin-1", errors="replace")

def parse_uploaded_file(uploaded_file) -> Tuple[str, str]:
    name = uploaded_file.name
    raw = uploaded_file.read()
    lower = name.lower()
    if lower.endswith(".pdf"):
        text = read_pdf_bytes(raw)
    elif lower.endswith(".docx"):
        text = read_docx_bytes(raw)
    else:
        text = read_text_bytes(raw)
    return name, text[:120000]

def generate_response(prompt: str, model_name: str = MODEL_NAME_DEFAULT, max_output_tokens: int = 400) -> str:
    client = get_client()
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"max_output_tokens": max_output_tokens}
        )
        if hasattr(resp, "text") and resp.text:
            return resp.text
        return str(resp)
    except Exception as e:
        return f"[LLM Error] {e}"

def init_session_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "files" not in st.session_state:
        st.session_state.files = []

def push_chat(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content, "ts": time.time()})

def render_chat():
    for msg in st.session_state.chat:
        role = msg["role"]
        if role == "user":
            st.markdown(f"<div style='text-align:right;color:white;background:#6a49ff;padding:10px;border-radius:12px;margin:5px'>{msg['content']}</div>", unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f"<div style='text-align:left;color:black;background:#f7f7fb;padding:10px;border-radius:12px;margin:5px'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.info(msg["content"])

def main():
    st.set_page_config(page_title="Study Wise Ai Tutor", page_icon="📘", layout="wide")
    init_session_state()

    st.title("📘 Study Wise Ai Tutor")
    st.caption("Your personal AI-powered study companion. Study Wise Ai Tutor helps you learn faster with smart explanations, file analysis, external references, and interactive reasoning.")

    render_chat()

    with st.form("input_form", clear_on_submit=False):
        user_text = st.text_input("Ask a question...")
        uploaded = st.file_uploader("Upload file", type=["pdf", "docx", "txt", "md"])
        send_btn = st.form_submit_button("Send")

        if send_btn:
            if uploaded:
                filename, txt = parse_uploaded_file(uploaded)
                st.session_state.files.append({"name": filename, "text": txt})
                push_chat("user", f"Uploaded {filename}")
                resp = generate_response(f"Summarize this study document:\n{txt[:6000]}")
                push_chat("assistant", resp)
            if user_text:
                push_chat("user", user_text)
                prompt = f"You are Study Wise Ai Tutor. Provide a step-by-step explanation.\nUser: {user_text}"
                resp = generate_response(prompt)
                push_chat("assistant", resp)

if __name__ == "__main__":
    main()
