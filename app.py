# app.py
"""
Study Wise Ai Tutor - Streamlit app (single-file)
Features:
- Modern splash screen + gradient
- Bubble-style chat UI (user: right purple | assistant: left red/white)
- File upload inside input area (pdf/docx/txt/md)
- File parsing with PyPDF2, python-docx, markdown2
- Auto-generated: summary, key concepts, quiz Qs, suggested search queries
- Reasoning modes: Explain, Quiz, Review, Deep Thinking
- Edit last question per-bubble, copy assistant reply to clipboard
- Animated loading indicator (blinking dots)
- External references toggle in sidebar (LLM decides links)
- LLM integration: google.genai preferred, openai fallback
- Modular, well-commented
"""

import os
import io
import time
import html
import textwrap
import tempfile
from typing import Tuple, List, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components

# File parsing libs
from PyPDF2 import PdfReader
import docx
import markdown2

# Optional LLM libs (may or may not exist in environment)
try:
    from google import genai
except Exception:
    genai = None

try:
    import openai
except Exception:
    openai = None

# ----------------------
# Configuration & Utils
# ----------------------

PAGE_TITLE = "Study Wise Ai Tutor"
PAGE_ICON = "✨"
MODEL_NAME_DEFAULT = os.getenv("LLM_MODEL", "gemini-2.0-flash")
MAX_FILE_TEXT = 120000  # characters allowed from file
MAX_PROMPT_CHUNK = 6000  # chunk size to send to LLM for file summarization

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ----------------------
# LLM helper functions
# ----------------------

def get_gemini_client():
    """Return a google.genai client if configured."""
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else None
    if api_key and genai is not None:
        return genai.Client(api_key=api_key)
    return None

def get_openai_client():
    """Return openai module configured if OPENAI_API_KEY exists."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
    if api_key and openai is not None:
        openai.api_key = api_key
        return openai
    return None

def generate_response(prompt: str, mode_meta: dict = None, max_output_tokens: int = 500) -> str:
    """
    Generate a response using configured LLM.
    Prefers google.genai -> openai -> fallback echo heuristic.
    mode_meta can include 'external_refs' boolean to instruct model.
    """
    # Build system/preamble for tutor style & reasoning modes
    system_preamble = (
        "You are Study Wise Ai Tutor — a helpful, patient, step-by-step AI tutor. "
        "When asked, provide clear explanations, list key steps, produce short quizzes, and suggest external resources "
        "if enabled. Keep answers concise but thorough; when asked for step-by-step, number steps. "
    )

    if mode_meta is None:
        mode_meta = {}

    external_flag = mode_meta.get("external_refs", True)
    reasoning = mode_meta.get("reasoning", "explain")

    # Append instructions to prompt
    assembled_prompt = f"{system_preamble}\nMode: {reasoning}\nExternalLinksAllowed: {external_flag}\n\n{prompt}"

    # Try Gemini first (if available)
    gemini = get_gemini_client()
    if gemini:
        try:
            resp = gemini.models.generate_content(
                model=MODEL_NAME_DEFAULT,
                contents=assembled_prompt,
                config={"max_output_tokens": max_output_tokens}
            )
            # Different SDK versions return different shapes; handle common ones:
            if hasattr(resp, "text") and resp.text:
                return resp.text
            # Fallback stringify
            return str(resp)
        except Exception as e:
            st.debug = getattr(st, "debug", lambda *a, **k: None)
            st.debug(f"Gemini error: {e}")

    # Try OpenAI ChatCompletion fallback
    openai_client = get_openai_client()
    if openai_client:
        try:
            if hasattr(openai_client, "ChatCompletion"):
                messages = [
                    {"role": "system", "content": system_preamble},
                    {"role": "user", "content": prompt}
                ]
                resp = openai_client.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            else:
                resp = openai_client.Completion.create(
                    engine="text-davinci-003",
                    prompt=assembled_prompt,
                    max_tokens=max_output_tokens,
                    temperature=0.2,
                )
                return resp.choices[0].text.strip()
        except Exception as e:
            st.debug(f"OpenAI error: {e}")

    short = assembled_prompt.strip()
    return ("[Local fallback response]\n\n"
            + (short[:800] + ("..." if len(short) > 800 else "")))

# ----------------------
# File parsing utilities
# ----------------------

def read_pdf_bytes(bytes_data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(bytes_data))
        pages = []
        for p in reader.pages:
            try:
                text = p.extract_text()
            except Exception:
                text = ""
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        return f"[Error reading PDF: {e}]"

def read_docx_bytes(bytes_data: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(bytes_data)
            tmp.flush()
            doc = docx.Document(tmp.name)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
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
        if lower.endswith(".md"):
            text = markdown2.markdown(text)
    if len(text) > MAX_FILE_TEXT:
        text = text[:MAX_FILE_TEXT] + "\n\n[Truncated]"
    return name, text

# ----------------------
# (rest of code omitted for brevity - it's identical to the long version in the previous assistant message)
# ----------------------
