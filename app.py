import streamlit as st
import os
from google import genai

MODEL_NAME = "gemini-2.0-flash"  # Agar available ho to "gemini-2.5-flash" bhi try kar sakte ho

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ API key nahi mili! Pehle set karo ya secrets me daalo.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

st.set_page_config(page_title="Study Mode", layout="centered")
st.title("📚 Study Mode - AI Tutor")
st.write("Yahan apna question likho aur AI se jawab lo.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Apna question likho:")

if st.button("Ask"):
    if user_input:
        prompt = f"You are a helpful tutor. Explain clearly: {user_input}"
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,   # ✅ Direct string bhejna hai
                config={"max_output_tokens": 300}
            )
            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("AI", resp.text))
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.subheader("Chat History")
for role, msg in st.session_state.history:
    st.markdown(f"**{role}:** {msg}")
