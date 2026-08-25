import streamlit as st
from src.predict import summarize_text


st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🤖"
)

st.title("🤖 AI Text Summarization Assistant")
st.write("Enter a piece of text and let AI generate a concise summary.")

text = st.text_area(
    "Enter your text:",
    height=250,
    placeholder="Paste your text here..."
)

if st.button("Generate Summary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating AI summary..."):
            summary = summarize_text(text)

        st.subheader("AI Generated Summary")
        st.success(summary)