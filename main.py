
import streamlit as st


st.title("Sarcasm Detection")
chat_tab, analysis_tab = st.tabs(["Chat", "Analysis"])
with chat_tab:
    st.header("Chat")
with analysis_tab:
    st.header("Analysis")