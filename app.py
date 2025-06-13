# app.py

import streamlit as st
from agent import PokemonReasoningAgent

# --- Page Configuration ---
st.set_page_config(
    page_title="Pokémon AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Pokémon AI Agent (RAG-Enabled)")
st.write(
    "Ask me anything about Pokémon! I can use the PokéAPI for live data and consult my own knowledge base for complex topics.")

# --- Agent and Chat History Initialization ---
if 'agent' not in st.session_state:
    with st.spinner("Initializing AI Agent... This may take a moment as it loads the knowledge base."):
        st.session_state.agent = PokemonReasoningAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Pokémon..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 The agent is thinking..."):
            response_data = st.session_state.agent.process_query(prompt)
            response = response_data.get('response', "Sorry, I encountered an error.")
            st.markdown(response)

    # Add agent response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})