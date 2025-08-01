import streamlit as st
from src.pipeline import Pipeline

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Ask Me Anything - RAG Chatbot")

# Load the pipeline
chatbot = Pipeline()

# Sidebar information
st.sidebar.title("Model Info")
st.sidebar.write("Current Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
st.sidebar.write(f"Chunks Indexed: {chatbot.get_num_chunks()}")

# Clear state
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []

# Clear button
if st.button("Clear Conversation"):
    st.session_state.answer = ""
    st.session_state.sources = []
    st.session_state.user_input = ""

# Input field
user_input = st.text_input("Enter your question:", key="user_input")

# Generate and display answer
if user_input:
    with st.spinner("Generating response..."):
        full_response = ""
        sources = []

        for response_part, retrieved_chunks in chatbot.stream_response(user_input):
            full_response += response_part
            sources = retrieved_chunks

        st.session_state.answer = full_response
        st.session_state.sources = sources

# Display only the final answer
if st.session_state.answer:
    st.markdown("**Answer:**")
    st.markdown(st.session_state.answer)

# Display only the final sources
if st.session_state.sources:
    st.markdown("**Sources Used:**")
    for i, chunk in enumerate(st.session_state.sources, 1):
        st.markdown(f"**Source {i}:**")
        st.markdown(chunk.page_content)
        st.caption(f"Source: {chunk.metadata.get('source', 'Unknown')}")
