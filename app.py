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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for user question
user_input = st.text_input("Enter your question:")

# Button to clear conversation
if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

# Generate and display answer
if user_input:
    with st.spinner("Generating response..."):
        full_response = ""
        sources = []

        for response_part, retrieved_chunks in chatbot.stream_response(user_input):
            full_response += response_part
            sources = retrieved_chunks

        st.session_state.chat_history.append({
            "question": user_input,
            "answer": full_response,
            "sources": sources
        })

# Display past chat history
for idx, entry in enumerate(reversed(st.session_state.chat_history), 1):
    st.markdown(f"### Question {len(st.session_state.chat_history) - idx + 1}")
    st.markdown(entry["question"])
    
    st.markdown("**Answer:**")
    st.write(entry["answer"])

    if entry["sources"]:
        st.markdown("**Sources Used:**")
        for i, chunk in enumerate(entry["sources"], 1):
            st.markdown(f"Source {i}:")
            st.code(chunk.page_content)
            st.caption(f"Source: {chunk.metadata.get('source', 'Unknown')}")

# Footer
st.markdown("---")
st.caption("RAG Chatbot powered by custom pipeline and Streamlit")
