"""Streamlit chatbot interface for Charles Dickens QA system."""

import os
import uuid
import requests

import streamlit as st
from dotenv import load_dotenv


STREAMLIT_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-title {
        font-weight: bold;
        color: #1f77b4;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
"""
STREAMLIT_INSTRUCTIONS_EXAMPLES = """
        ### 📝 Instructions
        1. Ensure Qdrant, Redis, and Opik services are running (`make docker-up`)
        2. Ensure FastAPI backend is running (`docker ps -a | grep api`)
        3. Documents are already preloaded,
        4. Click "Re-initialize System" if not auto-initialized
        5. Start asking questions!

        ### 💡 Example Questions
        - What are the main themes in Great Expectations?
        - Who is Pip in Charles Dickens' novels?
        - Describe the setting of Oliver Twist
        - What is the plot of A Tale of Two Cities?

        ### 🔧 Backend URL
        `{}`
        """
STREAMLIT_BACKEND_INSTRUCTIONS = """
        ### Starting the Backend

        Make sure that backend is live in Docker after startup
        ```bash
        make docker-up
        docker ps -a | grep api
        ```

        """
STREAMLIT_INITIALIZE_INSTRUCTIONS = """
        ### Getting Started

        Before using the chatbot, you need to:

        1. **Start required services:**
            ```bash
            make docker-up
            ```

        2. **Ingest documents** (Optional since Qdrant is preloaded with backup snapshot):
            - Use the notebook `notebooks/ragflow.ipynb`

        3. **Re-initialize the system** using the sidebar if not properly loaded

        Once initialized, you can start asking questions about Charles Dickens novels!
        """


# Load environment variables
load_dotenv()

# FastAPI backend URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# Page configuration
st.set_page_config(
    page_title="Charles Dickens QA Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    STREAMLIT_CSS,
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "config" not in st.session_state:
        st.session_state.config = None

    if "thread_id" not in st.session_state:
        # Generate random UUID for each conversation
        st.session_state.thread_id = str(uuid.uuid4())


def check_backend_health():
    """Check if FastAPI backend is running and get initialization status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return (True, data.get("initialized", False))
        return False, False
    except requests.exceptions.RequestException:
        return False, False


def get_backend_config():
    """Get configuration from FastAPI backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def initialize_rag_system():
    """Initialize the RAG system via FastAPI backend."""
    try:
        response = requests.post(f"{API_BASE_URL}/initialize", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "System initialized successfully")
        else:
            error_data = response.json()
            return False, error_data.get("detail", "Unknown error occurred")
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"


def query_rag_system(question: str, thread_id: str):
    """Query the RAG system via FastAPI backend."""

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "thread_id": thread_id},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            return {"error": error_data.get("detail", "Unknown error occurred")}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def display_sources(sources):
    """Display source documents in an organized format."""
    if not sources:
        return

    st.markdown("---")
    st.markdown("### 📖 Source Documents")

    for idx, source in enumerate(sources, 1):
        with st.expander(
            f"Source {idx}: {source['metadata'].get('document_title', 'Unknown')}"
        ):
            st.markdown(
                f"**Gutenberg ID:** {source['metadata'].get('gutenberg_id', 'Unknown')}"
            )
            st.markdown(f"**Source:** {source['metadata'].get('source', 'Unknown')}")

            if source["score"] is not None:
                st.markdown(f"**Relevance Score:** {source['score']:.4f}")

            st.markdown("**Context:**")
            st.markdown(f"_{source['text']}_")


def main():
    initialize_session_state()

    # Header
    st.markdown(
        '<div class="main-header">📚 Charles Dickens QA Chatbot</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Ask questions about Charles Dickens novels using RAG</div>',
        unsafe_allow_html=True,
    )

    # Check backend health
    backend_alive, backend_initialized = check_backend_health()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Backend status
        if not backend_alive:
            st.error("❌ Backend Offline")
            st.warning(
                "FastAPI backend is not running. Please start it with:\n\n`make docker-up`"
            )
        elif backend_initialized:
            st.success("✅ System Ready")
            if st.session_state.config is None:
                st.session_state.config = get_backend_config()

            if st.session_state.config:
                st.info(
                    f"**Model:** {st.session_state.config['llm_model']}\n\n"
                    f"**Collection:** {st.session_state.config['collection_name']}\n\n"
                    f"**Opik Project Name:** {st.session_state.config['opik_project']}"
                )
            st.session_state.initialized = True
        else:
            st.warning("⚠️ System Not Initialized")
            st.session_state.initialized = False

        st.markdown("---")

        # Initialize button
        if backend_alive and not backend_initialized:
            if st.button(
                "🚀 Re-initialize System", type="primary", use_container_width=True
            ):
                with st.spinner("Initializing RAG system..."):
                    success, message = initialize_rag_system()

                    if success:
                        st.success(message)
                        st.session_state.initialized = True
                        st.session_state.config = get_backend_config()
                        st.rerun()
                    else:
                        st.error(message)

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown(STREAMLIT_INSTRUCTIONS_EXAMPLES.format(API_BASE_URL))

    # Main chat interface
    if not backend_alive:
        st.error(
            "🚫 Cannot connect to backend. Please ensure FastAPI server is running."
        )
        st.markdown(STREAMLIT_BACKEND_INSTRUCTIONS)
        return

    if not st.session_state.initialized:
        st.info("👈 Please initialize the system using the sidebar.")
        st.markdown(STREAMLIT_INITIALIZE_INSTRUCTIONS)
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about Charles Dickens novels..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_rag_system(prompt, thread_id=st.session_state.thread_id)

                if "error" in result:
                    error_msg = f"Error: {result['error']}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
                else:
                    answer = result["answer"]
                    sources = result["sources"]

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    display_sources(sources)

                    # Add to message history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "⚠️ OPENAI_API_KEY not found in environment variables. Please set it in your .env file."
        )
        st.stop()

    main()
