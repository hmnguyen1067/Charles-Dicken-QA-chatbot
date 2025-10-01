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
        1. Ensure FastAPI backend is running (`make fastapi-app`)
        2. Ensure Qdrant, Redis, and Opik services are running (`make docker-up`)
        3. Documents must be ingested first (use notebooks)
        4. Click "Initialize System" if not auto-initialized
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

        Run the FastAPI backend with:
        ```bash
        make fastapi-app
        ```

        Or directly:
        ```bash
        uvicorn api:app --reload --port 8001
        ```
        """
STREAMLIT_INITIALIZE_INSTRUCTIONS = """
        ### Getting Started

        Before using the chatbot, you need to:

        1. **Start FastAPI backend:**
            ```bash
            make fastapi-app
            ```

        2. **Start required services:**
            ```bash
            make docker-up
            ```

        3. **Ingest documents** (if not already done):
            - Use the notebook `notebooks/ragflow.ipynb`

        4. **Initialize the system** using the sidebar (or it may auto-initialize on backend startup)

        Once initialized, you can start asking questions about Charles Dickens novels!
        """
