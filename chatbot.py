import streamlit as st
from models import RAGEngine
import PyPDF2

# Mapping for user-friendly model names to actual model names
MODEL_NAME_MAPPING = {
    "Claude": "claude-3-5-sonnet-20240620",
    "Gemini Pro": "gemini-1.5-pro",
    "LLaMA 70B": "llama-3.1-70b-versatile",
}

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def main():
    st.title("RAG Model Streamlit Interface")
    st.write("Upload your documents and interact with the model!")

    # Step 1: Upload files
    st.header("Step 1: Upload Files")
    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

    # Display simplified model names in the selectbox
    user_friendly_model_name = st.selectbox("Select model to use", list(MODEL_NAME_MAPPING.keys()))

    # Get the actual model name from the dictionary
    model_name = MODEL_NAME_MAPPING[user_friendly_model_name]

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if st.button("Upload and Process"):
        if uploaded_files:
            with st.spinner("Uploading and processing..."):
                # Initialize the RAGEngine instance and store it in session_state
                if st.session_state.rag_engine is None:
                    st.session_state.rag_engine = RAGEngine()

                # Load documents using the created rag_engine
                documents = []
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "text/plain":
                        documents.append(uploaded_file.read().decode("utf-8"))
                    elif uploaded_file.type == "application/pdf":
                        documents.append(extract_text_from_pdf(uploaded_file))
                
                try:
                    st.session_state.rag_engine.load_documents(documents)
                    st.success("Files uploaded and processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred while processing documents: {e}")
        else:
            st.error("Please upload at least one file.")

    # Step 2: Ask a question
    st.header("Step 2: Ask a Question")
    query = st.text_input("Enter your question")

    if st.button("Ask Model"):
        if query:
            if st.session_state.rag_engine is not None:
                with st.spinner("Getting the response from the model..."):
                    try:
                        response = st.session_state.rag_engine.process_query(query, model_name)
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred while processing your query: {e}")
            else:
                st.error("Model not initialized. Upload files first.")
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
