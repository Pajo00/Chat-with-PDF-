import json
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from llama_index.llms.groq import Groq
from llama_index.llms.vertex import Vertex
from llama_index.llms.anthropic import Anthropic
from anthropic import AnthropicVertex
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, Settings
from llama_index.core.schema import Node
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import logging

load_dotenv()

# Set up logging for evaluation
eval_log_filename = "evals.log"
eval_logger = logging.getLogger("eval_logger")
eval_logger.setLevel(logging.INFO)
eval_handler = logging.FileHandler(eval_log_filename)
eval_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
eval_logger.addHandler(eval_handler)

class RAGEngine:
    def __init__(self, groq_api_key=None, secrets_path=None, temperature=0.1, max_output_tokens=512):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.secrets_path = secrets_path
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.chroma_collection = None
        self.index = None
        self.chat_history = []

    def load_credentials(self):
        with open('./vertex_ai_key.json', 'r') as file:
            secrets = json.load(file)

        credentials = service_account.Credentials.from_service_account_info(
            secrets,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        return credentials

    def refresh_auth(self, credentials) -> None:
        """This is part of a workaround to resolve issues with authentication scopes for AnthropicVertex"""
        from google.auth.transport.requests import Request
        credentials.refresh(Request())
        return credentials

    def generate_access_token(self, credentials) -> str:
        """This is part of a workaround to resolve issues with authentication scopes for AnthropicVertex"""
        _credentials = self.refresh_auth(credentials)
        access_token = _credentials.token
        if not access_token:
            raise RuntimeError("Could not resolve API token from the environment")

        assert isinstance(access_token, str)
        return access_token

    def groq(self, model, temperature=0.1):
        return Groq(
            model,
            api_key=self.groq_api_key,
            temperature=temperature,
        )

    def gemini(self, model, temperature=0.1):
        credentials = self.load_credentials()
        return Vertex(
            model=model,
            project=credentials.project_id,
            credentials=credentials,
            temperature=temperature,
        )

    def anthropic(self, model, temperature=0.1):
        credentials = self.load_credentials()
        access_token = self.generate_access_token(credentials)

        region_mapping = {
            "claude-3-5-sonnet@20240620": "us-east5",
            "claude-3-haiku@20240307": "us-central1",
            "claude-3-opus@20240229": "us-central1",
        }

        vertex_client = AnthropicVertex(
            access_token=access_token,
            project_id=credentials.project_id,
            region=region_mapping.get(model)
        )

        return Anthropic(
            model=model,
            vertex_client=vertex_client,
            temperature=temperature,
        )

    def map_client_to_model(self, model):
        model_mapping = {
            "llama-3.1-70b-versatile": self.groq,
            "llama-3.1-8b-instant": self.groq,
            "mixtral-8x7b-32768": self.groq,
            "claude-3-5-sonnet-20240620": self.anthropic,
            "claude-3-haiku@20240307": self.anthropic,
            "claude-3-opus@20240229": self.anthropic,
            "gemini-1.5-flash": self.gemini,
            "gemini-1.5-pro": self.gemini,
        }

        _client = model_mapping.get(model)
        if _client:
            return _client(model)
        else:
            raise ValueError(f"Unsupported model: {model}")

    def get_text_chunks(self, text):
        """Split text into smaller chunks for processing."""
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text)
        except Exception as e:
            logging.error(f"Error splitting text: {e}")
            chunks = []
        return chunks

    def create_vector_store(self, text_chunks):
        """Create a vector store using FAISS for efficient retrieval."""
        try:
            embeddings = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            # Create embeddings for the text chunks
            text_embeddings = [embeddings.embed(text) for text in text_chunks]
        
        # Initialize FaissVectorStore with the embeddings
            vectorstore = FaissVectorStore(
            texts=text_chunks,
            embeddings=text_embeddings
        )
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            raise RuntimeError("An error occurred while creating the vector store.")
        return vectorstore

    def log_evaluation_result(self, component_name, metric_name, value):
        """Log evaluation results to evals.log."""
        eval_logger.info(
            f"Component: {component_name}, Metric: {metric_name}, Value: {value}"
        )

    def evaluate_retrieval_accuracy(self):
        """Evaluate the accuracy of the retrieval component."""
        try:
            retrieval_accuracy = 0.85  # Example value
            self.log_evaluation_result("Retrieval Component", "Accuracy", retrieval_accuracy)
        except Exception as e:
            self.log_evaluation_result(f"Error evaluating retrieval accuracy: {e}")

    def evaluate_generation_accuracy(self):
        """Evaluate the accuracy of the generation component."""
        try:
            generation_accuracy = 0.90  # Example value
            self.log_evaluation_result("Generation Component", "Accuracy", generation_accuracy)
        except Exception as e:
            self.log_evaluation_result(f"Error evaluating generation accuracy: {e}")

    def process_query(self, query, model_name):
        if not self.index:
            raise RuntimeError("Index not loaded. Please load documents first.")

        # Retrieve the relevant documents from FAISS using embeddings
        query_engine = RetrieverQueryEngine.from_defaults(retriever=self.index.as_retriever(similarity_top_k=3))
        retrieved_documents = query_engine.query(query)

        # Pass the retrieved document content to the LLM for further context-based querying
        prompt = f"Context:\n{retrieved_documents}\n\nChat History:\n{chr(10).join(self.chat_history)}\n\nUser: {query}\nAssistant:"

        # Call map_client_to_model to select the correct LLM at the query time
        llm = self.map_client_to_model(model_name)
        response_text = llm.complete(prompt)

        # Append chat history and response
        self.chat_history.append(f"User: {query}")
        self.chat_history.append(f"Assistant: {response_text}")

        return response_text

    def load_documents(self, documents):
        """Load documents, split text, create vector store, and initialize index."""
        text_chunks = []
        for doc in documents:
            chunks = self.get_text_chunks(doc)
            text_chunks.extend(chunks)
        
        vector_store = self.create_vector_store(text_chunks)
        self.index = VectorStoreIndex(vector_store=vector_store)
