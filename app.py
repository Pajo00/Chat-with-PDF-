from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from models import RAGEngine 
import uvicorn
import tempfile
import os
import traceback
import PyPDF2
#from helpers import system_logger, userops_logger, llmresponse_logger 

# Initialize FastAPI app
app = FastAPI()

# Dictionary to store RAGEngine instances
rag_engine = RAGEngine()

@app.get("/")  
async def root():
    return {"message": "Welcome to the RAG Chatbot API!"}

@app.get('/healthz')
async def health():
    return {
        "application": "Simple LLM API",
        "message": "running successfully"
    }

@app.post("/upload_files/")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        # Store files temporarily and extract content
        documents = []
        for file in files:
            file_content = await file.read()
            if file.filename.endswith('.pdf'):
                # Use PyMuPDF or PyPDF2 to extract text from PDF
                pdf_reader = PyPDF2.PdfReader(file_content)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()

                if text.strip():
                    documents.append(text)
                else:
                    print(f"Warning: No text extracted from PDF: {file.filename}")
            else:
                try:
                    text = file_content.decode('utf-8')
                except UnicodeDecodeError:
                    text = file_content.decode('latin1')

                if text.strip():
                    documents.append(text)
                else:
                    print(f"Warning: Empty file: {file.filename}")

        if not documents:
            return JSONResponse(content={"error": "No documents with content found. Please check your files."}, status_code=400)

        # Load documents into RAG engine and check if index is set
        rag_engine.load_documents(documents)
        if rag_engine.index:
            print("Index successfully loaded with documents")
        else:
            print("Failed to load index")
            return JSONResponse(content={"error": "Failed to load documents into the index."}, status_code=500)

        return JSONResponse(content={"status": "Documents uploaded and processed successfully"}, status_code=200)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"error": f"Could not upload files: {e}"}, status_code=500)

@app.post("/chat/")
async def chat(request: Request):
    try:
        # Print the incoming request JSON data for debugging purposes
        query_data = await request.json()
        print(f"Incoming query data: {query_data}")

        # Extract the question and model from the JSON data
        query = query_data.get("question")
        model_name = query_data.get("model")

        # Check if both 'question' and 'model' are provided
        if not query or not model_name:
            return JSONResponse(content={"error": "Invalid input. Ensure 'question' and 'model' are provided."}, status_code=400)

        # Process the user's query using the engine
        response = rag_engine.process_query(query, model_name)
        return PlainTextResponse(content=response, status_code=200)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"error": f"Could not process query: {e}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)