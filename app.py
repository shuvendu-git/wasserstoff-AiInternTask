# Import necessary modules
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import os
import warnings
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pytesseract import image_to_string
import sqlite3

# Suppress any unnecessary warnings
warnings.filterwarnings('ignore')

# Define upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'png', 'jpg', 'jpeg'}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session security
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

# Set up SQLite database to store document content and metadata
DB_PATH = "file_store.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, content TEXT, metadata TEXT)''')
conn.commit()
conn.close()

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# OCR extraction for image files
def ocr_image(file_path):
    return image_to_string(file_path)

# OCR for scanned PDF pages
def ocr_pdf(file_path):
    images = convert_from_path(file_path)
    text = ""
    for img in images:
        text += image_to_string(img)
    return text

# Main text extraction function for PDF, image, and text files
def extract_text(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text if text.strip() else ocr_pdf(file_path)  # Fallback to OCR if no text
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return ocr_image(file_path)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return ""
    except:
        return ""

# Initialize the LangChain QA system with Pinecone and Gemini 
def initialize_qa_system():
    # Load all documents from DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, content FROM documents")
    rows = c.fetchall()
    conn.close()

    all_docs = []
    for doc_id, filename, content in rows:
        metadata = {"doc_id": f"DOC{doc_id:03d}", "source": filename}
        all_docs.append({"content": content, "metadata": metadata})

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = []
    for doc in all_docs:
        split = text_splitter.create_documents([doc["content"]], metadatas=[doc["metadata"]])
        split_docs.extend(split)

    # Generate embeddings using Google Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key="AIzaSyBW9OS7P95Rw5zswVkYIAkJjM1Rcpljto0",
        task_type="retrieval_query"
    )

    # Store vectors in Pinecone
    os.environ['PINECONE_API_KEY'] = "pcsk_KQWq2_9v4Axog6B5hpfbk3RZtnx24HMziXBwzChJNvvsGqYwtG4X1yFRsUyEU6vLjVDab"
    index_name = "rag-index"
    vectordb = PineconeVectorStore.from_documents(split_docs, embeddings, index_name=index_name)

    # Define prompt template for the research assistant
    prompt_template = """
    You are a research assistant. Use the context to answer the question with cited evidence.
    Context:\n{context}
    Question:\n{question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Define LLM with safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="AIzaSyBW9OS7P95Rw5zswVkYIAkJjM1Rcpljto0",
        temperature=0.5,
        safety_settings=safety_settings
)

    # Enhance retrieval using multiple LLM queries
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    # Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# Initialize the QA system once
qa_chain = initialize_qa_system()

# Helper to get chat history from session
def get_conversation_history():
    if 'conversation' not in session:
        session['conversation'] = []
    return session['conversation']

# Update session with latest user-bot interaction
def update_conversation_history(user_question, bot_response):
    conversation = get_conversation_history()
    conversation.append({"user": user_question, "bot": bot_response})
    session['conversation'] = conversation

# Home page: shows document list, accepts queries
@app.route("/", methods=["GET", "POST"])
def index():
    # Load uploaded documents
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename FROM documents")
    documents = c.fetchall()
    conn.close()

    if request.method == "POST":
        # Process user query
        user_question = request.form["question"]
        selected_docs = request.form.getlist("selected_docs")

        # Combine selected document contexts
        context_docs = []
        if selected_docs:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            for doc_id in selected_docs:
                c.execute("SELECT content FROM documents WHERE id = ?", (doc_id,))
                doc = c.fetchone()
                if doc:
                    context_docs.append(doc[0])
            conn.close()

        combined_context = "\n".join(context_docs)
        synthetic_query = f"What are the common themes across these documents? {user_question}"
        
        # Get LLM-based response
        response = qa_chain.invoke({"query": synthetic_query})
        bot_response = response['result']

        # Extract citation metadata
        source_documents = response.get('source_documents', [])
        source_info = []
        seen_sources = set()
        for doc in source_documents:
            meta = doc.metadata
            doc_id = meta.get("doc_id", "Unknown")
            source_name = meta.get("source", "Unknown")
            unique_key = (doc_id, source_name)
            if unique_key not in seen_sources:
                seen_sources.add(unique_key)
                source_info.append({
                    "doc_id": doc_id,
                    "source": source_name
                })

        update_conversation_history(user_question, bot_response)

        # Render page with result and citations
        return render_template("index.html", user_question=user_question, bot_response=bot_response, source_documents=source_info, documents=documents)

    return render_template("index.html", user_question=None, bot_response=None, documents=documents)

# File upload endpoint
@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    files = request.files.getlist("file")
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            content = extract_text(filepath)
            if content:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("INSERT INTO documents (filename, content) VALUES (?, ?)", (filename, content))
                conn.commit()
                conn.close()
    return redirect(url_for('index'))

# Delete document from system
@app.route("/delete/<int:doc_id>", methods=["POST"])
def delete_document(doc_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Delete the actual file
    c.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
    row = c.fetchone()
    if row:
        filename = row[0]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Delete from database
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

    conn.close()
    return redirect(url_for('index'))

# Start Flask server
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

