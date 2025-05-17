## 🚀 AI-Powered Document Research Chatbot

This is a Flask-based chatbot that allows users to upload documents (PDF, TXT, PNG, JPG), extract their contents using OCR, store them in a local SQLite database, and perform research-based queries using AI. It uses **Google Gemini Pro**, **LangChain**, **Pinecone**, and **OCR tools** to answer questions and find themes across documents with cited sources.

### 📌 Features

* ✅ Upload PDFs, text files, or scanned images (JPG/PNG)
* ✅ Extract text using PyPDF2 and Tesseract OCR
* ✅ Store and manage uploaded documents
* ✅ Ask questions across multiple documents
* ✅ Get AI-generated answers with **source citations**
* ✅ Identify **common themes** across selected documents
* ✅ Simple and clean **web interface** using Flask & HTML

---

### 🛠️ Tech Stack

| Layer        | Tech Used                                      |
| ------------ | ---------------------------------------------- |
| Backend      | Python, Flask                                  |
| Document OCR | PyPDF2, pdf2image, pytesseract                 |
| Database     | SQLite                                         |
| LLM & QA     | LangChain, Gemini Pro (ChatGoogleGenerativeAI) |
| Embeddings   | Google Generative AI Embeddings                |
| Vector DB    | Pinecone                                       |
| Frontend     | Jinja2 templates (HTML)                        |
| Deployment   | Docker, GitHub                                 |

---

### 📁 Project Structure

```
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── uploads/
├── file_store.db
├── Dockerfile
└── README.md
```

---

### ⚙️ Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/shuvendu-barik/wasserstoff/AiInternTask.git

cd AiInternTask
```

#### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the App

```bash
python app.py
```

Then open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 📦 Docker Deployment

Build and run using Docker:

```bash
docker build -t document-chatbot .
docker run -p 5000:5000 document-chatbot
```

---


### 📄 License

This project is developed as part of an internship task submission. All rights reserved.

---

### 🙋 Author

**Shuvendu Barik**
[GitHub](https://github.com/shuvendu-git) · [LinkedIn](https://www.linkedin.com/in/shuvendubarik)

---