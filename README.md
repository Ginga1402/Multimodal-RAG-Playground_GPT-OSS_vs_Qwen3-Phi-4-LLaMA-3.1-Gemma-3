# 🚀 Multimodal RAG Playground

## Compare GPT-OSS with Leading Open Source LLMs  
**GPT-OSS vs Qwen3, Phi-4, LLaMA 3.1, Gemma 3**

---

## 📖 Project Description
**Multimodal RAG Playground** is an interactive application that allows you to **benchmark and compare multiple Large Language Models (LLMs)** side-by-side in a **Retrieval-Augmented Generation (RAG)** setup.  

With this tool, you can:
- 📄 Upload PDF documents for semantic indexing  
- 🔍 Ask context-aware questions about your documents  
- 🤖 Get parallel responses from **GPT-OSS, Qwen3, Phi-4, LLaMA 3.1, and Gemma 3**  
- 📊 Evaluate results in a **side-by-side comparison view** for accuracy, reasoning, and style  

Built using **Streamlit**, **LangChain**, **ChromaDB**, **BAAI BGE embeddings** (Hugging Face), and **Ollama** for running local LLMs, the app is designed for developers, researchers, and AI enthusiasts who want to explore how different models perform on the same task.

---

## 💡 Use Cases
- **Model Benchmarking** – Compare multiple open-source LLMs in a consistent RAG pipeline  
- **Research & Evaluation** – Study differences in accuracy, reasoning style, and creativity  
- **Custom Playground** – Easily add any LLM by updating the `model_configuration` list  
- **Local AI Testing** – Run models locally with **Ollama** or connect to hosted APIs  

---

## ⚙️ Installation Instructions


### 1️⃣ Clone the repository
```bash
git clone https://github.com/Ginga1402/Multimodal-RAG-Playground_GPT-OSS_vs_Qwen3-Phi-4-LLaMA-3.1-Gemma-3.git
cd Multimodal-RAG-Playground_GPT-OSS_vs_Qwen3-Phi-4-LLaMA-3.1-Gemma-3
```
### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

---

## 🧰 Technologies Used

| Technology | Description | Link |
|------------|-------------|------|
| **LangChain** | A powerful framework for building applications with LLMs, providing modular tools like retrievers, chains, memory, and agents. Used here to orchestrate the RAG pipeline and handle model interactions. | [LangChain](https://www.langchain.com/) |
| **ChromaDB** | An open-source vector database for storing and retrieving embeddings efficiently, enabling semantic search and context retrieval in RAG workflows. | [ChromaDB](https://www.trychroma.com/) |
| **BAAI BGE Embeddings** | High-quality embedding models from BAAI, hosted on Hugging Face, used for encoding documents into dense vector representations for similarity search. | [BAAI BGE on Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| **Streamlit** | A Python-based rapid app development framework for creating interactive web interfaces, used here to build the model comparison playground. | [Streamlit](https://streamlit.io/) |
| **Ollama** | A local model hosting and management tool for running open-source LLMs directly on your machine without external API calls. | [Ollama](https://ollama.ai/) |
| **GPT-OSS, Qwen3, Phi-4, LLaMA 3.1, Gemma 3** | The set of LLMs benchmarked in this playground, covering both OpenAI's GPT-OSS and leading open-source models. | [OpenAI gpt-oss](https://ollama.com/library/gpt-oss) / [Llama3.1](https://ollama.com/library/llama3.1:8b) / [Gemma](https://ollama.com/library/gemma3)/ [Qwen3](https://ollama.com/library/qwen3)/ [Phi4](https://ollama.com/library/phi4:14b) |
| **Python** | The primary programming language used to develop the application, handle backend logic, and integrate all components. | [Python](https://www.python.org/) |

## 📸 How It Works

Below are sample screenshots demonstrating the workflow and comparison results.

#### Uploading and indexing your document
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/17fa116e-8703-4ddb-9b79-d16df6c62676" />

#### Side-by-side responses from multiple models
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/377e55e6-6d40-4455-b81b-1fdf2d561b2a" />


## 🤝 Contributing
Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
