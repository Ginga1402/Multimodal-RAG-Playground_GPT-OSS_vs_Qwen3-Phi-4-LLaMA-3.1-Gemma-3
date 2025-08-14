import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_community.chat_models import ChatOllama
import torch
import time
import json
import tempfile

from model_configuration import ollama_model_list

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Playground",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize device for embeddings (CUDA if available)
@st.cache_resource
def initialize_embeddings():
    print("[INIT] Initializing embeddings...")
    try:
        cuda_available = torch.cuda.is_available()
        DEVICE = "cuda" if cuda_available else "cpu"
        print(f"[INIT] CUDA Available: {cuda_available}, Using device: {DEVICE}")
        st.sidebar.success(f"🧠 Device: {DEVICE}")
    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        st.sidebar.error(f"❌ CUDA Error: {e}")
        DEVICE = "cpu"
    
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': True}
    print(f"[INIT] Loading embedding model: {model_name}")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("[INIT] Embeddings initialized successfully")
    return embeddings

def process_pdf(uploaded_file, embeddings):
    """Process uploaded PDF and create in-memory vector store"""
    print(f"[PDF] Processing PDF: {uploaded_file.name}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        print(f"[PDF] Loading PDF from: {tmp_file_path}")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        print(f"[PDF] Loaded {len(documents)} pages")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"[PDF] Split into {len(texts)} text chunks")
        
        # Create in-memory vector store
        print("[PDF] Creating vector store...")
        vector_store = Chroma.from_documents(
            texts, 
            embeddings, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print("[PDF] Vector store created successfully")
        return vector_store, len(documents), len(texts)
    finally:
        os.unlink(tmp_file_path)
        print(f"[PDF] Cleaned up temporary file: {tmp_file_path}")

def get_response_all_models(query, retriever):
    """Query all models and return JSON results"""
    model_list = ollama_model_list
    print(f"[QUERY] Starting multi-model query: '{query}'")
    print(f"[QUERY] Models to process: {model_list}")
    
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain_type_kwargs = {"prompt": prompt}
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model in enumerate(model_list):
        print(f"\n[MODEL {i+1}/{len(model_list)}] Processing with model: {model}")
        status_text.text(f"🤖 Processing with model: {model} ({i+1}/{len(model_list)})")
        
        try:
            print(f"[MODEL] Creating LLM instance for {model}")
            llm = ChatOllama(base_url="http://localhost:11434", model=model, temperature=0.3)
            
            print(f"[MODEL] Creating QA chain for {model}")
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever, 
                return_source_documents=True, 
                chain_type_kwargs=chain_type_kwargs, 
                verbose=False
            )
            
            print(f"[MODEL] Executing query with {model}")
            response = qa(query)
            output_text = response['result']
            print(f"[MODEL] Raw output from {model}: {output_text[:100]}...")
            
            # Handle think tags
            if "</think>" in output_text:
                print(f"[MODEL] Found </think> tag in {model} output")
                answer = output_text.split("</think>")[-1].strip("</s>").strip()
            else:
                answer = output_text.strip("</s>").strip()
            
            model_key = model.split(":")[0]
            results[model_key] = answer
            print(f"[MODEL] Successfully processed {model}, answer length: {len(answer)} chars")
            
        except Exception as e:
            print(f"[ERROR] Error with model {model}: {e}")
            model_key = model.split(":")[0]
            results[model_key] = f"Error: {str(e)}"
        
        progress_bar.progress((i + 1) / len(model_list))
        
        # Sleep between models (except after the last one)
        if i < len(model_list) - 1:
            print(f"[SLEEP] Waiting 5 seconds before next model...")
            time.sleep(5)
    
    print(f"[QUERY] All models processed. Results: {list(results.keys())}")
    status_text.text("✅ Processing complete!")
    return results

# Streamlit UI
st.title("🚀 Multimodal RAG Playground")
st.markdown("""
### Compare GPT-OSS with Leading Open Source LLMs

**GPT-OSS vs Qwen3, Phi-4, LLaMA 3.1, Gemma 3**

### **Compare outputs from 5 leading AI language models in real time**

Easily benchmark and explore how top LLMs perform on the same task.

With this app, you can:
- 📄 Upload & analyze PDF documents instantly
- 🔍 Ask context-aware questions about your files
- 🤖 Get answers from multiple models — Qwen3, Phi-4, LLaMA 3.1, GPT-OSS, Gemma 3
- 📊 View side-by-side comparisons to evaluate accuracy, style, and reasoning

---
""")

# Sidebar with system info
with st.sidebar:
    st.header("🔧 System Information")
    embeddings = initialize_embeddings()
    
    st.header("📋 Available Models")
    models = ["qwen3", "phi4", "llama3.1", "gpt-oss", "gemma3"]
    for i, model in enumerate(models, 1):
        st.write(f"{i}. **{model.split(':')[0].upper()}** `{model}`")
    
    if "doc_stats" in st.session_state:
        st.header("📊 Document Stats")
        st.metric("Pages", st.session_state.doc_stats["pages"])
        st.metric("Text Chunks", st.session_state.doc_stats["chunks"])

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Document Upload")
    st.markdown("Upload a PDF document to start asking questions")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document that you want to ask questions about"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing PDF document..."):
            vector_store, pages, chunks = process_pdf(uploaded_file, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            st.session_state.retriever = retriever
            st.session_state.doc_stats = {"pages": pages, "chunks": chunks}
            
        st.success(f"✅ PDF processed successfully!")
        st.info(f"📄 **{uploaded_file.name}** - {pages} pages, {chunks} text chunks")

with col2:
    st.header("💬 Multi-Model Chat")
    
    if "retriever" not in st.session_state:
        st.warning("⚠️ Please upload a PDF document first to enable chat functionality.")
        st.info("💡 **Tip:** Upload a PDF in the left panel to start asking questions!")
    else:
        st.success("🟢 Ready to answer questions about your document!")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and isinstance(message["content"], dict):
                        st.markdown("**🤖 Model Responses:**")
                        
                        # Create tabs for each model response
                        model_names = list(message["content"].keys())
                        tabs = st.tabs([f"**{name.upper()}**" for name in model_names])
                        
                        for tab, (model, response) in zip(tabs, message["content"].items()):
                            with tab:
                                if response.startswith("Error:"):
                                    st.error(f"❌ {response}")
                                else:
                                    st.markdown(response)
                                    st.caption(f"Response from {model.upper()}")
                    else:
                        st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💭 Ask a question about your PDF document..."):
            print(f"[CHAT] User asked: {prompt}")
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response from all models
            with st.chat_message("assistant"):
                with st.spinner("🔄 Querying all 5 models... This may take up to 30 seconds"):
                    response = get_response_all_models(prompt, st.session_state.retriever)
                    
                st.markdown("**🤖 Model Responses:**")
                
                # Create tabs for each model response
                model_names = list(response.keys())
                tabs = st.tabs([f"**{name.upper()}**" for name in model_names])
                
                for tab, (model, answer) in zip(tabs, response.items()):
                    with tab:
                        if answer.startswith("Error:"):
                            st.error(f"❌ {answer}")
                        else:
                            st.markdown(answer)
                            st.caption(f"Response from {model.upper()}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            print(f"[CHAT] Response added to chat history")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Multimodal RAG Playground | Built with Streamlit & LangChain</p>
    <p>Powered by: Qwen3 • Phi4 • Llama3.1 • GPT-OSS • Gemma3</p>
</div>
""", unsafe_allow_html=True)
