
import os
import requests
import json
import streamlit as st
from urllib.parse import quote
from dotenv import load_dotenv

# LangChain imports
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

# Gemini imports
import google.generativeai as genai

# Configure API
load_dotenv()
opengraph_app_id = os.environ.get("OPENGRAPH_APP_ID")
# Configure page
st.set_page_config(
    page_title="Document QA System",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'prompt_template' not in st.session_state:
    st.session_state.prompt_template = None

# Gemini embeddings class
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text):
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result["embedding"]

# Custom LLM wrapper for Gemini
class GeminiLLM(LLM):
    def _call(self, prompt, stop=None):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self):
        return {"name_of_model": "gemini-1.5-flash"}

    @property
    def _llm_type(self):
        return "custom-gemini"

# Existing helper functions
def get_url_content(url):
    base_url = "https://opengraph.io/api/1.1/extract"
    encoded_url = quote(url, safe='')
    api_url = f"{base_url}/{encoded_url}"

    params = {
        "accept_lang": "auto",
        "html_elements": "h1,h2,p",
        "app_id": opengraph_app_id
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing the URL: {e}")
        return None

def process_extracted_data(data):
    text_contents = []
    for tag in data.get('tags', []):
        if tag.get('innerText'):
            text_contents.append(tag['innerText'])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text('\n'.join(text_contents))

    embeddings = GeminiEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def create_retrieval_qa_chain(vectorstore):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use the following context:\n"
            "{context}\n\n"
            "Answer the question below using only the provided context, If you don't know the answer, just say that the answer can't be infered from the retrieved text.\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    chain = RetrievalQA.from_chain_type(
        llm=GeminiLLM(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return chain, prompt_template

def main():
    st.title("📚 Document QA System")
    
    # Configure API keys
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            
    # URL input and processing
    url = st.text_input("Enter URL to process:", "https://www.presight.io/privacy-policy.html")
    
    if st.button("Process URL"):
        with st.spinner("Processing URL..."):
            data = get_url_content(url)
            if data:
                st.session_state.vectorstore = process_extracted_data(data)
                st.session_state.chain, st.session_state.prompt_template = create_retrieval_qa_chain(
                    st.session_state.vectorstore
                )
                st.success("URL processed successfully!")
            else:
                st.error("Failed to process URL")

    # Q&A Section
    if st.session_state.vectorstore is not None:
        st.header("Ask Questions")
        user_question = st.text_input("Enter your question:")
        
        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Thinking..."):
                    docs = st.session_state.chain.retriever.get_relevant_documents(user_question)
                    context = "\n".join([d.page_content for d in docs])
                    
                    answer_results = st.session_state.chain({"query": user_question})
                    answer = answer_results["result"]
                    
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    with st.expander("View Context"):
                        st.write(context)
            else:
                st.warning("Please enter a question")
    else:
        st.info("Please process a URL first to start asking questions")

if __name__ == "__main__":
    main()