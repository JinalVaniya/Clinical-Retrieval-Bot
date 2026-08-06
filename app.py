import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from pathlib import Path

from LLM_connection import get_prompt   

load_dotenv()

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing.")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.6,
    max_tokens=512,
)


embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                        cache_folder=".\hf_cache")

DB_FAISS_PATH = Path("VectorStore/db_faiss")

if not DB_FAISS_PATH.exists():
    st.error(f"Missing FAISS folder: {DB_FAISS_PATH}")
    st.stop()

try:
    db = FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
except Exception as e:
    st.error(f"Could not load FAISS index: {e}")
    st.stop()
    
prompt = get_prompt()

# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization = True)

rag_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type = "stuff",
                                        retriever=db.as_retriever(search_kwargs={'k': 3}),
                                        return_source_documents=True,
                                        chain_type_kwargs = {'prompt': prompt})


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=".\hf_cache",
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db


def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_prompt = st.chat_input("Pass your prompt here")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            vectorstore = get_vectorstore()

            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                st.error("GROQ_API_KEY is missing.")
                return

            llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.5,
                max_tokens=512,
            )

            prompt = get_prompt()

            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )

            response = rag_chain.invoke({"query": user_prompt})

            st.chat_message("assistant").markdown(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

            with st.expander("Source documents"):
                for i, doc in enumerate(response["source_documents"], start=1):
                    source = doc.metadata.get("source", "Unknown source")
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"**{i}. {source} | Page {page}**")
                    st.write(doc.page_content[:500])

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()