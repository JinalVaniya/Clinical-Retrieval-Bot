import os

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Setup LLM (GROQ with HuggingFace)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


llm = ChatGroq(model_name=GROQ_MODEL_NAME,
               api_key=GROQ_API_KEY,
               temperature= 0.6,
               max_tokens=512 )

# Connect LLM with FAISS and Create chain
DB_FAISS_PATH = "vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE = """

You are a medical assistant.

Answer the question ONLY from the provided medical context.

If the information is insufficient, say:
"I could not find enough information in the provided medical documents."

Provide:
- concise medical explanation
- treatment if available
- important warnings if relevant

Context:
{context}

Question:
{question}

Answer:
"""

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                        cache_folder=".\hf_cache")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization = True)

# Create QA chain

def get_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, 
                            input_variables=['context', 'question']) 
   
rag_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type = "stuff",
                                        retriever=db.as_retriever(search_kwargs={'k': 3}),
                                        return_source_documents=True,
                                        chain_type_kwargs = {'prompt': get_prompt()}
                                        )

# Run chain with a Query
user_query = input("Write your prompt: ")
response = rag_chain.invoke({'query': user_query})
print("\n================ ANSWER ================\n")
print(response["result"])
print("\n================ SOURCES ================\n")

for i, doc in enumerate(response["source_documents"], start=1):

    source = doc.metadata.get("source", "Unknown Source")
    page = doc.metadata.get("page", "N/A")

    print(f"\nSource {i}")
    print(f"File : {source}")
    print(f"Page : {page}")

    print("\nExcerpt:")
    print(doc.page_content[:300])
    print("-" * 60)
