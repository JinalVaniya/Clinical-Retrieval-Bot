import os
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv
load_dotenv()

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

# Create QA chain

def get_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, 
                            input_variables=['context', 'question']) 
   