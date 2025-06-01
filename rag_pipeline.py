#Step 1 : LLM Set-Up (Deepseek R1 with Groq)
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key=GROQ_API_KEY
)

#Step 2 : Retrieve Docs
from vectordb import faiss_db
from langchain_core.output_parsers.string import StrOutputParser

def retrieve_docs(query):
    return faiss_db.similarity_search(query)
    
def get_context(documents) :
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


#Step 3 : Answer Questions
from langchain.prompts import ChatPromptTemplate

custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer , just say that you don't know , don't try to make up an answer.
Don't provide anything out of the given context.
Question : {question} 
Context : {context}
Answer :
"""

def answer_query(documents, model, query) :
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model 
    return chain.invoke({"question" : query,
                 "context" : context
                 })

# question = "If a government forbids the right to assemble peacefully which articles are violated and why ?"
# retrieved = retrieve_docs(question)
# print("AI Lawyer : ",  
#       answer_query(documents=retrieved, model=llm, query= question)
#       )


    
