import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough



API_KEY = "AIzaSyAO0YIheYwDRO4R_62bS-gEW7j5aAX_tjU"
st.title(" RAG System on “Leave No Context Behind” Paper")


chat_template = ChatPromptTemplate.from_messages([

    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

chat_model = ChatGoogleGenerativeAI(google_api_key=API_KEY, 
                                   model="gemini-1.5-pro-latest")

output_parser = StrOutputParser()

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, 
                                               model="models/embedding-001")

db_connection = Chroma(persist_directory="./embeddings/", embedding_function=embedding_model)

retriever = db_connection.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

question = st.text_area("Input your query related to paper")
submit = st.button("Submit")
if submit:
    response = rag_chain.invoke(question)
    st.write(response)