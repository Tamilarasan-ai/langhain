from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Set up environment variables for Google API Key and LangChain Tracing
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI setup
st.title('Langchain Demo With Google Gemini LLM API')
input_text = st.text_input("Search the topic you want")

# Initialize Google Gemini LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # specify the version of the Gemini model
    temperature=0.7,         # you can adjust temperature for creativity
    max_tokens=150,          # adjust tokens based on response length needed
    timeout=10,              # specify timeout for the API call
)

# Set up output parser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Invocation
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
