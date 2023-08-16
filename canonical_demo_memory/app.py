"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
"""

import os
from typing import Any, Dict

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from ui import display_file_read_error

from canonical_demo_memory.core.chunking import chunk_file
from canonical_demo_memory.core.embedding import embed_files
from canonical_demo_memory.core.parsing import read_file

VECTOR_STORE = "faiss"
MODEL = "openai"
EMBEDDING = "openai"
MODEL = "gpt-3.5-turbo-16k"
K = 3
USE_VERBOSE = True

class AnswerConversationBufferMemory(ConversationBufferMemory):
  def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})

system_template = """
You are a college English Professor, you teach english composition. Your textbook is The Little Seagull Handbook and you assign this textbook to your students.
Your are currently sitting in your office during office hours, enjoying an espresso and having a conversation about The Little Seagull Handbook with one of your students. Your student is asking you questions about the book.
Use the context below to answer the questions. You must only use the Context to answer questions. If you cannot find the answer from the Context below, you must respond with
"I'm sorry, but I can't find the answer to your question in, The Little Seagull Handbook." All answers must be in English unless you are explicitly asked to translate to a different language.

Here is the context:
{context}
{chat_history}
"""
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)
st.set_page_config(page_title="Canonical.chat Demo. The Little Seagull Handbook", layout='wide')
# Initialize session states
# st.session_state["temp"] = ""
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "just_sent" not in st.session_state:
    st.session_state["just_sent"] = False
if "temp" not in st.session_state:
    st.session_state["temp"] = ""

def clear_text():
    st.session_state["temp"] = st.session_state["input"]
    st.session_state["input"] = ""

def get_text():
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Ask The Little Seagull...",
                            on_change=clear_text,
                            label_visibility='hidden')
    input_text = st.session_state["temp"]
    return input_text

@st.cache_data(show_spinner=False)
def getretriever():
  with open("./resources/little-seagull.pdf", 'rb') as uploaded_file:
    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e)

  chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)
  with st.spinner("Loading book..."):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING,
        vector_store=VECTOR_STORE,
        openai_api_key=API_KEY,
    )
    return folder_index.index.as_retriever(verbose=True, search_type="similarity", search_kwargs={"k": K})

def getanswer(question, chat):
  output = chat({"question": question})
  output = output["answer"]
  st.session_state.past.append(question)
  st.session_state.generated.append(output)
  return output

# Set up the Streamlit app layout
st.title("Canonical.chat Demo")
st.subheader("The Little Seagull Handbook")

# Read API from Streamlit secrets
API_KEY = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(
        openai_api_key=API_KEY,
        model_name=MODEL,
        verbose=True)

retriever = getretriever()
if 'entity_memory' not in st.session_state:
  st.session_state.entity_memory = AnswerConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat = ConversationalRetrievalChain.from_llm(
  llm,
  retriever=retriever,
  return_source_documents=USE_VERBOSE,
  memory=st.session_state.entity_memory,
  verbose=USE_VERBOSE,
  combine_docs_chain_kwargs={"prompt": qa_prompt})

with st.sidebar:
  if st.button("What are some strategies for writing a first draft?"):
    getanswer("What are some strategies for writing a first draft?", chat)
  if st.button("How do I find sources for my research paper?"):
    getanswer("How do I find sources for my research paper?", chat)
  if st.button("When I'm citing a source in text, can I just write the page number? Provide examples."):
    getanswer("When I'm citing a source in text, can I just write the page number? Provide examples.", chat)
  if st.button('Do I say, "I wish I was a dolphin" or "I wish I were a dolphin"?'):
    getanswer('Do I say, "I wish I was a dolphin" or "I wish I were a dolphin"?', chat)
  if st.button("How do I punctuate dialogue? Provide examples of different methods."):
    getanswer("How do I punctuate dialogue? Provide examples of different methods.", chat)

user_input = get_text()
if user_input:
  getanswer(user_input, chat)

with st.expander("Conversation", expanded=True):
  for i in range(len(st.session_state['generated'])-1, -1, -1):
    st.info(st.session_state["past"][i])
    st.success(st.session_state["generated"][i], icon="🤖")

