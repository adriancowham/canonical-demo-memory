"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
"""

import logging
import os
import re
from typing import Any, Dict

# Import necessary libraries
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

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

VECTOR_STORE = "faiss"
MODEL = "openai"
EMBEDDING = "openai"
MODEL = "gpt-3.5-turbo-16k"
K = 10
USE_VERBOSE = True

# Set Streamlit page configuration
st.set_page_config(page_title="Canonical.chat Demo. Let's Talk...", layout='wide')

class AnswerConversationBufferMemory(ConversationBufferMemory):
  def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})

system_template = """
Use the context below to answer questions. You must only use the Context to answer questions. If you cannot find the answer from the Context below, you must respond with
"I'm sorry, but I can't find the answer to your question in the book, 'Let's Talk...,' by Andrea Lunsford." All answers must be in English unless you are explicitly asked to translate to a different language.
----------------
{context}
{chat_history}
"""

# Create the chat prompt templates
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

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
                            placeholder="Let's talk...",
                            on_change=clear_text,
                            label_visibility='hidden')
    input_text = st.session_state["temp"]
    return input_text

@st.cache_resource(show_spinner=False)
def getretriever(_llm):
  with open("./resources/lets-talk.pdf", 'rb') as uploaded_file:
    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e)

  chunked_file = chunk_file(file, chunk_size=512, chunk_overlap=0)
  with st.spinner("Loading book..."):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING,
        vector_store=VECTOR_STORE,
        openai_api_key=API_KEY,
    )
    vector_retriever = folder_index.index.as_retriever(verbose=True, search_type="similarity", search_kwargs={"k": K})
    return vector_retriever

def getanswer(question, chat):
  output = chat({"question": question})
  output = output["answer"]
  st.session_state.past.append(question)
  st.session_state.generated.append(output)
  st.session_stat["temp"] = ""
  return output

# Set up the Streamlit app layout
st.title("Canonical.chat Demo")
st.subheader("Let's Talk...")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

API_KEY = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(
        openai_api_key=API_KEY,
        model_name=MODEL,
        verbose=True)
retriever = getretriever(llm)

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
  st.title("Suggested Questions")
  if st.button("How can I make myself be heard?"):
    getanswer("How can I make myself be heard?", chat)
  if st.button("How can I connect with people I disagree with?"):
    getanswer("How can I connect with people I disagree with?", chat)
  if st.button("How do I come up with ideas for my essay?"):
    getanswer("How do I come up with ideas for my essay?", chat)
  if st.button("My professor reviewed my first draft. She circled a sentence and said I need to support it more. How do I do that?"):
    getanswer("My professor reviewed my first draft. She circled a sentence and said I need to support it more. How do I do that?", chat)
  if st.button("How do I cite a Reddit thread?"):
    getanswer("How do I cite a Reddit thread?", chat)

user_input = get_text()
if user_input:
  output = getanswer(user_input, chat)

with st.expander("Conversation", expanded=True):
  for i in range(len(st.session_state['generated'])-1, -1, -1):
    st.info(st.session_state["past"][i])
    st.success(st.session_state["generated"][i], icon="🤖")

