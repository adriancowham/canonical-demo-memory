"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, cast

import faiss
# Import necessary libraries
import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import \
    ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.retrievers.llama_index import LlamaIndexRetriever
from langchain.schema import BaseRetriever, Document
from llama_index import StorageContext, VectorStoreIndex, download_loader
from llama_index.vector_stores.faiss import FaissVectorStore
from pydantic import Field

from canonical_demo_memory.core.caching import bootstrap_caching
from canonical_demo_memory.core.chunking import chunk_file
from canonical_demo_memory.core.embedding import embed_files
from canonical_demo_memory.core.parsing import read_file
from canonical_demo_memory.core.qa import query_folder
from ui import (display_file_read_error, is_file_valid, is_open_ai_key_valid,
                is_query_valid, wrap_doc_in_html)

VECTOR_STORE = "faiss"
MODEL = "openai"
EMBEDDING = "openai"
MODEL = "gpt-3.5-turbo-16k"
K = 100
USE_VERBOSE = True

class CanonicalLlamaIndexRetriever(BaseRetriever):
    """Retriever for the question-answering with sources over
    an LlamaIndex data structure."""

    index: Any
    """LlamaIndex index to query."""
    query_kwargs: Dict = Field(default_factory=dict)
    """Keyword arguments to pass to the query method."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.indices.base import BaseGPTIndex
            from llama_index.response.schema import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(BaseGPTIndex, self.index)

        response = index.as_query_engine(response_mode="no_text", **self.query_kwargs).query(query)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.node.metadata or {}
            docs.append(
                Document(page_content=source_node.node.get_content(), metadata=metadata)
            )
        return docs

class AnswerConversationBufferMemory(ConversationBufferMemory):
  def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
    return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})
# bootstrap_caching()

system_template = """
You are a knowledgeable software developer and you work on a team with other software developers. You and your team use Git every day and you can answer any question about it. You are a Git wizard and expert.
Use the Context below to answer questions about Git. You must only use the Context to answer questions. If you cannot find the answer from the Context below, you must respond with
"I'm sorry, but I can't find the answer to your question in the ProGit book."
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

# Set Streamlit page configuration
st.set_page_config(page_title="Canonical.chat Demo", layout='wide')
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

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Let's Talk...",
                            on_change=clear_text,
                            label_visibility='hidden')
    input_text = st.session_state["temp"]
    return input_text


    # Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

@st.cache_resource(show_spinner=False)
def getretriever():
  with st.spinner("Loading Let's Talk..."):
    with open('./resources/progit.pdf', 'rb') as uploaded_file:
      try:
        file = read_file(uploaded_file)
      except Exception as e:
        display_file_read_error(e)

    chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

    faiss_index = faiss.IndexFlatL2(1536)
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path('./resources/progit.pdf'))
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_query_engine()
    # class CompatibleIndex():
      # def query(self, query, response_mode, **kwargs):
        # return query_engine.query(query)
    # tools = [
    #   Tool(
    #     name="LlamaIndex",
    #     func=lambda q: str(index.as_query_engine().query(q)),
    #     description="Useful for when you want to answer questions about Git. The input to this tool should be a complete english sentence.",
    #     return_direct=True,
    #   ),
    # ]
    # return tools
    return CanonicalLlamaIndexRetriever(index=index, vector_store=vector_store, embeddings=OpenAIEmbeddings)

    # folder_index = embed_files(
    #     files=[chunked_file],
    #     embedding=EMBEDDING,
    #     vector_store=VECTOR_STORE,
    #     openai_api_key=API_KEY,
    # )
    # return folder_index.index.as_retriever(verbose=True, search_type="similarity", search_kwargs={"k": 10})

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

# Read API from Streamlit secrets
API_KEY = os.environ["OPENAI_API_KEY"]

# Session state storage would be ideal
if API_KEY:
  # Create an OpenAI instance
  llm = ChatOpenAI(
          openai_api_key=API_KEY,
          model_name=MODEL,
          verbose=True)

# if 'loaded' not in st.session_state:
#   st.session_state.loaded = 'loaded'
#   with open('./resources/progit.pdf', 'rb') as uploaded_file:
#     try:
#         file = read_file(uploaded_file)
#     except Exception as e:
#         display_file_read_error(e)

#   chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

#   folder_index = None
#   with st.spinner("Indexing document...this may take a while."):
#       folder_index = embed_files(
#           files=[chunked_file],
#           embedding=EMBEDDING,
#           vector_store=VECTOR_STORE,
#           openai_api_key=API_KEY,
#       )
#       # Create a ConversationEntityMemory object if not already created
#       if 'entity_memory' not in st.session_state:
#         st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )

retriever = getretriever()

# Create a ConversationEntityMemory object if not already created
if 'entity_memory' not in st.session_state:
  st.session_state.entity_memory = AnswerConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Create the ConversationChain object with the specified configuration
# agent_executor = initialize_agent(
#   tools, llm, agent="conversational-react-description", memory=st.session_state.entity_memory
# )

chat = ConversationalRetrievalChain.from_llm(
  llm,
  retriever=retriever,
  return_source_documents=USE_VERBOSE,
  memory=st.session_state.entity_memory,
  verbose=USE_VERBOSE,
  combine_docs_chain_kwargs={"prompt": qa_prompt})
# chat.rephrase_question = True

# Conversation = ConversationChain(
#         llm=llm,
#         prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
#         memory=st.session_state.entity_memory
#     )

# Get the user input
user_input = get_text()
# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    with get_openai_callback() as cb:
      # ouput = agent_executor.run(input=user_input)
      output = chat({"question": user_input})
      output = output["answer"]
      st.session_state.past.append(user_input)
      st.session_state.generated.append(output)

with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i], icon="🤖")
