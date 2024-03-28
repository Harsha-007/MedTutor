import os
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import torch
from llama_index import set_global_service_context
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import chromadb
from llama_index.vector_stores import PineconeVectorStore

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=3000,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=4000,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)
set_global_service_context(service_context)

# from pinecone import Pinecone
# pinecone_api_key = "a51bd13c-0dee-42b1-9ff0-2b2f5fba771b"

# pc = Pinecone(
#         api_key = pinecone_api_key
#     )
# pinecone_index = pc.Index("megaindex")

# import os
# import shutil

# book_titles = []
# path = "./test"
# for entry in os.scandir(path):
#     book_titles.append(str(entry.name))

# from llama_index.node_parser import SentenceSplitter
# node_parser = SentenceSplitter()

# all_nodes = []
# for book_title in book_titles:
#     document = SimpleDirectoryReader(input_files=[book_title]).load_data()
#     nodes = node_parser.get_nodes_from_documents(document)
#     all_nodes.extend(nodes)

# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex(all_nodes, storage_context=storage_context, show_progress=True)

from llama_index import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# while True:
#     prompt = input()
#     response = query_engine.query(prompt)
#     print(response)

import streamlit as st

st.title("💬 MedTutor")
st.caption("🚀 Powered by Mistral LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = query_engine.query(prompt).response
    msg = str(response)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

# st.title('🦙 MedTutor')
# # Create a text input box for the user
# prompt = st.text_input('Input your prompt here')

# # If the user hits enter
# if prompt:
#     response = query_engine.query(prompt)
#     # ...and write it out to the screen
#     st.write(response)

#     # Display raw response object
#     with st.expander('Response Object'):
#         st.write(response)
#     # Display source text
#     with st.expander('Source Text'):
#         st.write(response.get_formatted_sources())