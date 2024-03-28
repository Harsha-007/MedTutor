import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.vector_stores import PGVectorStore
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import torch
from llama_index import set_global_service_context
from llama_index.storage.storage_context import StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding

import psycopg2

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large")
)

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    #model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=4000,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=8096,
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

connection_string = "postgresql://postgres:root123@localhost:5432"
db_name = "vector_db"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

    
from sqlalchemy import make_url
url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="test_index",
    embed_dim="1024",  # gte-large embedding dimension
)

documents = SimpleDirectoryReader("./test").load_data()

#Creating the index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context,storage_context=storage_context, show_progress=True
)
#Loading the existing index; Create another file/cell and check the below code
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# query_engine = index.as_query_engine()

# while True:
#     prompt = input()
#     response = query_engine.query(str(prompt))
#     print(response)