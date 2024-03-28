import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, SimpleKeywordTableIndex
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import torch
from llama_index import set_global_service_context
from llama_index.storage.storage_context import StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent

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

import os
import shutil

book_titles = []
path = "./test"
for entry in os.scandir(path):
    book_titles.append(str(entry.name))

med_docs = {}
for book_title in book_titles:
    med_docs[book_title] = SimpleDirectoryReader(
        input_files=[f"./test/{book_title}"]
    ).load_data()
    print(f"====== Sucessfully loaded the data from {book_title} ======")
    
from llama_index.node_parser import SentenceSplitter
node_parser = SentenceSplitter()

# Build agents dictionary
agents = {}
query_engines = {}

for idx, book_title in enumerate(book_titles):
    nodes = node_parser.get_nodes_from_documents(med_docs[book_title])

    folder = book_title[:-4]
    if not os.path.exists(f"./test/{folder}"):
        # build vector index
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.storage_context.persist(
            persist_dir=f"./test/{folder}"
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./test/{folder}"),
        )

    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=llm)

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for tasks related to specific aspects of"
                    f" {folder}."
                ),
            ),
        ),
    ]
    # build agent
    function_llm = llm
    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )
    agents[book_title] = agent
    query_engines[book_title] = vector_index.as_query_engine(similarity_top_k=3)

# define tool for each document agent
objects = []
for book_title in book_titles:
    summary = (
        f"This content contains Wikipedia articles about {book_title}. Use"
        f" this tool if you want to answer any questions about {book_title}.\n"
    )
    node = IndexNode(
        text=summary, index_id=book_title, obj=agents[book_title]
    )
    objects.append(node)

vector_index = VectorStoreIndex(nodes=objects)

# query_engine = vector_index.as_query_engine(similarity_top_k=3, verbose=True)
# response = query_engine.query("PROMPT")

# print(response)