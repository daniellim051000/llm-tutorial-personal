import getpass
import os

from langchain.globals import set_verbose
set_verbose(True)

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

from langchain_community.document_loaders import PyPDFLoader

file_path = "sematic-search-engine/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

# print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

length = len(all_splits)

if not os.environ.get("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureOpenAIEmbeddings

# Embeddings
# Must use embeddings AI model
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="text-embedding-3-small",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

# Vector Store
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

# Having instantiated our vector store, we can now index the documents.
ids = vector_store.add_documents(documents=all_splits)

# Usage or Output
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

# print(results[0])

# results = await vector_store.asimilarity_search("When was Nike incorporated?")

# print(results[0])

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
# print(f"Score: {score}\n")
# print(doc)

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
# print(results[0])


# Retrievers
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)