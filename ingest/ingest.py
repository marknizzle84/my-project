import os
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

DOCS_PATH = "/data/docs"

def load_docs():
    docs = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_PATH, file))
            docs.extend(loader.load())
    return docs

def main():
    docs = load_docs()
    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="/chroma"
    )

    db.persist()

if __name__ == "__main__":
    main()