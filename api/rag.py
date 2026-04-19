from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

def get_chain():
    embeddings = OpenAIEmbeddings()
    
    db = Chroma(
        persist_directory="/chroma",
        embedding_function=embeddings
    )

    retriever = db.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini")

    def query(q):
        docs = retriever.get_relevant_documents(q)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {q}
        """

        return llm.predict(prompt)

    return query