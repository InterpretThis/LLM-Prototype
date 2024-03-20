"""
A demonstration of retrieval-augmented generation with LangChain.
"""

from time import time

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamafile import Llamafile
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter


def load_pdf(file_path: str):
    """
    Load a PDF and split it into pages.
    """

    return PyPDFLoader(file_path).load_and_split()


DIR_CAMPAIGNS = "pdf/campaigns"
DIR_CHARACTERS = "pdf/characters"
DIR_GUIDES = "pdf/guides"

FILE_CAMPAIGN = "A Thousand Tiny Deaths"
FILE_CHARACTER = "Dragonborn Sorcerer 1"
FILE_GUIDE = "DnD Adventurers League Player_s Guide v13.0"


def load_documents():
    """
    Load the campaign, character sheet, and player's guide PDFs.
    """

    print("Loading PDFs...")

    campaigns = load_pdf(f"{DIR_CAMPAIGNS}/{FILE_CAMPAIGN}.pdf")
    characters = load_pdf(f"{DIR_CHARACTERS}/{FILE_CHARACTER}.pdf")
    # guides = load_pdf(f"{DIR_GUIDES}/{FILE_GUIDE}.pdf")

    return campaigns + characters


def create_vectorstore():
    """
    Create a vector-store from the PDF documents.
    """

    print("Creating vector-store...")

    return FAISS.from_documents(
        CharacterTextSplitter(chunk_size=1024, chunk_overlap=256).split_documents(
            load_documents()
        ),
        HuggingFaceEmbeddings(),
    )


def format_docs(docs: list[Document]):
    """
    Format the vector-store documents as a string.
    """

    return "\n\n".join(doc.page_content for doc in docs)


def setup():
    """
    Instantiate the retrieval-augmented generation (RAG) chain.
    """

    vectorstore = create_vectorstore()

    print("Creating retriever...")
    retriever = vectorstore.as_retriever(search_type="similarity")

    print("Creating LLM...")
    llm = Llamafile()

    prompt_template = ChatPromptTemplate.from_template(
        """You are an assistant for a Dungeons and Dragons (DnD) game.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.

    {context}

    Question: {question}

    Helpful Answer:"""
    )

    print("Creating RAG chain...")
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)


if __name__ == "__main__":
    rag_chain = setup()

    while True:
        prompt = input("Prompt: ")

        start = time()
        response = rag_chain.invoke(prompt)

        context: list[Document] = response["context"]
        metadata = [document.metadata for document in context]

        print("Context:")
        for meta in metadata:
            print(f"    {meta}")
        print()

        print(f"Answer: {response['answer']}\n")

        print(f"Time: {time() - start:2f}s\n")
