"""
Flask server for retrieval-augmented generation with LangChain.
"""

from subprocess import Popen
from sys import executable, stderr, stdout
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.llamafile import Llamafile
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langserve import add_routes

ModelType = Literal["llama-cpp", "llamafile"]


def create_llm(
    model_type: ModelType,
    callback_manager: CallbackManager | None = None,
):
    """
    Create a language model.
    """

    if model_type == "llama-cpp":
        return LlamaCpp(  # type: ignore
            model_path="gguf/mistral-7b-instruct-v0.1.Q2_K.gguf",
            n_ctx=4096,
            n_gpu_layers=1,
            streaming=True,
            callback_manager=callback_manager,
            verbose=True,
        )

    if model_type == "llamafile":
        name = "llamafile/llava-v1.5-7b-q4.llamafile"
        # name = "llamafile/mistral-7b-instruct-v0.2.Q5_K_M.llamafile"
        # name = "llamafile/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile"

        Popen(
            executable=executable,
            args=f"{name} --server --nobrowser",
            stdout=stdout,
            stderr=stderr,
        )

        return Llamafile(
            name=name,
            streaming=True,
            callback_manager=callback_manager,
            verbose=True,
        )

    raise ValueError(f"Invalid model type: {model_type}")


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
        CharacterTextSplitter(chunk_size=512, chunk_overlap=128).split_documents(
            load_documents()
        ),
        HuggingFaceEmbeddings(),
    )


def format_docs(docs: list[Document]):
    """
    Format the vector-store documents as a string.
    """

    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    """
    Create the retrieval-augmented generation (RAG) chain.
    """

    vectorstore = create_vectorstore()

    print("Creating retriever...")
    retriever = vectorstore.as_retriever(search_type="similarity")

    print("Creating LLM callback manager...")
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    print("Creating LLM...")
    llm = create_llm("llamafile", callback_manager)

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


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

chain = create_rag_chain()

add_routes(app, chain, path="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7000)
