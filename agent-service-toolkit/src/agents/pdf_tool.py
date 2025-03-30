# pdf_tool.py
from typing import List
from langchain.schema import Document
from langchain_core.tools import tool
from .util import format_docs
from .pdf_loader import retriever
@tool
def retrieve_from_pdf(query: str) -> str:
    """
    주어진 쿼리에 대해 PDF 문서에서 관련 정보를 검색하고,
    포맷팅된 결과를 반환합니다.
    """
    docs = retriever.invoke(query)  # retriever는 외부에서 정의되어 있다고 가정합니다.
    formatted_docs = format_docs(docs)
    return formatted_docs

tools = [retrieve_from_pdf]

@tool
def retrieve_from_web(query: str) -> str:
    """
    Search and return blogs on langgraph multiturn and parallel node execution. Run this function if you need to search for langgraph.
    """
    docs = retriever.invoke(query)
    formatted_docs = format_docs(docs)
    return formatted_docs

tools = [retrieve_from_web]
