# utils.py
from typing import List
from langchain.schema import Document

def format_docs(docs: List[Document]) -> str:
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source></document>"
            for doc in docs
        ]
    )
