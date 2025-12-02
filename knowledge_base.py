# realtime_sam/knowledge_base.py
# 一个超轻量级的「外挂知识库」，在发给模型前做简单检索。
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Document:
    id: str
    title: str
    content: str
    meta: dict = field(default_factory=dict)


class KnowledgeBase:
    """极简内存知识库，按文档级别存储，在内存中做检索。"""

    def __init__(self) -> None:
        self.docs: List[Document] = []

    def add_document(self, doc: Document) -> None:
        self.docs.append(doc)

    def add_raw(
        self,
        content: str,
        title: Optional[str] = None,
        doc_id: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> None:
        """直接通过原始字符串添加一篇文档。"""
        if meta is None:
            meta = {}
        doc_id = doc_id or str(len(self.docs) + 1)
        title = title or f"Doc {len(self.docs) + 1}"
        self.add_document(Document(id=doc_id, title=title, content=content, meta=meta))

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        极简检索：
        - 把 query 和 doc 做分词；
        - 按 token 重叠数打分；
        - 返回得分最高的前 top_k 篇文档。
        """
        query = (query or "").strip()
        if not query or not self.docs:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        q_set = set(q_tokens)
        scored = []
        for doc in self.docs:
            doc_tokens = self._tokenize(doc.title + " " + doc.content)
            doc_set = set(doc_tokens)
            score = len(q_set & doc_set)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def _tokenize(self, text: str) -> List[str]:
        """
        非常粗糙的分词：
        - 英文与数字：按连续字母数字切词；
        - 中文及其他：每个非空白字符单独作为一个 token。
        """
        text = (text or "").lower()
        tokens: List[str] = []
        buf = ""
        for ch in text:
            if ch.isalnum():
                buf += ch
            else:
                if buf:
                    tokens.append(buf)
                    buf = ""
                if not ch.isspace():
                    tokens.append(ch)
        if buf:
            tokens.append(buf)
        return tokens
