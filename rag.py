"""Retrieval for RAG: BM25 over data/rag/corpus.jsonl.

Pure Python with no extra dependencies, so it runs the same on the laptop,
Colab, or a CPU box. BM25 suits this corpus because questions name exact
identifiers (T1059.001, CVE-2023-4966, AS-REP) that keyword matching finds
reliably. Build the corpus first with scripts/build_corpus.py.

    from rag import Retriever
    r = Retriever()
    for p in r.search("How do I detect AS-REP roasting?", k=4):
        print(p["title"])
"""
import json
import math
import re
from collections import Counter, defaultdict

CORPUS = "data/rag/corpus.jsonl"
TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOP = set("""a an and are as at be by can do does for from how i in is it its of on
or that the this to was what when which who why with you your would should about
into than then there these they their them use used using via vs""".split())


def tokenize(text):
    out = []
    for t in TOKEN.findall(text.lower()):
        if t in STOP:
            continue
        out.append(t)
        if "-" in t or "." in t:          # also index the parts: cve-2023-4966 -> 4966
            out.extend(p for p in re.split(r"[.\-]", t) if p and p not in STOP)
    return out


class Retriever:
    def __init__(self, path=CORPUS, k1=1.5, b=0.75):
        self.docs = [json.loads(l) for l in open(path) if l.strip()]
        self.k1, self.b = k1, b
        self.tf, self.len = [], []
        self.post = defaultdict(list)
        for i, d in enumerate(self.docs):
            # weight the title by repeating it; titles carry the IDs and names
            c = Counter(tokenize(d["title"] + " " + d["title"] + " " + d["text"]))
            self.tf.append(c)
            self.len.append(sum(c.values()))
            for t in c:
                self.post[t].append(i)
        self.avg = sum(self.len) / max(len(self.len), 1)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.post.items()}

    def search(self, query, k=4, sources=None):
        scores = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += idf * f * (self.k1 + 1) / norm
        ranked = sorted(scores, key=scores.get, reverse=True)
        out = []
        for i in ranked:
            if sources and self.docs[i]["source"] not in sources:
                continue
            out.append({**self.docs[i], "score": round(scores[i], 2)})
            if len(out) == k:
                break
        return out


def context_block(passages, max_chars=1400):
    """Format retrieved passages for the prompt."""
    lines = []
    for n, p in enumerate(passages, 1):
        lines.append(f"[{n}] {p['title']}\n{p['text'][:max_chars]}")
    return "\n\n".join(lines)


def rag_question(question, passages):
    """The user turn the model sees when RAG is on."""
    return ("Reference material retrieved for this question. It may be incomplete "
            "or partly off-topic: use what is relevant and accurate, and rely on your "
            "own knowledge for the rest. Do not mention the reference numbers.\n\n"
            f"{context_block(passages)}\n\n"
            f"Question: {question}")
