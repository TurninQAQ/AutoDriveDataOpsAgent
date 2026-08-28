from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from deploy_ci_cloud_agentv3.platform_backend.rag.sources import KnowledgeSourceConfig, KnowledgeSourceLoader
from deploy_ci_cloud_agentv3.platform_backend.rag.text import tokenize
from deploy_ci_cloud_agentv3.rag.embeddings import EmbeddingProvider
from deploy_ci_cloud_agentv3.rag.index import DenseIndex


class RAGService:
    def __init__(self, source_dir: Path, index_dir: Path, *, mode: str = "hybrid", embedding_provider: EmbeddingProvider | None = None, rrf_k: int = 60):
        self.source_dir = Path(source_dir)
        self.index = DenseIndex(Path(index_dir))
        self.requested_mode = mode.lower()
        self.embedding_provider = embedding_provider
        self.rrf_k = int(rrf_k)
        self._chunks = KnowledgeSourceLoader(KnowledgeSourceConfig(self.source_dir)).load()

    @property
    def effective_mode(self) -> str:
        if self.requested_mode == "bm25": return "bm25"
        if self.embedding_provider is None: return "bm25"
        if not self.index.is_fresh(self.source_dir, model=self.embedding_provider.model_name, dimension=self.embedding_provider.dimension): return "bm25"
        return self.requested_mode if self.requested_mode in {"dense", "hybrid"} else "hybrid"

    @staticmethod
    def _search_text(chunk) -> str:
        return f"{chunk.title} {chunk.section} {chunk.content}"

    def _bm25(self, query: str) -> list[tuple[int, float]]:
        docs = [tokenize(self._search_text(c), expand=False) for c in self._chunks]
        q = tokenize(query, expand=True)
        if not q or not docs: return []
        n = len(docs); avgdl = sum(map(len, docs)) / max(1, n); k1=1.5; b=0.75
        df = Counter()
        for tokens in docs:
            for token in set(tokens): df[token]+=1
        scores=[]
        for i,tokens in enumerate(docs):
            tf=Counter(tokens); score=0.0
            for token in q:
                freq=tf.get(token,0)
                if not freq: continue
                idf=math.log(1+(n-df.get(token,0)+0.5)/(df.get(token,0)+0.5))
                score += idf*(freq*(k1+1))/(freq+k1*(1-b+b*len(tokens)/max(avgdl,1)))
            scores.append((i,score))
        return sorted(scores,key=lambda x:(-x[1],x[0]))

    async def _dense(self, query: str) -> list[tuple[int,float]]:
        if self.embedding_provider is None: return []
        chunks,matrix,_=self.index.load()
        # Dense index order is authoritative. Reuse those chunks for aligned ranking.
        self._chunks = chunks
        q=np.asarray(await self.embedding_provider.embed_query(query),dtype=np.float32)
        if q.shape != (matrix.shape[1],): raise RuntimeError("query embedding dimension mismatch")
        scores=matrix @ q
        return sorted([(i,float(score)) for i,score in enumerate(scores)],key=lambda x:(-x[1],x[0]))

    async def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        query=query.strip()
        if not query: raise ValueError("query must not be empty")
        mode=self.effective_mode
        bm25=self._bm25(query)
        dense=await self._dense(query) if mode in {"dense","hybrid"} else []
        bm_rank={i:r for r,(i,_) in enumerate(bm25,1)}; bm_score=dict(bm25)
        de_rank={i:r for r,(i,_) in enumerate(dense,1)}; de_score=dict(dense)
        candidates=set(bm_rank) if mode=="bm25" else set(de_rank) if mode=="dense" else set(bm_rank)|set(de_rank)
        rows=[]
        for i in candidates:
            fusion=(1/(self.rrf_k+bm_rank[i]) if i in bm_rank else 0.0)+(1/(self.rrf_k+de_rank[i]) if i in de_rank else 0.0)
            rank_score=bm_score.get(i,0.0) if mode=="bm25" else de_score.get(i,0.0) if mode=="dense" else fusion
            c=self._chunks[i]
            rows.append({"chunk_id":c.chunk_id,"source":c.source_path,"title":c.title,"section":c.section,"text":c.content,"bm25_score":bm_score.get(i),"bm25_rank":bm_rank.get(i),"dense_score":de_score.get(i),"dense_rank":de_rank.get(i),"fusion_score":fusion if mode=="hybrid" else None,"rank_score":rank_score})
        rows.sort(key=lambda r:(-float(r["rank_score"]),r["chunk_id"]))
        for rank,row in enumerate(rows[:top_k],1): row["rank"]=rank
        return {"query":query,"mode":mode,"requested_mode":self.requested_mode,"results":rows[:top_k]}
