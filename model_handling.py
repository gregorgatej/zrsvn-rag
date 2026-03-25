import os
from FlagEmbedding import FlagAutoModel

embedding_model = FlagAutoModel.from_finetuned(
    "BAAI/bge-m3",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=False,
    device=["cuda:0"]
)