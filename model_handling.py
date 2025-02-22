import os
from FlagEmbedding import FlagAutoModel

# ─────────────────────────────────────────────────────────────────────────────
# Load the BGE-M3 model. This model is used for all semantic/hybrid queries.
# Using the snippet you provided for "FlagAutoModel.from_finetuned".
# ─────────────────────────────────────────────────────────────────────────────

# You can set query instruction or pass other options to the model as you see fit.
embedding_model = FlagAutoModel.from_finetuned(
    "BAAI/bge-m3",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=False,  # Adjust based on memory or your preference
    device=["cuda:0"]  # or "cpu", or e.g. ["cuda:0"] if you want GPU
)

# ─────────────────────────────────────────────────────────────────────────────
# What Was Removed or Changed?
# ─────────────────────────────────────────────────────────────────────────────
# 1) Removed references to SentenceTransformer and older "init_embedding_model".
# 2) Removed usage of openai or huggingface-based GPT calls. 
#    This file strictly loads the BGE-M3 model via FlagAutoModel.
# 3) Minimizes code to just define 'embedding_model' as a singleton to be used 
#    across the app.
