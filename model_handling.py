import os
# Uvozimo razred za nalaganje in uporabo BGE-M3 vložitvenega modela.
from FlagEmbedding import FlagAutoModel

# ─────────────────────────────────────────────────────────────────────────────
# Naložimo fino nastavljeni BGE-M3 model, ki ga bomo uporabljali
# za semantično in hibridno iskanje.
# - "query_instruction_for_retrieval" pomaga modelu razumeti, da naj vektor
#   oz. vložitev predstavlja stavek za iskanje relevantnih odlomkov.
# - use_fp16=False pomeni, da namesto 16-bitne (polovične) natančnosti 
#   za numerično reprezentacijo uporabljamo
#   32-bitno realno število z enojno natančnostjo. To pomeni, da je rezultat kvalitetnejši,
#   čeprav na račun večje spominske požrešnosti, a ne tolikšne, da bi bil to problem.
# - device=["cuda:0"] določi, da se model izvaja na prvi (in edini) izmed GPU naprav.
embedding_model = FlagAutoModel.from_finetuned(
    "BAAI/bge-m3",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=False,
    device=["cuda:0"]
)