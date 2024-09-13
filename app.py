from dotenv import load_dotenv
import os
from scripts.preprocess import preprocess_and_save
from rag_components.query_handler import generate_response
from rag_components.model_handling import LLM

# Load environment variables once
load_dotenv()

# Environment variables are now globally available
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Call preprocessing function
folder_path = "./data"
patterns = ['See discussions, ', 'References', 'Subject Index', 'Weizenbaum, J.: Wer']
embedding_file = "embeddings.index"
bm25_file = "bm25.pkl"

# Run preprocessing (if needed)
index, bm25_model, tokenized_chunks, pages_and_chunks = preprocess_and_save(
    folder_path=folder_path, 
    patterns=patterns, 
    output_embedding_file=embedding_file, 
    output_bm25_file=bm25_file
)

# Define the LLM you want to use
selected_llm = LLM.GPT_4O_MINI  # or any other model from the LLM Enum

# Get user query
user_query = "What does Bach think of Lanier?"

# Generate response
response = generate_response(
    llm=selected_llm, 
    query=user_query,
    pages_and_chunks=pages_and_chunks,
    tokenized_chunks=tokenized_chunks,
    index=index,
    bm25_model=bm25_model,
    num_context_items=5,
    use_lexical_search=False,
    use_reranking=False,
    return_answer_only=True
)

print(response)