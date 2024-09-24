# Final app.py 
# Import necessary files
from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
from scripts.preprocess import preprocess_and_save
from rag_components.query_handler import generate_response, format_response
from rag_components.model_handling import LLM

app = Flask(__name__)

# Load environment variables once
load_dotenv()

# Environment variables are now globally available
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Initialize model options (same as in the Streamlit sidebar)
llm_options = {
    "OpenAI GPT-3.5 Turbo": LLM.GPT_35_TURBO,
    "OpenAI GPT-4o Mini": LLM.GPT_4O_MINI,
    "Google Gemma 7B": LLM.GEMMA_7B_IT,
    "Meta LLaMA 2 7B Chat": LLM.LLAMA_2_7B_CHAT,
    "Mistral 7B Instruct": LLM.MISTRAL_7B_INSTRUCT,
    "Local Llama-2 7B Server": LLM.LOCAL_LLM_SERVER,
    "Azure OpenAI Deployment": LLM.AZURE_OAI_DEPLOYMENT
}

# Default routes
@app.route("/")
def home():
    # Render the main page template
    return render_template("index.html", llm_options=llm_options)

@app.route("/get", methods=["POST"])
def get_bot_response():
    # Collect data from the form
    user_query = request.form["user_query"]
    selected_model_name = request.form["selected_model"]
    num_context_items = int(request.form["num_context_items"])
    use_lexical_search = "use_lexical_search" in request.form
    use_reranking = "use_reranking" in request.form

    # Model selection
    selected_llm = llm_options[selected_model_name]

    # Preprocessing (similar to the Streamlit app)
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

    # Generate response
    response = generate_response(
        llm=selected_llm, 
        query=user_query,
        pages_and_chunks=pages_and_chunks,
        tokenized_chunks=tokenized_chunks,
        index=index,
        bm25_model=bm25_model,
        num_context_items=num_context_items,
        use_lexical_search=use_lexical_search,
        use_reranking=use_reranking,
    )

    # Format response
    formatted_response = format_response(response, return_answer_only=False)

    # Ensure context_source and context_content are not None, assign empty lists if they are
    context_source = formatted_response.get('context_source', [])
    context_content = formatted_response.get('context_content', [])

    # Render the template with the generated response
    return render_template("response.html", 
                           answer=formatted_response['answer'], 
                           context_source=context_source,
                           context_content=context_content
    )

if __name__ == "__main__":
    app.run(debug=True)