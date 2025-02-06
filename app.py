import os, csv
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from scripts.preprocess import preprocess_and_save
from rag_components.query_handler import generate_response, format_response
from rag_components.model_handling import LLM

# Load environment variables
load_dotenv()

# Environment variables are now globally available
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Define LLM options
llm_options = {
    "OpenAI GPT-3.5 Turbo": LLM.GPT_35_TURBO,
    "OpenAI GPT-4o Mini": LLM.GPT_4O_MINI,
    "Google Gemma 7B": LLM.GEMMA_7B_IT,
    "Meta LLaMA 2 7B Chat": LLM.LLAMA_2_7B_CHAT,
    "Mistral 7B Instruct": LLM.MISTRAL_7B_INSTRUCT,
    "Local Llama-2 7B Server": LLM.LOCAL_LLM_SERVER,
    "Azure OpenAI Deployment": LLM.AZURE_OAI_DEPLOYMENT
}

# The function to handle LLM selection and response generation
def handle_query(user_query, selected_model_name, num_context_items, use_lexical_search, use_reranking):
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
        num_context_items=int(num_context_items),
        use_lexical_search=use_lexical_search,
        use_reranking=use_reranking,
    )

    # Format response
    formatted_response = format_response(response, return_answer_only=False)

    answer = formatted_response['answer']
    
    # Construct context information
    context_info = ""
    if num_context_items > 0 and 'context_source' in formatted_response and 'context_content' in formatted_response:
        for i, (source, content) in enumerate(zip(formatted_response['context_source'], formatted_response['context_content']), start=1):
            context_info += f"Vir {i}: {source}  \nBesedilo: {content}  \n\n"

    return answer, context_info

# Define the CSV file path
feedback_csv = "feedback_data.csv"

# Function to handle user feedback and save it to CSV
def handle_feedback(model_selector, num_context_items, use_lexical_search, use_reranking, feedback, comment, question, answer, context_output):
    # Save feedback in a CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(feedback_csv):
        # If the file doesn't exist, create it and add the headers
        with open(feedback_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Model", "num_context_items", "use_lexical_search", "use_reranking", "Question", "Answer", "Context", "Feedback", "Comment"])

    # Append the feedback information to the CSV file
    with open(feedback_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, model_selector, num_context_items, use_lexical_search, use_reranking, question, answer, context_output, feedback, comment])
    
    return "Povratna informacija uspešno zabeležena."

# Gradio Blocks implementation with vertical alignment
def build_interface():
    with gr.Blocks(title="ZRSVN RAG") as demo:
        gr.Markdown("# ZRSVN RAG aplikacija za odgovarjanje na vprašanja")
        gr.Markdown("Vnesite svoje vprašanje spodaj in prejmite odgovor, ki ga ustvari izbrani veliki jezikovni model (LLM).")
        
        with gr.Column(elem_id="main_column", scale=1):
            question = gr.Textbox(label="Vaše vprašanje", lines=2, placeholder="Vnesite vaše vprašanje...")
            model_selector = gr.Dropdown(choices=list(llm_options.keys()), label="Izberite veliki jezikovni model", value="OpenAI GPT-4o Mini")
            num_context_items = gr.Slider(minimum=0, maximum=10, step=1, value=5, label="Število dokumentov zajetih v kontekst")
            
            # Additional Options as an Accordion
            with gr.Accordion("Dodatne možnosti iskanja", open=False):
                use_lexical_search = gr.Checkbox(label="Uporabi klasično (leksikalno) iskanje")
                use_reranking = gr.Checkbox(label="Uporabi naknadno rangiranje virov konteksta")
            
            submit_button = gr.Button("Generiraj odgovor")
                
            answer_output = gr.Textbox(label="Odgovor")
            context_output = gr.Textbox(label="Kontekst")

            # Add feedback mechanism with thumbs up/down
            feedback_buttons = gr.Radio(choices=["👍", "👎"], label="Se vam je zdel odgovor uporaben?")
            comment_box = gr.Textbox(label="Napišite komentar (opcijsko)", placeholder="Vaš komentar vnesite tu...")
            feedback_button = gr.Button("Pošlji povratno informacijo")

            feedback_output = gr.Markdown()

        # Define the button's functionality
        submit_button.click(
            fn=handle_query, 
            inputs=[question, model_selector, num_context_items, use_lexical_search, use_reranking], 
            outputs=[answer_output, context_output]
        ).then(
            fn=lambda: (gr.update(value=None), gr.update(value=""), gr.update(value="")),
            inputs=[],
            outputs=[feedback_buttons, comment_box, feedback_output]
        )


        feedback_button.click(
            fn=handle_feedback, 
            inputs=[model_selector, num_context_items, use_lexical_search, use_reranking, feedback_buttons, comment_box, question, answer_output, context_output], 
            outputs=feedback_output
        )

    return demo

# Launch the Gradio app
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(favicon_path="./assets/zrsvn_logo.png", share=True)
