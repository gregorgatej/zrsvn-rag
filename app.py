import os
import json
import gradio as gr
import psycopg2
import requests
import boto3
import openai
import csv
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Local imports for our custom modules
# ─────────────────────────────────────────────────────────────────────────────
from model_handling import embedding_model  # The BGE-M3 model instance (if needed)
from query_handler import (
    lexical_search_limited_scope, 
    semantic_search_limited_scope, 
    hybrid_search_limited_scope
)

# Load environment variables for DB password
load_dotenv()
POSTGIS_PASSWORD = os.getenv("POSTGIS_TEST1_PASSWORD")

# Load S3 credentials
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = "https://moja.shramba.arnes.si"
bucket_name = "zrsvn-monitoringi-raziskave"

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=s3_endpoint_url,
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_access_key
)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In practice, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Database Connection
# ─────────────────────────────────────────────────────────────────────────────
db_params = {
    "dbname": "postgis_test1",
    "user": "ggatej-pg",
    "password": POSTGIS_PASSWORD,
    "host": "localhost",
    "port": "5432"
}

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints for map usage
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/get_all_points/")
def get_all_points():
    """
    GET endpoint that retrieves all geometry points in 'najdbe' table 
    and streams them back as newline-delimited JSON.
    """
    query = """
        SELECT ST_AsGeoJSON(ST_Transform(wkb_geometry, 4326)) AS geom
        FROM najdbe;
    """
    cur.execute(query)

    def generate():
        for row in cur.fetchall():
            yield json.dumps({
                "type": "Point",
                "coordinates": json.loads(row[0])["coordinates"]
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/json")


@app.get("/get_filenames/")
def get_filenames(min_lat: float, max_lat: float, min_lon: float, max_lon: float):
    """
    GET endpoint that determines which files (in table 'files') fall within 
    the user-drawn bounding box. Then logs those filenames into 'selected_files_log'.
    If no selection is made (on startup) or if no files are found in the selection, 
    all files will be selected.
    """
    if None in (min_lat, max_lat, min_lon, max_lon):  # Case: App starts
        cur.execute("SELECT file_name FROM files;")
        results = cur.fetchall()
    else:
        query = """
            SELECT DISTINCT f.file_name 
            FROM files f
            JOIN najdbe n ON f.id = n.files_id
            WHERE ST_Within(
                ST_Transform(n.wkb_geometry, 4326), 
                ST_MakeEnvelope(%s, %s, %s, %s, 4326)
            );
        """
        cur.execute(query, (min_lon, min_lat, max_lon, max_lat))
        results = cur.fetchall()

        if not results:  # Case: No files in the selected region
            cur.execute("SELECT file_name FROM files;")
            results = cur.fetchall()
    
    filenames = []
    for row in results:
        filenames.append(row[0])

    cur.execute("DELETE FROM selected_files_log;")
    for fn in filenames:
        cur.execute("INSERT INTO selected_files_log (filename) VALUES (%s);", (fn,))
    conn.commit()

    return {"selected_filenames": filenames}

@app.get("/get_selected_filenames/")
def get_selected_filenames():
    """
    Return all filenames currently in 'selected_files_log'.
    """
    cur.execute("SELECT filename FROM selected_files_log;")
    rows = cur.fetchall()
    filenames = [row[0] for row in rows]
    return {"selected_filenames": filenames}

@app.post("/reset_selected_filenames/")
def reset_selected_filenames():
    """
    Clears all entries in `selected_files_log`.
    """
    try:
        cur.execute("DELETE FROM selected_files_log;")
        conn.commit()
        return {"message": "Selected files reset."}
    except Exception as e:
        return {"error": f"Failed to reset selected files: {str(e)}"}
    
@app.post("/reset_to_all_files/")
def reset_to_all_files():
    """
    Forcefully resets 'selected_files_log' so that all files from 'files'
    become selected. This runs whenever the page is refreshed.
    """
    # Clear old selections
    cur.execute("DELETE FROM selected_files_log;")
    # Insert all files
    cur.execute("INSERT INTO selected_files_log (filename) SELECT file_name FROM files;")
    conn.commit()

    return {"message": "Selection reset to all files."}

# ─────────────────────────────────────────────────────────────────────────────
# Mount templates directory for the Leaflet map
# ─────────────────────────────────────────────────────────────────────────────
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")

@app.get("/map")
def serve_map(request: Request):
    """
    Serves the Leaflet map from map.html.
    """
    return templates.TemplateResponse("map.html", {"request": request})

def generate_presigned_url(file_key, page_number):
    """
    Generates a presigned URL to access a specific file, appending a page anchor.
    """

    if file_key is None:
        return None  # No presigned URL for missing S3 link

    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_key},
            ExpiresIn=3600
        )
        return f"{presigned_url}#page={page_number}"
    except Exception as e:
        print(f"Error generating link: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK & Chatbot
# ─────────────────────────────────────────────────────────────────────────────
feedback_csv = "custom_log.csv"
if not os.path.exists(feedback_csv):
    with open(feedback_csv, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Chat History", "User Comment", "Index"])

global_chat_history = []

global_search_method = "Hybrid"

global_k_context_items = 5

def update_search_method(search_method):
    global global_search_method
    global_search_method=search_method

def update_context_k(k):
    global global_k_context_items
    global_k_context_items=k

# In place for possible future use if user feedback will problematise referals to chat history.
# def check_query(original_query):
#     system_prompt = (
#         "You are an advanced query checking assistant that checks if user queries "
#         "demand refering to the history of the chat or not. "
#         "If yes, return 'True', if no, return 'False'. Do this for the following user query:"
#     )
#     client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": original_query}
#         ],
#         temperature=0.3
#     )
#     return response.choices[0].message.content.strip()
    

#We leave query rewriting aside for now, as the used cgpt model is strong enough
#to manage typos etc., while domain specific rewriting seems unnecessary at this stage.
def rewrite_query(original_query):
    system_prompt = (
        "You are an advanced query rewriting assistant that refines user queries "
        "to improve clarity, retrieval effectiveness, and relevance while maintaining "
        "the original intent and language. Do this with the following user query:"
    )
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_query}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def add_context(query):
    """
    Retrieves relevant context from `run_search` and formats it into a prompt.
    
    Ensures:
    1. `run_search` returns the expected two values.
    2. Proper error handling in case of unexpected return values.
    """

    #We check if global_k_context_items is equal to 0. If yes,
    #we bypass running search_result and set results_md to "To display context items the 'Number of context items' should be set to bigger then zero."
    #and prompt_with_context to "ZERO_K"
    #If not, we continue with the rest of the code.

    if global_k_context_items == 0:
        return "To display context items the 'Number of context items' should be set to bigger then zero.", "ZERO_K"
    else:
    
        search_result = run_search(query, global_search_method, global_k_context_items)

        if not isinstance(search_result, tuple) or len(search_result) != 2:
            raise ValueError(f"Unexpected return value from run_search: {search_result}")

        results_md, results_list = search_result

        if not isinstance(results_list, list):
            raise ValueError(f"Expected list for results_list but got: {type(results_list)}")

        # if not results_list:
        #     return "No relevant context found."

        chunk_texts=[]
        for item in results_list:
            chunk_texts.append(item["chunk_text"])

        if chunk_texts:
            context = "- " + "\n- ".join(chunk_texts)
        else:
            context = "No relevant context items found."

        # query_references_chat_history = check_query(query)

        base_prompt = (
            "With your general knowledge and with the help of the following context items, please answer the query. "
            "Give yourself room to think by extracting relevant passages from the context before answering the query. "
            "Don't return the thinking, only return the answer. "
            "Make sure your answers are as explanatory as possible.\n\n"
            "Context items:\n"
            "{context}\n\n"
            "User query: {query}\n\n"
            "Answer:"
        )

        # In place for possible future use if user feedback will problematise referals to chat history.
        # if query_references_chat_history:
        #     base_prompt = (
        #         "With your general knowledge and with the help of the following context items, please answer the query. "
        #         "Give yourself room to think by extracting relevant passages from the context before answering the query. "
        #         "Don't return the thinking, only return the answer. "
        #         "Make sure your answers are as explanatory as possible.\n\n"

        #         "Context items:\n"
        #         "{context}\n\n"

        #         "User query: {query}\n\n"

        #         "Answer:"
        #     )
        # else:
        #     base_prompt = (
        #         "With your general knowledge and with the help of the following context items, please answer the query. "
        #         "Give yourself room to think by extracting relevant passages from the context before answering the query. "
        #         "Don't return the thinking, only return the answer. "
        #         "Make sure your answers are as explanatory as possible.\n\n"

        #         "Context items:\n"
        #         "{context}\n\n"

        #         "User query: {query}\n\n"

        #         "Answer:"
        #     )

        prompt_with_context = base_prompt.format(context=context, query=query)
        print(prompt_with_context)

        return results_md, prompt_with_context

def predict(message, history):
    global global_chat_history

    if history is None:
        history = []

    #rewritten_query = rewrite_query(message)

    results_md, query_with_context = add_context(message)

    # In place for possible future use if user feedback will problematise referals to chat history.
    # system_prompt_content = """You are a helpful AI assistant. Your goal is to provide helpful, accurate, and safe responses to user queries. You have access to an external retrieval system to enhance your responses with relevant documents.

    # - If retrieval results are available, prioritize incorporating relevant extracted information into your response.
    # - Clearly differentiate between retrieved knowledge and your internal knowledge. Cite sources when applicable.
    # - If retrieved documents do not contain relevant information, inform the user and fall back on your internal knowledge.
    # - Do not hallucinate sources or fabricate references. If uncertain, state so explicitly.
    # - If a user asks for real-time or external information, direct them to relevant sources or inform them that you do not have current data.
    # - When responding, consider not only the retrieved context but also the broader conversation history and any relevant details from the ongoing interaction.
    # - If the user’s query relates to prior messages in the conversation, incorporate relevant context from the chat history alongside retrieved information.
    # - If there is ambiguity in whether the user is referring to retrieved content, chat history, or broader context, seek clarification before responding.
    # """

    system_prompt_content = """You are a helpful AI assistant. Always respond in Slovenian unless the context clearly requires another language for better understanding or relevance."""
    system_prompt = {
        "role": "system",
        "content": system_prompt_content
    }
    messages = [system_prompt]  # Start with system prompt
    # messages = []

    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})

    if query_with_context != "ZERO_K":
        messages.append({"role": "user", "content": query_with_context})
    else:
        messages.append({"role": "user", "content": message})

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    chunks = []

    # Log user input first
    global_chat_history.append({
        "role": "user",
        "content": message,
        "original_content": message
    })

    # Append an empty assistant response, to be filled incrementally
    global_chat_history.append({
        "role": "assistant", 
        "content": ""
    })

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)

        # Update last assistant message incrementally
        global_chat_history[-1]["content"] += delta

        full_response = "".join(chunks)
        yield full_response, results_md
        time.sleep(0.025)

    # # Save user input and then assistant input to global_chat_history
    # global_chat_history.append({
    #     "role": "user",
    #     "content": message,
    #     "original_content": message
    # })

    # global_chat_history.append({
    #     "role": "assistant",
    #     "content": "".join(chunks)
    # })

def handle_feedback(comment):
    global global_chat_history

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    chat_history_cleaned = []
    for entry in global_chat_history:
        newdict = {"role": entry["role"], "content": entry["content"]}
        if "original_content" in entry:
            newdict["original_content"] = entry["original_content"]
        chat_history_cleaned.append(newdict)

    chat_history_json = json.dumps(chat_history_cleaned, ensure_ascii=False)
    last_index = max(0, len(global_chat_history) - 1)

    with open(feedback_csv, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, chat_history_json, comment, last_index])

    return "", gr.update(placeholder="Povratna informacija uspešno zabeležena. Lahko vnesete novo mnenje ali komentar...")


# ─────────────────────────────────────────────────────────────────────────────
# Searching logic
# ─────────────────────────────────────────────────────────────────────────────
def run_search(query_text, search_method, k_results):
    """
    Return a tuple with:
    1. A Markdown string containing clickable links
    2. A list of result dictionaries

    Ensures that the function always returns exactly two values to prevent unpacking errors.
    """
    if not query_text or not search_method:
        return "Please enter a query and select a search method.", []

    try:
        k = int(k_results)
    except ValueError:
        k = 5

    if search_method == "Lexical":
        results = lexical_search_limited_scope(query_text, k=k, db_params=db_params)
    elif search_method == "Semantic":
        results = semantic_search_limited_scope(query_text, k=k, db_params=db_params)
    else:
        results = hybrid_search_limited_scope(query_text, k=k, db_params=db_params)

    if not results:
        # return "No results found.", []
        return "No context items could be displayed. Please refine your question.", []

    answers = []
    results_list = []
    file_nr = 1

        # For lexical/semantic: chunk_id, chunk_text, file_name, page_number, score
        # For hybrid: chunk_id, score, chunk_text, file_name, page_number
    for row in results:
        # Ensure row has the correct length before unpacking
        if search_method in ("Lexical", "Semantic"):
            if len(row) != 6:
                continue  # Skip invalid results
            chunk_id, chunk_text, file_name, s3_link, page_number, score = row
        else:
            if len(row) != 6:
                continue  # Skip invalid results
            chunk_id, score, chunk_text, file_name, s3_link, page_number = row

        presigned_url = generate_presigned_url(s3_link, page_number)

        if presigned_url:
            # Markdown with a clickable link:
            snippet = f"""File nr. {file_nr}: [{file_name} (page {page_number})]({presigned_url})  
            Score: {score:.4f}  
            """
        else:
            snippet = f"""File nr. {file_nr}: {file_name} (page {page_number})  
            Score: {score:.4f}  
            """  # No hyperlink if S3 link is missing

        answers.append(snippet)

        result_dict = {
            "file_number": file_nr,
            "file_name": file_name,
            "s3_link": s3_link,
            "page_number": page_number,
            "chunk_text": chunk_text,
            "presigned_url": presigned_url if presigned_url else "",
            "score": round(score, 4),
            "snippet": snippet
        }

        results_list.append(result_dict)

        file_nr += 1

    markdown_answers = "\n".join(answers)

    # Join all snippet lines into Markdown
    return markdown_answers, results_list


def fetch_selected_docs():
    """
    Returns just a count of the selected filenames, not the full list.
    """
    try:
        response = requests.get("http://localhost:8000/get_selected_filenames/")
        data = response.json()
        filenames = data.get("selected_filenames", [])
        nr_docs = len(filenames)
        return f"Currently selected docs: {nr_docs}"

    except Exception as e:
        return f"Error retrieving selected docs: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Build the Gradio interface
# ─────────────────────────────────────────────────────────────────────────────
def build_gradio_interface():
    with gr.Blocks(title="ZRSVN RAG") as demo:

        gr.Markdown('<div style="text-align: center; font-size: 24px; font-weight: bold;">Chatbot z možnostjo povratne informacije</div>')

        # We define search_results beforehand so that the md formatted additional output from ChatInterface/predict can be passed to it,
        # but we do not yet render it.
        search_results_md = gr.Markdown("Context items will appear here...", label="Search Results", render=False)

        chat = gr.ChatInterface(
            chatbot=gr.Chatbot(placeholder="<strong>Kaj vas tokrat zanima o monitoringih?</strong><br>Vprašanje vnesite v spodnjo vrstico."),
            fn=predict,
            type="messages",
            theme="ocean",
            flagging_mode="manual",
            flagging_dir="/mnt/partit1/projects/zrsvn-rag",
            additional_outputs=[search_results_md]
        )

        with gr.Accordion(label="✉️ Send feedback", open=False):
            feedback_box = gr.Textbox(
            label="Povratna informacija", 
            placeholder="Vnesite vaše mnenje ali komentar..."
            )
            feedback_button = gr.Button("Pošlji povratno informacijo")
            feedback_button.click(handle_feedback, inputs=[feedback_box], outputs=[feedback_box, feedback_box])

        with gr.Accordion(label="🌍 Geographically determine scope of documents that are searched over", open=False):
            # The map iframe
            # Geographically determine scope of documents that are searched over
            gr.HTML("""
                <style>
                    .map-container {
                        display: flex;
                        flex-direction: column;
                        height: 70vh; 
                        border: none; 
                        border-radius: 5px; 
                        box-sizing: border-box;
                        padding: 0; 
                        background-color: white; 
                    }
                    .map-container iframe {
                        flex: 1;
                        width: 100%;
                        display: block;
                        margin: 0;
                        padding: 0;
                        border: none;
                        overflow: hidden;
                    }
                </style>
                <div class="map-container">
                    <iframe src="http://127.0.0.1:8000/map" scrolling="no"></iframe>
                </div>
            """)
            # Button to update the count
            show_docs_button = gr.Button("Click to see updated number of currently selected docs")
            # Markdown for displaying the doc count
            docs_text = gr.Markdown()
            show_docs_button.click(fetch_selected_docs, inputs=[], outputs=docs_text)

            # --- Search section ---

        
        with gr.Accordion(label="⚙️ Context settings", open=False):
            with gr.Row():
                search_method = gr.Radio(
                    choices=["Lexical", "Semantic", "Hybrid"],
                    value=global_search_method,
                    label="Search Method",
                    interactive=True
                )
                k_slider = gr.Slider(
                    minimum=0,
                    maximum=15,
                    value=global_k_context_items,
                    step=1,
                    label="Number of Context Items",
                    interactive=True
                )
                search_method.change(fn=update_search_method, inputs=[search_method], outputs=[])
                k_slider.change(fn=update_context_k, inputs=[k_slider], outputs=[])

        # We render the md formatted result.
        with gr.Accordion(label="📝 View context items", open=False):
            search_results_md.render()
        

        with gr.Accordion("📖 Navodila za uporabo", open=False):
            gr.Markdown("""
                By default, all documents are included in the search.
                By drawing a rectangle on the map, you can narrow down
                the search to only documents connected to that area.
            """)

        # Connect search button -> run_search -> search_results_md
        # search_btn.click(
        #     fn=lambda query, method, k: run_search(query, method, k)[0],  # Extract only the first return value
        #     inputs=[query_box, search_method, k_slider],
        #     outputs=search_results_md
        # )

    return demo


if __name__ == "__main__":
    interface = build_gradio_interface()
    interface.launch(favicon_path="./assets/zrsvn_logo.png", server_name="0.0.0.0", server_port=7950, share=False)
