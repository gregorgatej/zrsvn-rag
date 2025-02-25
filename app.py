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
# (Optional) from query_handler import rewrite_query, handle_feedback  # if you move them out

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
    allow_origins=["*"],   # In practice, restrict to your front-end domain
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

# Create one global connection and cursor
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
    GET endpoint that determines which files (in table 'files') fall within the 
    user-drawn bounding box (given by min_lat, max_lat, min_lon, max_lon).
    Then logs those filenames into 'selected_files_log' so that future searches 
    will only consider these files.
    """
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
    
    filenames = []
    for row in results:
        filenames.append(row[0])

    # Clear old log results and insert the new set
    cur.execute("DELETE FROM selected_files_log;")
    for i in range(len(filenames)):
        cur.execute("INSERT INTO selected_files_log (filename) VALUES (%s);", (filenames[i],))
    conn.commit()

    return {"selected_filenames": filenames}


@app.get("/get_selected_filenames/")
def get_selected_filenames():
    """
    Return all filenames currently in 'selected_files_log'.
    This is used by the Gradio UI to see which files were selected on the map.
    """
    cur.execute("SELECT filename FROM selected_files_log;")
    rows = cur.fetchall()
    filenames = []
    for row in rows:
        filenames.append(row[0])
    return {"selected_filenames": filenames}


@app.post("/reset_selected_filenames/")
def reset_selected_filenames():
    """
    Clears all entries in the `selected_files_log` table, effectively resetting the selected document state.
    """
    try:
        cur.execute("DELETE FROM selected_files_log;")
        conn.commit()
        return {"message": "Selected files reset."}
    except Exception as e:
        return {"error": f"Failed to reset selected files: {str(e)}"}


# ─────────────────────────────────────────────────────────────────────────────
# Mount "templates" directory for serving the static map
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
    Generates a presigned URL for accessing a specific file and appends a page anchor.
    """
    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_key},
            ExpiresIn=3600  # URL valid for 1 hour
        )
        return f"{presigned_url}#page={page_number}"
    except Exception as e:
        return f"Error generating link: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# REPLACE THE OLD CHATBOT IMPLEMENTATION WITH A STREAMING VERSION
# ─────────────────────────────────────────────────────────────────────────────

# For logging feedback
feedback_csv = "custom_log.csv"
if not os.path.exists(feedback_csv):
    with open(feedback_csv, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Chat History", "User Comment", "Index"])

# Global variable to store chat history
global_chat_history = []

def rewrite_query(original_query):
    """
    Modify this function to implement your query rewriting logic.
    For example: rewriting to Slovene, etc.
    """
    system_prompt = "Rewrite the following user query to Slovene:"

    # Ensure your openai.api_key is set
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

def predict(message, history):
    """
    Generate chatbot response in streaming mode, rewriting query first, 
    then calling GPT-4o-mini with streaming on.
    """
    global global_chat_history

    if history is None:
        history = []

    # Rewrite the query
    rewritten_message = rewrite_query(message)

    # Convert existing history to API-ready format
    messages = []
    for i in range(len(history)):
        msg_role = history[i]["role"]
        msg_content = history[i]["content"]
        # We won't pass metadata/options to openai if not needed
        messages.append({"role": msg_role, "content": msg_content})

    # Append the rewritten user message
    messages.append({"role": "user", "content": rewritten_message})

    # Set the key again if needed
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    # We'll collect chunks in a list and yield partial output
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)
        full_response = "".join(chunks)
        yield full_response
        # small delay so that the user sees the streaming effect
        time.sleep(0.05)

    # Store final messages in global_chat_history
    # Store the *original* user message too:
    global_chat_history.append({
        "role": "user",
        "content": rewritten_message,
        "original_content": message
    })
    global_chat_history.append({
        "role": "assistant",
        "content": "".join(chunks)
    })

def handle_feedback(comment):
    """
    Handle user feedback after a chat session. Saves to CSV.
    """
    global global_chat_history

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    # Prepare a cleaned version of the chat history for CSV
    chat_history_cleaned = []
    for i in range(len(global_chat_history)):
        entry = global_chat_history[i]
        newdict = {
            "role": entry["role"],
            "content": entry["content"]
        }
        if "original_content" in entry:
            newdict["original_content"] = entry["original_content"]
        chat_history_cleaned.append(newdict)

    # Convert to JSON
    chat_history_json = json.dumps(chat_history_cleaned, ensure_ascii=False)

    # Index is just the last message index
    last_index = max(0, len(global_chat_history) - 1)

    # Append to CSV
    with open(feedback_csv, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, chat_history_json, comment, last_index])

    # Reset or update the placeholder
    new_placeholder = "Povratna informacija uspešno zabeležena. Lahko vnesete novo mnenje ali komentar..."
    return "", gr.update(placeholder=new_placeholder)


# ─────────────────────────────────────────────────────────────────────────────
# Search logic (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def run_search(query_text, search_method, k_results):
    if not query_text or not search_method:
        return """
        <div style='
            border: 1px solid #e5e7eb; 
            padding: 10px; 
            min-height: 80px; 
            border-radius: 5px; 
            font-family: Inter, sans-serif; 
            font-size: 14px; 
            color: #666;'
        >Please enter a query and select a search method.</div>
        """, "<div style='border: 1px solid #e5e7eb; padding: 10px; min-height: 80px; border-radius: 5px; font-family: Inter, sans-serif; font-size: 14px; color: #666;'>Status or summary.</div>"

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

    answers = []
    for i in range(len(results)):
        row = results[i]
        # Depending on which search method was used, row might have a certain order
        if search_method in ("Lexical", "Semantic"):
            chunk_id, chunk_text, file_name, page_number, score = row
        else:
            chunk_id, score, chunk_text, file_name, page_number = row

        presigned_url = generate_presigned_url(file_name, page_number)
        snippet = f"""
        <p>
            <b>File:</b> <a href="{presigned_url}" target="_blank">{file_name} (Page {page_number})</a><br>
            <b>Score:</b> {score:.4f}<br>
            <b>Chunk:</b> {chunk_text[:500]}...
        </p>
        <hr>
        """
        answers.append(snippet)

    return f"""
    <div style='
        border: 1px solid #e5e7eb; 
        padding: 10px; 
        min-height: 80px; 
        border-radius: 5px; 
        font-family: Inter, sans-serif; 
        font-size: 14px; 
        color: #333; 
        transition: background-color 0.2s ease-in-out;'
    >
        {''.join(answers) if len(answers) else 'No results found.'}
    </div>
    """, f"""
    <div style='
        border: 1px solid #e5e7eb; 
        padding: 10px; 
        min-height: 80px; 
        border-radius: 5px; 
        font-family: Inter, sans-serif; 
        font-size: 14px; 
        color: #333;'
    >
        Found {len(answers)} chunks.
    </div>
    """

def fetch_selected_docs():
    """
    GET request to /get_selected_filenames/ to see which docs are currently selected
    via bounding-box in the DB.
    """
    try:
        response = requests.get("http://localhost:8000/get_selected_filenames/")
        data = response.json()
        filenames = data.get("selected_filenames", [])

        if len(filenames) == 0:
            return """
            <div style='
                border: 1px solid #e5e7eb; 
                padding: 10px; 
                min-height: 80px; 
                border-radius: 5px; 
                font-family: Inter, sans-serif; 
                font-size: 14px; 
                color: #666;'
            >No documents are currently selected.</div>
            """

        # build an HTML with line breaks
        joined = ""
        for i in range(len(filenames)):
            joined += filenames[i] + "<br>"

        return f"""
        <div style='
            border: 1px solid #e5e7eb; 
            padding: 10px; 
            min-height: 80px; 
            border-radius: 5px; 
            font-family: Inter, sans-serif; 
            font-size: 14px; 
            color: #333; 
            transition: background-color 0.2s ease-in-out;'
            onmouseover="this.style.backgroundColor='#f9fafb';"
            onmouseout="this.style.backgroundColor='transparent';"
        >
            <b>Currently selected docs:</b><br>{joined}
        </div>
        """
    except Exception as e:
        return f"""
        <div style='
            border: 1px solid #e5e7eb; 
            padding: 10px; 
            min-height: 80px; 
            border-radius: 5px; 
            font-family: Inter, sans-serif; 
            font-size: 14px; 
            color: red;'
        >Error retrieving selected docs: {e}</div>
        """

# ─────────────────────────────────────────────────────────────────────────────
# Build the Gradio interface
# ─────────────────────────────────────────────────────────────────────────────
def build_gradio_interface():
    with gr.Blocks(title="ZRSVN RAG with Postgres & BGE-M3 Embeddings") as demo:
        gr.Markdown("## Example RAG App with Leaflet Map + Postgres-based Searching")

        # NEW streaming-based chat
        gr.Markdown("## Chatbot z možnostjo povratne informacije")
        chat = gr.ChatInterface(
            fn=predict,
            type="messages",
            theme="ocean",
            flagging_mode="manual",
            #flagging_options=["Like", "Dislike", "Spam", "Inappropriate", "Other"],
            flagging_dir="/mnt/partit1/projects/zrsvn-rag",
            # you can supply any other ChatInterface parameters if needed
        )

        feedback_box = gr.Textbox(
            label="Povratna informacija", 
            placeholder="Vnesite vaše mnenje ali komentar..."
        )
        feedback_button = gr.Button("Pošlji povratno informacijo")

        # Clicking feedback will reset the feedback_box and update placeholder
        feedback_button.click(
            fn=handle_feedback,
            inputs=[feedback_box],
            outputs=[feedback_box, feedback_box]
        )

        gr.Markdown("---")

        # The rest of your existing search UI:
        gr.Markdown("### Iskanje po dokumentih (PostgreSQL + S3)")

        with gr.Row():
            query_box = gr.Textbox(
                label="Enter your query",
                placeholder="E.g., 'Environmental impact of project X?'",
            )
            search_method = gr.Radio(
                choices=["Lexical", "Semantic", "Hybrid"],
                value="Hybrid",
                label="Search Method",
                interactive=True,
            )
            k_slider = gr.Slider(
                minimum=1,
                maximum=15,
                value=5,
                step=1,
                label="Number of Results",
                interactive=True,
            )

        with gr.Accordion(
            label="Geographically determine scope of documents that are searched over",
            open=False):
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
                        transition: background-color 0.2s ease-in-out;
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

        with gr.Row():
            submit_button = gr.Button("Search")
            show_docs_button = gr.Button("Show Selected Docs")

        with gr.Row():
            output_area = gr.HTML(
                label="Search Results",
                value="""
                <div style='
                    border: 1px solid #e5e7eb; 
                    padding: 10px; 
                    min-height: 80px; 
                    border-radius: 5px; 
                    font-family: Inter, sans-serif; 
                    font-size: 14px; 
                    color: #666;'
                >Results will appear here...</div>
                """
            )
            status_area = gr.HTML(
                label="Status",
                value="""
                <div style='
                    border: 1px solid #e5e7eb; 
                    padding: 10px; 
                    min-height: 80px; 
                    border-radius: 5px; 
                    font-family: Inter, sans-serif; 
                    font-size: 14px; 
                    color: #666;'
                >Status or summary.</div>
                """
            )

        with gr.Accordion("📖 Navodila za uporabo", open=False):
            gr.Markdown("""
                By default, all documents are included in the search.
                By drawing a rectangle on the map, you can narrow down the search to include 
                only documents that are connected to a specific area.
            """)

        submit_button.click(
            fn=run_search,
            inputs=[query_box, search_method, k_slider],
            outputs=[output_area, status_area]
        )

        show_docs_button.click(
            fn=fetch_selected_docs,
            inputs=[],
            outputs=[status_area]
        )

    return demo


if __name__ == "__main__":
    interface = build_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7950,
        share=False
    )
