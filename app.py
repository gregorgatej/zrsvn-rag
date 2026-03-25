import os
import json
import gradio as gr
import psycopg2
import requests
from datetime import timedelta
from minio import Minio
from openai import AzureOpenAI
import csv
import time
 # For telemetry.
import logfire
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import base64
from pathlib import Path
from typing import Optional

 # Functions for lexical, semantic, and hybrid search.
from query_handler import (
    lexical_search_limited_scope, 
    semantic_search_limited_scope, 
    hybrid_search_limited_scope
)

load_dotenv()
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
s3_endpoint_url = "moja.shramba.arnes.si"
bucket_name = "zrsvn-rag-najdbe-najvecji"

s3_client = Minio(
    endpoint=s3_endpoint_url,
    access_key=s3_access_key,
    secret_key=s3_secret_access_key,
    secure=True
)

app = FastAPI()

from pathlib import Path

here = Path(__file__).parent.resolve() 
assets_path = here / "assets"

app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
db_params = {
    "dbname": "zrsvn",
    "user": "ggatej-pg",
    "password": POSTGRES_PASSWORD,
    "host": "localhost",
    "port": "5432",
    "options":  "-c search_path=public,rag_najdbe"
}

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

def get_logo_b64() -> str:
    here = Path(__file__).parent.resolve()
    logo_path = here / "assets" / "zrsvn_logo.png"
    
    with open(logo_path, "rb") as f:
        data = f.read()
    b64_data = base64.b64encode(data).decode("utf-8")
    
    return b64_data

@app.get("/get_all_points/")
def get_all_points():
    query = """
        SELECT ST_AsGeoJSON(ST_Transform(wkb_geometry, 4326)) AS geom
        FROM najdbe
        WHERE file_id IS NOT NULL;
    """

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                print("[DEBUG] Query executed, but no results were returned.")
                return {"message": "No points found in the database."}

            def generate():
                for row in results:
                    if row[0]:
                        yield json.dumps({
                            "type": "Point",
                            "coordinates": json.loads(row[0])["coordinates"]
                        }) + "\n"
            return StreamingResponse(generate(), media_type="application/json")

    except psycopg2.ProgrammingError as e:
        print(f"[ERROR] Database query failed: {e}")
        conn.rollback()  
        return {"error": f"Database query failed: {str(e)}"}

    except psycopg2.Error as e:
        print(f"[ERROR] General database error: {e}")
        conn.rollback()  
        return {"error": f"General database error: {str(e)}"}

@app.get("/get_files/")
def get_files(
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
):
    if None in (min_lat, max_lat, min_lon, max_lon):
        cur.execute("SELECT id FROM files;")
        results = cur.fetchall()
    else:
        query = """
          WITH bbox_3794 AS (
            SELECT ST_Transform(
                     ST_MakeEnvelope(%s, %s, %s, %s, 4326),
                     3794
                   ) AS geom
          )
          SELECT DISTINCT f.id
          FROM rag_najdbe.files AS f
          JOIN rag_najdbe.najdbe AS n
            ON f.id = n.file_id
          JOIN bbox_3794
            ON ST_Within(n.wkb_geometry, bbox_3794.geom);
        """

        cur.execute(query, (min_lon, min_lat, max_lon, max_lat))
        results = cur.fetchall()

        if not results:
            cur.execute("SELECT id FROM files;")
            results = cur.fetchall()
    
    file_ids = []
    for row in results:
        file_ids.append(row[0])

    cur.execute("DELETE FROM selected_files_log;")
    for fid in file_ids:
        cur.execute("INSERT INTO selected_files_log (file_id) VALUES (%s);", (fid,))
    conn.commit()

    return {"selected_files": file_ids}

@app.get("/get_selected_files/")
def get_selected_files():
    cur.execute("SELECT file_id FROM selected_files_log;")

    rows = cur.fetchall()
    file_ids = [row[0] for row in rows]

    return {"selected_files": file_ids}
    
@app.post("/reset_to_all_files/")
def reset_to_all_files():
    cur.execute("DELETE FROM selected_files_log;")

    cur.execute("INSERT INTO selected_files_log (file_id) SELECT id FROM files;")
    conn.commit()

    return {"message": "Selection reset to all files."}

app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")

@app.get("/map")
def serve_map(request: Request):
    return templates.TemplateResponse("map.html", {"request": request})

def generate_presigned_url(file_key, page_number):
    if file_key is None:
        return None

    try:
        presigned_url = s3_client.presigned_get_object(
            bucket_name,
            file_key,
            expires=timedelta(hours=1)
        )
        return f"{presigned_url}#page={page_number}"
    except Exception as e:
        print(f"Error generating link: {e}")
        return None

feedback_csv = "user_comment_log.csv"
if not os.path.exists(feedback_csv):
    with open(feedback_csv, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Chat History", "User Comment", "Index"])

global_chat_history = []
global_search_method = "Hibridni"
global_k_context_items = 5

def update_search_method(search_method):
    global global_search_method
    global_search_method=search_method

def update_context_k(k):
    global global_k_context_items
    global_k_context_items=k

 # Retrieve additional context items from the search system (run_search)
 # and form a prompt for the LLM, which includes:
 #   - [CHUNK_TEXT]
 #   - [SECTION_SUMMARY]
 #   - [FILE_SUMMARY]
def add_context(query):
    if global_k_context_items == 0:
        return "To display context items, the 'Number of context items' must be set to greater than zero.", "ZERO_K"
    else:
        search_result = run_search(query, global_search_method, global_k_context_items)

        if not isinstance(search_result, tuple) or len(search_result) != 2:
            raise ValueError(f"Unexpected search result: {search_result}")

        results_md, results_list = search_result

        if not isinstance(results_list, list):
            raise ValueError(f"Expected results as a list, but got: {type(results_list)}")

        chunk_texts = []
        section_summaries = []
        file_summaries = []

        # From each element of the received search result, extract the 
        # text block that matched the search query (chunk_text) and both associated summaries 
        # (section_summary, file_summary).
        for item in results_list:
            chunk_texts.append(item["chunk_text"])
            section_summaries.append(item["section_summary"])
            file_summaries.append(item["file_summary"])

        # Combine all elements together in the desired format.
        context = ""
        for i in range(len(chunk_texts)):
            context += f"[CHUNK_TEXT]:\n{chunk_texts[i]}\n"
            context += f"[SECTION_SUMMARY]:\n{section_summaries[i]}\n"
            context += f"[FILE_SUMMARY]:\n{file_summaries[i]}\n\n"

        if not chunk_texts:
            context = "No relevant context items found."
        
        base_prompt = (
            "With your general knowledge and with the help of the following context items, please answer the query. "
            "Give yourself room to think by extracting relevant passages from the context before answering the query. "
            "Don't return the thinking, only return the answer. "
            "Make sure your answers are as explanatory as possible.\n\n"
            "The context items will be provided in the following order: first, the [CHUNK_TEXT], which is the most granular and specific information, "
            "allowing you to focus directly on the most relevant content. After that, you will receive the [SECTION_SUMMARY], which corresponds to the section the chunk falls under, "
            "and then the [FILE_SUMMARY], which corresponds to the file the section is part of. These will provide additional context or clarification.\n\n"
            "Context items:\n\n"
            "{context}\n\n"
            "User query: {query}\n\n"
            "Answer:"
        )

        prompt_with_context = base_prompt.format(context=context, query=query)
        print(prompt_with_context)

        return results_md, prompt_with_context

 # Function called by Gradio ChatInterface for each user query.
 # - Prepares the message for the LLM (with or without context).
 # - Sends the message to Azure OpenAI (GPT-4o-mini) and returns the answer as a streaming response.
 # - Maintains the entire conversation history (via global_chat_history) for later logging to a CSV file.
 # Parameters:
 # - message: The current user query.
 # - history: Conversation history.
def predict(message, history):
    global global_chat_history

    if history is None:
        history = []

    results_md, query_with_context = add_context(message)

    system_prompt_content = """You are a helpful AI assistant. Always respond in Slovenian unless the context clearly requires another language for better understanding or relevance."""
    system_prompt = {
        "role": "system",
        "content": system_prompt_content
    }
    messages = [system_prompt]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    if query_with_context != "ZERO_K":
        messages.append({"role": "user", "content": query_with_context})
    else:
        messages.append({"role": "user", "content": message})

    endpoint = os.getenv("ZRSVN_AZURE_OPENAI_ENDPOINT")
    subscription_key = os.getenv("ZRSVN_AZURE_OPENAI_KEY")
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    logfire.configure(
        send_to_logfire=False, 
        service_name="zrsvn-rag"
    )
    logfire.instrument_openai(client)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    chunks = []

    global_chat_history.append({
        "role": "user",
        "content": message,
    })

    global_chat_history.append({
        "role": "assistant", 
        "content": ""
    })

    for chunk in stream:
        if not chunk.choices:
            print("Empty choices received:", chunk)
            continue
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)
        global_chat_history[-1]["content"] += delta
        full_response = "".join(chunks)
        yield full_response, results_md
        time.sleep(0.025)

 # Record user feedback and the entire conversation history in a CSV file, which contains:
 # - timestamp,
 # - conversation history (as JSON),
 # - user comment,
 # - index of the last element in global_chat_history.
def handle_feedback(comment):
    global global_chat_history

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    chat_history_cleaned = []
    for entry in global_chat_history:
        newdict = {"role": entry["role"], "content": entry["content"]}
        chat_history_cleaned.append(newdict)

    chat_history_json = json.dumps(chat_history_cleaned, ensure_ascii=False)
    last_index = max(0, len(global_chat_history) - 1)

    with open(feedback_csv, mode='a', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, chat_history_json, comment, last_index])

    return "", gr.update(placeholder="Feedback successfully recorded. You can enter a new opinion or comment...")

 # Executes the selected type of search (lexical, semantic, or hybrid) on text chunks and
 # returns a tuple:
 #   1) A Markdown string (with clickable links).
 #   2) A list of dictionaries with additional details (chunk_text, section_summary, file_summary, etc.)
def run_search(query_text, search_method, k_results):
    if not query_text or not search_method:
        return "Enter a query and select a search method.", []

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
        return "The search did not return any context items. Try again with a different question or query.", []

    answers = []
    results_list = []
    file_nr = 1

    for row in results:
        if search_method in ("Lexical", "Semantic"):
            if len(row) != 8:
                continue
            chunk_id, score, chunk_text, file_name, s3_link, page_number, section_summary, file_summary = row
        # Hibridni.
        else:
            if len(row) != 8:
                continue
            chunk_id, score, chunk_text, file_name, s3_link, page_number, section_summary, file_summary = row

        presigned_url = generate_presigned_url(s3_link, page_number)

        if presigned_url:
            snippet = f"""File nr. {file_nr}: [{file_name} (page {page_number})]({presigned_url})  
            Relevance score: {float(score):.4f}  
            """
        else:
            snippet = f"""File nr. {file_nr}: {file_name} (page {page_number})  
            Relevance score: {float(score):.4f}  
            """

        answers.append(snippet)

        result_dict = {
            "file_number": file_nr,
            "file_name": file_name,
            "s3_link": s3_link,
            "page_number": page_number,
            "chunk_text": chunk_text,
            "section_summary": section_summary,
            "file_summary": file_summary,
            "presigned_url": presigned_url if presigned_url else "",
            "score": round(score, 4),
            "snippet": snippet
        }

        results_list.append(result_dict)

        file_nr += 1

    markdown_answers = "\n".join(answers)

    return markdown_answers, results_list

def fetch_selected_docs():
    try:
        response = requests.get("http://localhost:8000/get_selected_files/")
        data = response.json()
        file_ids = data.get("selected_files", [])
        nr_docs = len(file_ids)
        return f"Number of currently selected documents: {nr_docs}"

    except Exception as e:
        return f"Error retrieving selected docs: {e}"

def build_gradio_interface():
    logo_b64 = get_logo_b64()

    with gr.Blocks(title="ZRSVN RAG") as demo:

        gr.HTML(f"""
        <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
            <img 
              src="data:image/png;base64,{logo_b64}" 
              alt="ZRSVN Logo" 
              style="height:50px; margin-right:15px;"
            >
            <h2 style="margin:0; font-size:24px; line-height:1;">ZRSVN RAG</h2>
        </div>
        """)
        search_results_md = gr.Markdown("Context elements for the answer will appear here...", label="Search Results", render=False)

        chat = gr.ChatInterface(
            chatbot=gr.Chatbot(placeholder="<strong>What would you like to know about monitoring this time?</strong><br>Enter your question below."),
            fn=predict,
            type="messages",
            theme="ocean",
            flagging_mode="manual",
            flagging_dir="/mnt/partit1/fis/mag/zrsvn-rag",
            additional_outputs=[search_results_md]
        )

        with gr.Accordion(label="✉️ Pošlji povratne informacije", open=False):
            feedback_box = gr.Textbox(
            label="Povratna informacija", 
            placeholder="Vnesite vaše mnenje ali komentar..."
            )
            feedback_button = gr.Button("Pošlji")
            feedback_button.click(handle_feedback, inputs=[feedback_box], outputs=[feedback_box, feedback_box])

        with gr.Accordion(label="🌍 Geografsko določite obseg dokumentov, vključenih v iskanje", open=False):
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
            show_docs_button = gr.Button("Klikni za posodobitev prikaza št. trenutno izbranih dokumentov")
            docs_text = gr.Markdown()
            show_docs_button.click(fetch_selected_docs, inputs=[], outputs=docs_text)
   
        with gr.Accordion(label="✉️ Send feedback", open=False):
            feedback_box = gr.Textbox(
                label="Feedback",
                placeholder="Enter your opinion or comment..."
            )
            feedback_button = gr.Button("Send")
            feedback_button.click(handle_feedback, inputs=[feedback_box], outputs=[feedback_box, feedback_box])

        with gr.Accordion(label="🌍 Geographically define the scope of documents included in the search", open=False):
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
            """
            )
            show_docs_button = gr.Button("Click to update the number of currently selected documents")
            docs_text = gr.Markdown()
            show_docs_button.click(fetch_selected_docs, inputs=[], outputs=docs_text)

        with gr.Accordion(label="⚙️ Context settings", open=False):
            with gr.Row():
                search_method = gr.Radio(
                    choices=["Lexical", "Semantic", "Hybrid"],
                    value=global_search_method,
                    label="Search method",
                    interactive=True
                )
                k_slider = gr.Slider(
                    minimum=0,
                    maximum=15,
                    value=global_k_context_items,
                    step=1,
                    label="Number of context items",
                    interactive=True
                )
                search_method.change(fn=update_search_method, inputs=[search_method], outputs=[])
                k_slider.change(fn=update_context_k, inputs=[k_slider], outputs=[])

        with gr.Accordion(label="📝 Review context items", open=False):
            search_results_md.render()

        with gr.Accordion("📖 Instructions for use", open=False):
            gr.Markdown("""
                By default, all documents are included in the search.  
                By drawing a bounding box on the map, you can limit  
                the search to only documents related to that area.
            """)