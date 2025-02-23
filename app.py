import os
import json
import gradio as gr
import psycopg2
import requests
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Local imports for our custom modules
# ─────────────────────────────────────────────────────────────────────────────
from model_handling import embedding_model  # The BGE-M3 model instance
from query_handler import lexical_search_limited_scope, semantic_search_limited_scope, hybrid_search_limited_scope

# Load environment variables for DB password
load_dotenv()
POSTGIS_PASSWORD = os.getenv("POSTGIS_TEST1_PASSWORD")

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
            # row[0] is a GeoJSON string for the geometry
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
    for filename in filenames:
        cur.execute("INSERT INTO selected_files_log (filename) VALUES (%s);", (filename,))
    conn.commit()

    return {"selected_filenames": filenames}


# ─────────────────────────────────────────────────────────────────────────────
# ─── NEW ENDPOINT ADDED HERE: /get_selected_filenames/ ──────────────────────
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/get_selected_filenames/")
def get_selected_filenames():
    """
    Return all filenames currently in 'selected_files_log'.
    This is used by the Gradio UI to see which files were selected on the map.
    """
    cur.execute("SELECT filename FROM selected_files_log;")
    rows = cur.fetchall()
    filenames = [row[0] for row in rows]
    return {"selected_filenames": filenames}
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset_selected_filenames/")
def reset_selected_filenames():
    """
    Clears all entries in the `selected_files_log` table, effectively resetting the selected document state.
    """
    try:
        cur.execute("DELETE FROM selected_files_log;")  # Remove all selected docs
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


# ─────────────────────────────────────────────────────────────────────────────
# Gradio integration
# ─────────────────────────────────────────────────────────────────────────────
def run_search(query_text, search_method, k_results):
    """
    Called by Gradio UI to run the user-specified search type 
    (Lexical, Semantic, or Hybrid) with top-k results.
    """
    if not query_text or not search_method:
        return "Please enter a query and select a search method.", ""

    # Ensure we do an integer cast for top-k
    try:
        k = int(k_results)
    except ValueError:
        k = 5

    # Run the desired search
    if search_method == "Lexical":
        results = lexical_search_limited_scope(query_text, k=k, db_params=db_params)
    elif search_method == "Semantic":
        results = semantic_search_limited_scope(query_text, k=k, db_params=db_params)
    else:
        results = hybrid_search_limited_scope(query_text, k=k, db_params=db_params)

    # Format results for display
    answers = []

    for result in results:
        if search_method == "Lexical":
            # Lexical search follows (chunk_id, chunk_text, file_name, page_number, score)
            chunk_id, chunk_text, file_name, page_number, score = result
        elif search_method == "Semantic":
            # Semantic search follows (chunk_id, chunk_text, file_name, page_number, score)
            chunk_id, chunk_text, file_name, page_number, score = result
        else:
            # Hybrid search follows (chunk_id, score, chunk_text, file_name, page_number)
            chunk_id, score, chunk_text, file_name, page_number = result

        snippet = (
            f"File: {file_name}, Page: {page_number}, Score: {score:.4f}\n"
            f"Chunk: {chunk_text[:500]}...\n"
            "────────────────────────────────────\n"
        )
        answers.append(snippet)

    return "\n".join(answers), f"Found {len(answers)} chunks."


# ─────────────────────────────────────────────────────────────────────────────
# ─── NEW HELPER FUNCTION TO FETCH FILENAMES FROM /get_selected_filenames/ ───
# ─────────────────────────────────────────────────────────────────────────────
def fetch_selected_docs():
    """
    Calls our new FastAPI endpoint /get_selected_filenames
    and returns a short message with the currently selected docs.
    """
    try:
        response = requests.get("http://localhost:8000/get_selected_filenames/")
        data = response.json()
        filenames = data.get("selected_filenames", [])
        if not filenames:
            return "No documents are currently selected."
        joined = ", ".join(filenames)
        return f"Currently selected docs: {joined}"
    except Exception as e:
        return f"Error retrieving selected docs: {e}"

def build_gradio_interface():
    with gr.Blocks(title="ZRSVN RAG with Postgres & BGE-M3 Embeddings") as demo:
        gr.Markdown("## Example RAG App with Leaflet Map + Postgres-based Searching")

        # The map is served at /map endpoint. We embed it via iframe
        gr.HTML("""
          <style>
            .map-container {
              display: flex;
              flex-direction: column;
              height: 70vh; 
              /* The black border goes HERE so it's around the entire iframe. */
              /*border: 1px solid black;*/  
              box-sizing: border-box;
              margin: 0;
              padding: 0;
              overflow: hidden;
            }
            .map-container iframe {
              flex: 1;
              width: 100%;
              display: block;
              margin: 0;
              padding: 0;
              /* No border on the iframe itself, so the container border is uniform. */
              border: none;
              overflow: hidden;
            }
          </style>

          <div class="map-container">
            <iframe src="http://127.0.0.1:8000/map" scrolling="no"></iframe>
          </div>
        """)





        #
        # ─────────────────────────────────────────────────────────────────
        #  Row for query, search method, slider
        # ─────────────────────────────────────────────────────────────────
        with gr.Row():
            query_box = gr.Textbox(
                label="Enter your query",
                placeholder="E.g., 'Environmental impact of project X?'"
            )
            search_method = gr.Radio(
                choices=["Lexical", "Semantic", "Hybrid"],
                value="Hybrid",
                label="Search Method"
            )
            k_slider = gr.Slider(
                minimum=1,
                maximum=15,
                value=5,
                step=1,
                label="Number of Results"
            )

        #
        # ─────────────────────────────────────────────────────────────────
        #  Row for buttons (Search & Show Docs)
        # ─────────────────────────────────────────────────────────────────
        with gr.Row():
            submit_button = gr.Button("Search")
            show_docs_button = gr.Button("Show Selected Docs")

        #
        # ─────────────────────────────────────────────────────────────────
        #  Row for the search results and status
        # ─────────────────────────────────────────────────────────────────
        with gr.Row():
            output_area = gr.Textbox(
                label="Search Results",
                placeholder="Results will appear here.",
                lines=15
            )
            status_area = gr.Textbox(
                label="Status",
                placeholder="Status or summary.",
                lines=2
            )

        # Wire up the callbacks *after* the components exist:
        submit_button.click(
            fn=run_search,
            inputs=[query_box, search_method, k_slider],
            outputs=[output_area, status_area]
        )

        show_docs_button.click(
            fn=fetch_selected_docs,
            inputs=[],
            outputs=[status_area]   # results go to 'status_area'
        )

    return demo



# ─────────────────────────────────────────────────────────────────────────────
# Main entry - run Gradio app
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    interface = build_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,  # or any port you prefer
        share=False
    )

# ─────────────────────────────────────────────────────────────────────────────
# What Was Removed or Changed?
# ─────────────────────────────────────────────────────────────────────────────
# 1) Removed references to the old 'rag_components', 'scripts.preprocess', 
#    'BM25Okapi', 'faiss' indexing, and older "LLM" enumerations for GPT or Azure.
# 2) Removed the CSV feedback logic and advanced search config from the old code. 
# 3) No longer doing local or Azure model calls. 
# 4) The code now uses the new Postgres-based chunk storage and the new BGE-M3 
#    embedding approach (see model_handling.py). 
# 5) Replaced old references to produce a simpler Gradio UI that calls the 
#    lexical/semantic/hybrid searches from query_handler.py.
# 6) Replaced or removed streaming + advanced logic for simpler classical loops 
#    and straightforward functionalities.
