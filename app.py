import os
import json
# Za uporabniški vmesnik.
import gradio as gr
import psycopg2
import requests
from datetime import timedelta
from minio import Minio
from openai import AzureOpenAI
# Za beleženje povratnih informacij.
import csv
import time
# Za telemetrijo.
import logfire
from datetime import datetime
# FastAPI za dostopne točke.
from fastapi import FastAPI, Request
# Serviranje statičnih datotek.
from fastapi.staticfiles import StaticFiles
# Renderiranje HTML predlog.
from fastapi.templating import Jinja2Templates
# Uravnavanje CORS za odjemalski del (t.i. frontend) aplikacije.
from fastapi.middleware.cors import CORSMiddleware
# Vračanje odgovor na zahteve v obliki tokov (ang. stream).
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
# Pretvorba binarnih podatkov v niz znakov.
import base64
from pathlib import Path
# Pomoč pri tipizaciji.
from typing import Optional

# Funkcije za leksično, semantično in hibridno iskanje.
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

# Inicializiramo FastAPI aplikacijo, ki bo sprejemala HTTP zahtevke, z njimi
# upravljala in vračala odgovore.
# Objekt app bomo večinoma uporabljali za definicijo dostopnih točk (ang. endpointov) 
# in serviranje statičnih datotek.
app = FastAPI()

from pathlib import Path

# Nastavimo pot do mape z viri, ki v našem primeru vsebuje samo sliko zavoskega
# logotipa.
here = Path(__file__).parent.resolve() 
assets_path = here / "assets"

app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

# Ker odjemalski del teče na drugem portu kot zaledni del
# aplikacije moramo omogočiti CORS (Cross-Origin Resource Sharing). Brez slednjega* brskalnik ne dovoli odjemalskemu
# delu naše aplikacije, da bi prejela odgovor na GET zahtevo po vseh relevantnih geografskih točkah, zaradi česar se
# te ne prikažejo na vgnezdenem zemljevidu. 
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

# Ustvarimo globalno povezavo do baze in kurzor, ki ju bomo delili med dostopnimi točkami.
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Prebere 'zrsvn_logo.png' iz mape assets in vrne base64 niz,
# ki ga lahko uporabimo kot vrednost src v <img> oznaki.
def get_logo_b64() -> str:
    here = Path(__file__).parent.resolve()
    logo_path = here / "assets" / "zrsvn_logo.png"
    
    with open(logo_path, "rb") as f:
        data = f.read()
    b64_data = base64.b64encode(data).decode("utf-8")
    
    return b64_data

# ─────────────────────────────────────────────────────────────────────────────
# Definicije dostopnih točk.
# ─────────────────────────────────────────────────────────────────────────────
# GET dostopna točka, ki na zahtevo vrne vse geometrijske točke iz tabele 'najdbe',
# ki so povezane s katerim od PDF dokumentov.
# Ker je teh točk lahko veliko se jih pošilja v obliki toka (StreamingResponse). Na
# ta način se omogoči takojšnji prikaz prvih izmed prejetih podatkov, tudi če se jih
# en del še prenaša. Alternativa bi bila čakanje na to, da bi se zbralo najprej
# vse točke skupaj in potem naenkrat poslalo klientu. Slednja rešitev ni optimalna
# saj hitreje vodi do zamikov pri pošiljanju in k večji porabi pomnilnika na strani
# strežnika.
@app.get("/get_all_points/")
def get_all_points():
    query = """
        SELECT ST_AsGeoJSON(ST_Transform(wkb_geometry, 4326)) AS geom
        FROM najdbe
        WHERE file_id IS NOT NULL;
    """

    try:
        # Uporabimo nov kurzor znotraj funkcije, da ne motimo globalnega stanja.
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                print("[DEBUG] Query executed, but no results were returned.")
                return {"message": "No points found in the database."}

            # Generator, ki na podlagi vrstic poizvedbe sestavi JSON objekt s koordinatami točk.
            def generate():
                for row in results:
                    # Preverimo ali se v odgovoru nahaja GeoJSON.
                    if row[0]:
                        yield json.dumps({
                            "type": "Point",
                            "coordinates": json.loads(row[0])["coordinates"]
                        }) + "\n"
            # Prek StreamingResponse v HTTP odzivu pošljemo vsako vrstico posebej.
            return StreamingResponse(generate(), media_type="application/json")

    except psycopg2.ProgrammingError as e:
        print(f"[ERROR] Database query failed: {e}")
        conn.rollback()  # Roll back in case of an error
        return {"error": f"Database query failed: {str(e)}"}

    except psycopg2.Error as e:
        print(f"[ERROR] General database error: {e}")
        conn.rollback()  # Roll back in case of an error
        return {"error": f"General database error: {str(e)}"}

# GET dostopna točka, ki preveri katere točke najdišč vrst 
# (vezane na naše datoteke) se nahajajo znotraj
# robnega okvirja (tj. bounding boxa), ki ga je na zemljevid narisal uporabnik.
# - Če so vsi štirje parametri None (tj. uporabnik na zemljevid ni vnesel robnega 
#   okvirja; ta možnost je v podpisu funkcije poudarjena prek tipa Optional iz knjižnice
#   typing) vrnemo vse datoteke.
# - Če so parametri zapolnjeni:
#       - Ustvarimo robni okvir v globalni projekciji EPSG:4326 prek PostGIS 
#         funkcije ST_MakeEnvelope in ga transformiramo v lokalno projekcijo 
#         EPSG:3794.
#       - Poiščemo vse datoteke, ki imajo vsaj eno 'najdbe' točko znotraj robnega okvirja.
#       - Izbrane datoteke zapišemo v tabelo 'selected_files_log'.
@app.get("/get_files/")
def get_files(
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
):
    if None in (min_lat, max_lat, min_lon, max_lon):  # Case: App starts
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

    # Počistimo morebitne obstoječe vnose v selected_files_log in zapišemo nove.
    cur.execute("DELETE FROM selected_files_log;")
    for fid in file_ids:
        cur.execute("INSERT INTO selected_files_log (file_id) VALUES (%s);", (fid,))
    conn.commit()

    return {"selected_files": file_ids}

# GET dostopna točka, ki prebere vse datoteke iz 'selected_files_log' in jih vrne kot JSON.
@app.get("/get_selected_files/")
def get_selected_files():
    cur.execute("SELECT file_id FROM selected_files_log;")

    rows = cur.fetchall()
    file_ids = [row[0] for row in rows]

    return {"selected_files": file_ids}
    
# POST dostopna točka, ki pobriše vse vnose v 'selected_files_log' in nato vanjo
# vpiše vse datoteke iz tabele 'files'. Uporabi se ob zagonu/ponovni osvežitvi strani in ob kliku
# na gumb za odstranitev robnega okvirja.
@app.post("/reset_to_all_files/")
def reset_to_all_files():
    cur.execute("DELETE FROM selected_files_log;")

    cur.execute("INSERT INTO selected_files_log (file_id) SELECT id FROM files;")
    conn.commit()

    return {"message": "Selection reset to all files."}

# Serviramo statične datoteke iz mape templates (tj. omogočimo, da je map.html na voljo prek URLja oz. 
# dostopna brskalniku).
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
# Integracija Jinja2 predlog (ang. templates) v FastAPI. Ustvarimo objekt templates, s katerim lahko znotraj
# dostopnih točk dinamično generiramo strani iz HTML predlog.
# TODO Slednja integracija (trenutno) ni potrebna.
templates = Jinja2Templates(directory="templates")

# GET dostopna točka za streženje strani z Leaflet zemljevidom (map.html).
@app.get("/map")
def serve_map(request: Request):
    # request je objekt, ki ga FastAPI samodejno poda funkciji dostopne točk, ko ta naredi
    # HTTP zahtevek na /map. Na podlagi tega objekta lahko Jinja2Templates poskrbi za dinamično 
    # renderiranje HTML strani.
    return templates.TemplateResponse("map.html", {"request": request})

# Generiranje vnaprej podpisanega (ang. presigned) URLja, ki kaže na določeno posamezno stran PDF dokumenta.
def generate_presigned_url(file_key, page_number):
    if file_key is None:
        return None

    try:
        presigned_url = s3_client.presigned_get_object(
            bucket_name,
            file_key,
            # Povezava je aktualna eno uro.
            expires=timedelta(hours=1)
        )
        return f"{presigned_url}#page={page_number}"
    except Exception as e:
        print(f"Error generating link: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Nastavitve za interakcijo s pogovornim robotom (ang. chatbot) in beleženje 
# povratnih informacij uporabnikov.
# ─────────────────────────────────────────────────────────────────────────────
feedback_csv = "user_comment_log.csv"
# Če datoteka še ne obstaja jo ustvarimo, pri čimer določimo vsebino vrhnje vrstice.
if not os.path.exists(feedback_csv):
    with open(feedback_csv, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Chat History", "User Comment", "Index"])

# Za shranjevanje pogovorov med uporabnikom in LLMom.
global_chat_history = []

# Privzeti način iskanja.
global_search_method = "Hibridni"
# Privzeto število kontekstualnih elementov (k).
global_k_context_items = 5

# Posodobi globalno izbrano metodo iskanja, ko uporabnik spremeni izbiro znotraj Gradio uporabniškega vmesnika.
def update_search_method(search_method):
    global global_search_method
    global_search_method=search_method
# Posodobi globalno število kontekstualnih elementov, v skladu s tem, kar v Gradiu up. vm. nastavi uporabnik.
def update_context_k(k):
    global global_k_context_items
    global_k_context_items=k

# Pridobi dodatne kontekstualne elemente iz iskalnega sistema (run_search)
# in oblikuje prompt za LLM, ki vključuje:
#   - [CHUNK_TEXT]
#   - [SECTION_SUMMARY]
#   - [FILE_SUMMARY]
# Če je global_k_context_items == 0, vrne sporočilo, da je treba povečati število kontekstnih elementov.
def add_context(query):
    if global_k_context_items == 0:
        return "Za prikaz kontekstualnih elementov mora biti »Število kontekstualnih elementov« nastavljeno na večje od nič.", "ZERO_K"
    else:
        # Pokličemo run_search, da dobimo spodnji par vrednosti:
        # - Znakovni niz v obliki Markdown.
        # - Seznam rezultatov (od katerih je vsak v obliki slovarja).
        search_result = run_search(query, global_search_method, global_k_context_items)

        if not isinstance(search_result, tuple) or len(search_result) != 2:
            raise ValueError(f"Nepričakovan rezultat iskanja: {search_result}")

        results_md, results_list = search_result

        if not isinstance(results_list, list):
            raise ValueError(f"Rezultate smo pričakovali v obliki seznama, vendar smo dobili nekaj drugega: {type(results_list)}")

        chunk_texts = []
        section_summaries = []
        file_summaries = []

        # Iz vsakega elementa prejetega rezultata iskanja izločimo besedilni blok na podlagi katerega
        # je prišlo do ujemanja z iskalno poizvedbo (chunk_text) in oba z njim povezana povzetka (section_summary, 
        # file_summary).
        for item in results_list:
            chunk_texts.append(item["chunk_text"])
            section_summaries.append(item["section_summary"])
            file_summaries.append(item["file_summary"])

        # Vse elemente sestavimo skupaj v željeni format.
        context = ""
        for i in range(len(chunk_texts)):
            context += f"[CHUNK_TEXT]:\n{chunk_texts[i]}\n"
            context += f"[SECTION_SUMMARY]:\n{section_summaries[i]}\n"
            context += f"[FILE_SUMMARY]:\n{file_summaries[i]}\n\n"

        # Če iskanje ni vrnilo rezultatov vrnemo niz, ki to sporoči LLMu.
        if not chunk_texts:
            context = "No relevant context items found."
        
        # Osnovni poziv, ki vključuje kontekstualne elemente in uporabnikovo poizvedbo.
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

# Funkcija, ki jo kliče Gradio ChatInterface ob vsaki posredovani poizvedbi uporabnika.
# - Pripravi sporočilo za LLM (s kontekstom ali brez).
# - Pošlje sporočilo Azure OpenAI (GPT-4o-mini) in odgovor poda v toku (ang. streaming response).
# - Vzdržuje spomin celotnega pogovora (prek global_chat_history) za kasnejši zapis v CSV datoteko.
# Parametri:
# - message: Aktualna poizvedba uporabnika.
# - history: Zgodovina pogovora. 
def predict(message, history):
    global global_chat_history

    if history is None:
        history = []

    # Pridobimo rezultate iskanja in poziv z dodanimi kontekstualnimi elementi.
    results_md, query_with_context = add_context(message)

    # Sistemsko sporočilo za LLM s katerim uravnavamo njegovo splošno vedenje.
    system_prompt_content = """You are a helpful AI assistant. Always respond in Slovenian unless the context clearly requires another language for better understanding or relevance."""
    system_prompt = {
        "role": "system",
        "content": system_prompt_content
    }
    # Sporočilo, ki ga bomo posredovali LLMu se bo pričelo s sistemskim sporočilom.
    messages = [system_prompt]
    # K sporočilu dodamo zgodovino pogovora, če ta obstaja.
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    # Če je iskalna poizvedba (prek add_context -> run_search) bila uspešna, oplemenitimo aktualno uporabnikovo
    # poizvedbo s kontekstualnimi elementi. V nasprotnem primeru k sporočilu pripnemo samo iskalno poizvedbo.
    if query_with_context != "ZERO_K":
        messages.append({"role": "user", "content": query_with_context})
    else:
        messages.append({"role": "user", "content": message})

    # Inicializiramo Azure OpenAI klienta.
    endpoint = os.getenv("ZRSVN_AZURE_OPENAI_ENDPOINT")
    subscription_key = os.getenv("ZRSVN_AZURE_OPENAI_KEY")
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    # Prek Logfire nastavimo beleženje telemetričnih podatkov.
    logfire.configure(
        send_to_logfire=False, 
        service_name="zrsvn-rag"
    )
    logfire.instrument_openai(client)

    # Na Azure OpenAI pošljemo zahtevek po tokovno podanem odgovoru.
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    # Tu bomo hranili vse prejete dele odgovora LLMa skupaj.
    chunks = []

    # V globalno zgodovino pogovora zabeležimo najprej uporabnikovo aktualno poizvedbo.
    global_chat_history.append({
        "role": "user",
        "content": message,
    })

    # V globalno zgodovino pogovora dodamo prazen vnos, ki se v nadaljevanju postopoma polni z odgovorom LLMa. 
    global_chat_history.append({
        "role": "assistant", 
        "content": ""
    })

    for chunk in stream:
        # Varnostni ukrep s katerim preverjamo ali trenutni delček (chunk) odgovora vsebuje uporabne informacije (choices).
        # Če ne, izpišemo opozorilo in gremo na naslednji delček.
        if not chunk.choices:
            print("Empty choices received:", chunk)
            continue
        # Pod spremenljivko delta shranimo prvi element delčka odgovora. Če je ta prazen oz. None nastavimo delta na 
        # prazen niz ("").
        delta = chunk.choices[0].delta.content or ""
        # Delček odgovora shranimo na seznam chunks.
        chunks.append(delta)
        # Posodobimo zadnji zapis v globalni zgodovini pogovora s trenutno aktualnim delčkom odgovora.
        global_chat_history[-1]["content"] += delta
        # Vse do tega trenutka prejete delčke odgovora združimo v enoten znakovni niz.
        full_response = "".join(chunks)
        # Funkcija je generator - z yield postopoma oddajamo vrednosti dveh spremenljivk:
        # - full_response predstavlja odgovor, ki se znotraj pogovornega okna sproti tekoče prikazuje uporabniku. 
        # - results_md se enači z search_results_md znotraj build_gradio_interface(). Njeno vrednost se uporabi
        #   znotraj Gradio Accordion komponente za prikaz kontekstualnih elementov, ki so prispevali k odgovoru.
        yield full_response, results_md
        # Dodamo kratek zamik za lepši prikaz odgovora.
        time.sleep(0.025)

# Zabeleži uporabnikove povratne informacije in celotno zgodovino pogovora v CSV datoteko, ki vsebuje:
# - časovni žig,
# - zgodovino pogovora (v obliki JSON),
# - komentar uporabnika,
# - indeks zadnjega elementa v global_chat_history.
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

    return "", gr.update(placeholder="Povratna informacija uspešno zabeležena. Lahko vnesete novo mnenje ali komentar...")

# ─────────────────────────────────────────────────────────────────────────────
# Iskanje.
# ─────────────────────────────────────────────────────────────────────────────
# Izvede izbrano vrsto iskanja (leksično, semantično ali hibridno) nad besedilnimi bloki (ang. text chunks) in
# vrne par vrednosti (ang. tuple):
#   1) Niz znakov v obliki Markdown (niz vsebuje klikabilne povezave).
#   2) Seznam slovarjev z dodatnimi detajli (chunk_text, section_summary, file_summary itd.)
def run_search(query_text, search_method, k_results):
    # Če se zgodi, da uporabnik ni vnesel poizvedbe ali metoda iskanja ni izbrana potem namesto Markdowna
    # s kontekstualnimi elementi in seznama rezultatov vrnemo niz z navodilom in prazen seznam.
    if not query_text or not search_method:
        return "Vnesite poizvedbo in izberite način iskanja.", []

    # Če nimamo podanega (pravilnega) števila kontekstualnih elementov ga
    # privzeto nastavimo na 5.
    try:
        k = int(k_results)
    except ValueError:
        k = 5

    # Pokličemo ustrezno funkcijo iz query_handler.
    if search_method == "Leksični":
        results = lexical_search_limited_scope(query_text, k=k, db_params=db_params)
    elif search_method == "Semantični":
        results = semantic_search_limited_scope(query_text, k=k, db_params=db_params)
    else:
        results = hybrid_search_limited_scope(query_text, k=k, db_params=db_params)

    if not results:
        return "Iskanje ni vrnilo nobenih kontekstualnih elementov. Poskusite znova, z drugačnim vprašanjem oz. poizvedbo.", []

    answers = []
    results_list = []
    file_nr = 1

    # Rezultate obdelamo po vrsticah.
    for row in results:
        if search_method in ("Leksični", "Semantični"):
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
            snippet = f"""Datoteka št. {file_nr}: [{file_name} (stran {page_number})]({presigned_url})  
            Ocena relevantnosti: {float(score):.4f}  
            """
        else:
            snippet = f"""Datoteka št. {file_nr}: {file_name} (stran {page_number})  
            Ocena relevantnosti: {float(score):.4f}  
            """

        answers.append(snippet)

        result_dict = {
            # TODO file_number bi lahko uporabili za indic LLMu kakšen je nivo relevantnosti izbranega
            # kontekstualnega elementa.
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

# GET dostopna točka za pridobitev števila trenutno izbranih datotek iz FastAPI endpoint-a 
# '/get_selected_filenames/'.
# Vrne znakovni niz, ki uporabniku sporoči št. trenutno izbranih dokumentov.
def fetch_selected_docs():
    try:
        response = requests.get("http://localhost:8000/get_selected_files/")
        data = response.json()
        file_ids = data.get("selected_files", [])
        nr_docs = len(file_ids)
        # return f"Currently selected docs: {nr_docs}"
        return f"Št. trenutno izbranih dokumentov: {nr_docs}"

    except Exception as e:
        # return f"Error retrieving selected docs: {e}"
        return f"Napaka pri pridobivanju izbranih dokumentov: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Gradio uporabniški vmesnik.
# ─────────────────────────────────────────────────────────────────────────────
def build_gradio_interface():
    # Ob zagonu pretvorimo logotip v base64 obliko.
    logo_b64 = get_logo_b64()

    with gr.Blocks(title="ZRSVN RAG") as demo:

        # Vrstica z logotipom in imenom aplikacije.
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
        # Definiramo Markdown Gradio gradnik kjer bomo prikazali kontekstualne elemente.
        search_results_md = gr.Markdown("Tu se bodo pojavili elementi, ki predstavljajo kontekst pri pripravi odgovora...", label="Rezultati iskanja", render=False)

        chat = gr.ChatInterface(
            chatbot=gr.Chatbot(placeholder="<strong>Kaj vas tokrat zanima o monitoringih?</strong><br>Vprašanje vnesite v spodnjo vrstico."),
            fn=predict,
            type="messages",
            theme="ocean",
            flagging_mode="manual",
            flagging_dir="/mnt/partit1/fis/mag/zrsvn-rag",
            additional_outputs=[search_results_md]
        )

        # Področje za vnašanje povratnih informacij.
        with gr.Accordion(label="✉️ Pošlji povratne informacije", open=False):
            feedback_box = gr.Textbox(
            label="Povratna informacija", 
            placeholder="Vnesite vaše mnenje ali komentar..."
            )
            feedback_button = gr.Button("Pošlji")
            feedback_button.click(handle_feedback, inputs=[feedback_box], outputs=[feedback_box, feedback_box])

        # Gradnik znotraj katerega uporabnik določi obseg dokumentov, ki bodo vključeni v iskanje.
        # To mu omogoča Leaflet zemljevid, ki je v gradnik vstavljen kot iframe element.
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
            # Gumb, ki ob kliku pokliče fetch_selected_docs in prikaže št. trenutno izbranih dokumentov.
            show_docs_button = gr.Button("Klikni za posodobitev prikaza št. trenutno izbranih dokumentov")
            docs_text = gr.Markdown()
            show_docs_button.click(fetch_selected_docs, inputs=[], outputs=docs_text)
   
        # Za nastavitve konteksta.
        with gr.Accordion(label="⚙️ Nastavitve konteksta", open=False):
            with gr.Row():
                search_method = gr.Radio(
                    choices=["Leksični", "Semantični", "Hibridni"],
                    value=global_search_method,
                    label="Način iskanja",
                    interactive=True
                )
                k_slider = gr.Slider(
                    minimum=0,
                    maximum=15,
                    value=global_k_context_items,
                    step=1,
                    label="Število kontekstualnih elementov",
                    interactive=True
                )
                # Poskrbimo, da se ob spremembi posodobita globalni spremenljivki.
                search_method.change(fn=update_search_method, inputs=[search_method], outputs=[])
                k_slider.change(fn=update_context_k, inputs=[k_slider], outputs=[])
        
        # Prikaz kontekstualnih elementov.
        with gr.Accordion(label="📝 Preglej kontekstualne elemente", open=False):
            search_results_md.render()
        
        with gr.Accordion("📖 Navodila za uporabo", open=False):
            gr.Markdown("""
                Privzeto so v iskanje vključeni vsi dokumenti.  
                Z risanjem robnega okvirja na zemljevidu lahko omejite  
                iskanje samo na dokumente, povezane s tem območjem.
            """)

    return demo

if __name__ == "__main__":
    interface = build_gradio_interface()
    interface.launch(
        favicon_path="./assets/zrsvn_logo.png", 
        server_name="127.0.0.1", 
        server_port=7950, 
        share=False
    )