import os
import fitz            # PyMuPDF
from tqdm import tqdm
import semchunk        # For semantic chunking
import psycopg2
from dotenv import load_dotenv
from model_handling import embedding_model
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from pgvector.psycopg2 import register_vector

# Register adapters for numpy
register_adapter(np.ndarray, lambda arr: AsIs(arr.tolist()))
register_adapter(np.float32, lambda val: AsIs(val))
register_adapter(np.float64, lambda val: AsIs(val))

load_dotenv()
POSTGIS_PASSWORD = os.getenv("POSTGIS_TEST1_PASSWORD")

db_params = {
    "dbname": "postgis_test1",
    "user": "ggatej-pg",
    "password": POSTGIS_PASSWORD,
    "host": "localhost",
    "port": "5432"
}

def read_pdfs(folder_path):
    """
    Scans a folder for PDF files, reads them with PyMuPDF, 
    and returns a dictionary containing pages for each PDF.

    Returns:
        pdf_data (dict):
            {
              0: {"file_name": "xyz.pdf", "pages": [... page dicts ...]},
              1: {"file_name": "abc.pdf", "pages": [...]},
              ...
            }
    """
    pdf_data = {}
    pdf_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_files.append(filename)

    pdf_files.sort()

    for index in tqdm(range(len(pdf_files)), desc="Processing PDFs"):
        file_name = pdf_files[index]
        file_path = os.path.join(folder_path, file_name)
        doc = fitz.open(file_path)
        pages = []

        # Classic loop over pages
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text")
            # We store page_number (1-based) plus the actual text
            pages.append({
                "page_number": page_num + 1,
                "page_text": page_text
            })

        pdf_data[index] = {
            "file_name": file_name,
            "pages": pages
        }

    return pdf_data


def count_words(text):
    """
    Returns the approximate number of words in a text, 
    using basic whitespace splitting.
    """
    return len(text.split())


def make_chunks(pdf_data, chunk_size=150, chunk_threshold=0):
    """
    Splits the text content of PDFs into smaller chunks 
    while preserving file/page structure using semchunk.

    Args:
        pdf_data (dict): The dictionary from read_pdfs(...)
        chunk_size (int): Number of words per chunk.
        chunk_threshold (int): Minimum #words for a chunk to be kept.

    Returns:
        chunked_data (list of dict): 
            Each dict has file_name, page_number, chunk_text, 
            chunk_char_count, chunk_word_count, etc.
    """
    chunker = semchunk.chunkerify(count_words, chunk_size)
    chunked_data = []

    for pdf_index, pdf_info in pdf_data.items():
        file_name = pdf_info["file_name"]

        # Classic loop over pages
        for page in pdf_info["pages"]:
            page_number = page["page_number"]
            page_text = page["page_text"]

            # Use semchunk to produce a list of text chunks
            chunks = chunker(page_text)

            # Another classic loop over each chunk
            for chunk_index in range(len(chunks)):
                chunk_text = chunks[chunk_index]
                word_count = count_words(chunk_text)
                if word_count >= chunk_threshold:
                    chunked_data.append({
                        "file_name": file_name,
                        "page_number": page_number,
                        "chunk_text": chunk_text,
                        "chunk_char_count": len(chunk_text),
                        "chunk_word_count": word_count
                    })
    return chunked_data


def store_chunks_in_db(chunked_data):
    """
    Stores chunk metadata into the 'chunks' table, linking each chunk
    to a file in the 'files' table by matching the file_name column.
    """
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Insert each chunk
    # We assume 'files' table has a unique 'file_name' column, 
    # and 'chunks' has (id SERIAL, files_id FK, page_number, chunk_text, etc.)
    for chunk in chunked_data:
        # 1) fetch files.id from 'files'
        get_file_id_query = "SELECT id FROM files WHERE file_name = %s;"
        cursor.execute(get_file_id_query, (chunk["file_name"],))
        file_id_result = cursor.fetchone()
        if not file_id_result:
            # If not found, we might skip or insert a new row in 'files'
            # for simplicity, we assume 'files' is pre-populated
            continue
        file_id = file_id_result[0]

        insert_query = """
            INSERT INTO chunks (files_id, page_number, chunk_text)
            VALUES (%s, %s, %s)
            RETURNING id;
        """
        cursor.execute(insert_query, (file_id, chunk["page_number"], chunk["chunk_text"]))

    conn.commit()
    cursor.close()
    conn.close()


def generate_and_store_embeddings():
    """
    Fetch all chunk texts from 'chunks', generate BGE-M3 embeddings,
    and store them into the 'embeddings' table. 
    (chunks_id -> embeddings.vector).
    """
    import logging
    from tqdm import tqdm

    logging.info("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()
    logging.info("Database connection established.")

    # 1) Fetch all chunks
    logging.info("Fetching text chunks from the database...")
    cursor.execute("SELECT id, chunk_text FROM chunks;")
    chunks = cursor.fetchall()  # list of (id, chunk_text)
    total_chunks = len(chunks)
    logging.info(f"Fetched {total_chunks} chunks from the database.")

    batch_size = 100
    total_batches = (total_chunks // batch_size) + (1 if total_chunks % batch_size else 0)
    progress_bar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")

    # 2) For each batch, generate embeddings + store
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        chunk_ids = []
        chunk_texts = []
        for row in batch:
            chunk_ids.append(row[0])
            chunk_texts.append(row[1])

        # The embedding model returns a dict with "dense_vecs"
        embeddings_output = embedding_model.encode(chunk_texts)
        dense_embeddings = embeddings_output["dense_vecs"]
        dense_embeddings = np.array(dense_embeddings, dtype=np.float32)

        insert_query = "INSERT INTO embeddings (vector, chunks_id) VALUES (%s, %s)"
        data = []
        for emb, cid in zip(dense_embeddings, chunk_ids):
            data.append((emb.tolist(), cid))

        cursor.executemany(insert_query, data)
        conn.commit()

        progress_bar.update(1)

    progress_bar.close()
    cursor.close()
    conn.close()
    logging.info("Embedding process completed successfully.")


def pipeline_ingest(folder_of_pdfs):
    """
    Example function showing how to read PDFs, chunk them, store to DB,
    and then generate + store embeddings. 
    This can be run once or repeatedly to ingest more data.
    """
    pdf_data = read_pdfs(folder_of_pdfs)
    chunked_data = make_chunks(pdf_data, chunk_size=150, chunk_threshold=0)
    store_chunks_in_db(chunked_data)
    generate_and_store_embeddings()
    print("Pipeline ingestion is complete.")


# ─────────────────────────────────────────────────────────────────────────────
# What Was Removed or Changed?
# ─────────────────────────────────────────────────────────────────────────────
# 1) Removed old references to FAISS, BM25, streaming or token-based chunkers 
#    in the older code. Instead we rely on semchunk for chunking.
# 2) Thoroughly commented each step to make it more interpretable with 
#    classic loops. 
# 3) Provided a "pipeline_ingest" function as a usage example.
