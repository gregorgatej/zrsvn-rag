import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from pgvector.psycopg2 import register_vector
import re
import logfire

logfire.configure(
        send_to_logfire=False, 
        service_name="zrsvn-rag"
    )
logfire.instrument_psycopg()

from model_handling import embedding_model

def escape_special_chars(query):
    """Escapes special characters for full-text search in ParadeDB/PostgreSQL so we don't receive ParseError when running lexical or hybrid search."""
    return re.sub(r'([^\w\s])', r'\\\1', query)  # Escape all non-alphanumeric characters

# Register numpy adapters so we can easily store/fetch arrays
register_adapter(np.ndarray, lambda arr: AsIs(arr.tolist()))
register_adapter(np.float32, lambda val: AsIs(val))
register_adapter(np.float64, lambda val: AsIs(val))

def lexical_search_limited_scope(query, k=5, db_params=None):
    """
    Performs a BM25-like lexical search using 'paradedb' or a Postgres text index, 
    but restricted to only those files present in 'selected_files_log'.
    
    Args:
        query (str): User's query string.
        k (int): Number of results to return.
        db_params (dict): DB connection parameters.

    Returns:
        list of tuples: (chunk_id, chunk_text, file_name, page_number, score).
    """
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    safe_query = escape_special_chars(query)

    search_query = """
        SELECT 
            c.id, 
            c.chunk_text, 
            f.file_name,
            f.s3_link,
            c.page_number, 
            paradedb.score(c.id) AS score
        FROM chunks c
        JOIN files f ON c.files_id = f.id
        JOIN selected_files_log sfl ON f.file_name = sfl.filename
        WHERE c.chunk_text @@@ %s
        ORDER BY score DESC
        LIMIT %s;
    """

    cursor.execute(search_query, (safe_query, k))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


def semantic_search_limited_scope(query, k=5, db_params=None):
    """
    Performs semantic search using vector similarity in PostgreSQL (pgvector),
    restricted to selected files only.
    
    Args:
        query (str): User's query string.
        k (int): Number of results to return.
        db_params (dict): DB connection parameters.

    Returns:
        list of tuples: (chunk_id, chunk_text, file_name, page_number, similarity_score).
    """
    # Generate the query embedding
    query_embedding_output = embedding_model.encode([query])
    query_embedding = np.array(query_embedding_output["dense_vecs"], dtype=np.float32).flatten()

    conn = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()

    # We do 1 - distance to transform L2 or cosine distance to similarity
    search_query = """
        SELECT 
            c.id, 
            c.chunk_text, 
            f.file_name,
            f.s3_link,
            c.page_number, 
            1 - (e.vector <=> %s) AS score
        FROM embeddings e
        JOIN chunks c ON e.chunks_id = c.id
        JOIN files f ON c.files_id = f.id
        JOIN selected_files_log sfl ON f.file_name = sfl.filename
        ORDER BY e.vector <=> %s
        LIMIT %s;
    """

    cursor.execute(search_query, (query_embedding, query_embedding, k))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results


def hybrid_search_limited_scope(query, k=5, lexical_k=20, semantic_k=20, db_params=None):
    """
    Performs a hybrid search that combines lexical (BM25-like) scoring and 
    semantic (pgvector) scoring, restricted to selected files only.
    
    Args:
        query (str): User's query string.
        k (int): Number of final results to return.
        lexical_k (int): Number of lexical candidates to retrieve prior to combining.
        semantic_k (int): Number of semantic candidates to retrieve prior to combining.
        db_params (dict): DB connection parameters.

    Returns:
        list of tuples: (id, chunk_text, file_name, page_number, combined_score).
    """
    # Generate the query embedding
    query_embedding_output = embedding_model.encode([query])
    query_embedding = np.array(query_embedding_output["dense_vecs"], dtype=np.float32).flatten()

    conn = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()

    # We'll do a 2-phase approach: gather top lexical and top semantic sets, 
    # rank them, then combine. Each side’s rank is turned into a small fraction 
    # so that better ranks produce bigger partial scores.

    # The <=> operator is the pgvector distance operator. 
    # paradedb.score(c.id) is the BM25-like function for lexical ranking.

    safe_query = escape_special_chars(query)

    search_query = """
        WITH bm25_candidates AS (
            SELECT c.id
            FROM chunks c
            JOIN files f ON c.files_id = f.id
            JOIN selected_files_log sfl ON f.file_name = sfl.filename
            WHERE c.chunk_text @@@ %s
            ORDER BY paradedb.score(c.id) DESC
            LIMIT %s
        ),
        bm25_ranked AS (
            SELECT 
                c.id, 
                RANK() OVER (ORDER BY paradedb.score(c.id) DESC) AS rank
            FROM bm25_candidates bc
            JOIN chunks c ON bc.id = c.id
        ),
        semantic_candidates AS (
            SELECT e.chunks_id AS id
            FROM embeddings e
            JOIN chunks c ON e.chunks_id = c.id
            JOIN files f ON c.files_id = f.id
            JOIN selected_files_log sfl ON f.file_name = sfl.filename
            ORDER BY e.vector <=> %s
            LIMIT %s
        ),
        semantic_ranked AS (
            SELECT 
                e.chunks_id AS id, 
                RANK() OVER (ORDER BY e.vector <=> %s) AS rank
            FROM semantic_candidates sc
            JOIN embeddings e ON sc.id = e.chunks_id
        )
        SELECT 
            COALESCE(sr.id, br.id) AS id,
            COALESCE(1.0 / (60 + sr.rank), 0) + COALESCE(1.0 / (60 + br.rank), 0) AS score,
            c.chunk_text,
            f.file_name,
            f.s3_link,
            c.page_number
        FROM semantic_ranked sr
        FULL OUTER JOIN bm25_ranked br ON sr.id = br.id
        JOIN chunks c ON c.id = COALESCE(sr.id, br.id)
        JOIN files f ON c.files_id = f.id
        ORDER BY score DESC, c.chunk_text
        LIMIT %s;
    """

    cursor.execute(search_query, (safe_query, lexical_k, query_embedding, semantic_k, query_embedding, k))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results