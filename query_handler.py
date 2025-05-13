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
    BM25-like lexical search over text_chunks → paragraphs → section_elements → sections → files,
    restricted to selected_files_log.
    Returns tuples: (chunk_id, chunk_text, filename, page_number, score).
    """
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    safe_q = escape_special_chars(query)

    ###YOUR TASK### Change this to return also
    # sections.summary (renamed AS section_summary) 
    # and files.summary (renamed AS file_summary)
    sql = """
    SELECT
      tc.id,
      tc.text                     AS chunk_text,
      f.filename,
      f.s3_key                    AS s3_link,
      se.page_nr                  AS page_number,
      s.summary                   AS section_summary,
      f.summary                   AS file_summary,
      paradedb.score(tc.id)      AS score
    FROM rag_najdbe.text_chunks AS tc
    JOIN rag_najdbe.paragraphs  AS p  ON p.prepared_text_id = tc.prepared_text_id
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    JOIN rag_najdbe.sections   AS s  ON s.id = se.section_id
    JOIN rag_najdbe.files      AS f  ON f.id = s.file_id
    JOIN rag_najdbe.selected_files_log AS sfl
      ON f.filename = sfl.filename
    WHERE tc.text @@@ %s
    ORDER BY score DESC
    LIMIT %s;
    """

    cursor.execute(sql, (safe_q, k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def semantic_search_limited_scope(query, k=5, db_params=None):
    """
    Semantic search via pgvector over embeddings → text_chunks → … → files,
    restricted to selected_files_log.
    Returns tuples: (chunk_id, chunk_text, filename, page_number, similarity_score).
    """
 # encode
    q_out = embedding_model.encode([query])
    q_vec = np.array(q_out["dense_vecs"], dtype=np.float32).flatten()

    conn   = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()


    ###YOUR TASK### Change this to return also
    # sections.summary (renamed AS section_summary) 
    # and files.summary (renamed AS file_summary)
    sql = """
    SELECT
      tc.id,
      tc.text                      AS chunk_text,
      f.filename,
      f.s3_key                     AS s3_link,
      se.page_nr                   AS page_number,
      s.summary                   AS section_summary,
      f.summary                   AS file_summary,
      1 - (e.vector <=> %s)        AS score
    FROM rag_najdbe.embeddings   AS e
    JOIN rag_najdbe.text_chunks  AS tc ON e.text_chunk_id = tc.id
    JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
    JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
    JOIN rag_najdbe.selected_files_log AS sfl
      ON f.filename = sfl.filename
    -- only embeddings tied to text chunks
    WHERE e.text_chunk_id IS NOT NULL
    ORDER BY e.vector <=> %s
    LIMIT %s;
    """

    cursor.execute(sql, (q_vec, q_vec, k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def hybrid_search_limited_scope(query, k=5, lexical_k=20, semantic_k=20, db_params=None):
    """
    Hybrid search combining BM25 and pgvector scores over the same join path,
    restricted to selected_files_log.
    Returns tuples: (chunk_id, chunk_text, filename, page_number, combined_score).
    """
    # encode
    q_out = embedding_model.encode([query])
    q_vec = np.array(q_out["dense_vecs"], dtype=np.float32).flatten()
    safe_q = escape_special_chars(query)

    conn   = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()


    ###YOUR TASK### Change this to return also
    # sections.summary (renamed AS section_summary) 
    # and files.summary (renamed AS file_summary)
    sql = """
    WITH bm25_candidates AS (
      SELECT tc.id
      FROM rag_najdbe.text_chunks AS tc
      JOIN rag_najdbe.paragraphs  AS p  ON p.prepared_text_id = tc.prepared_text_id
      JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
      JOIN rag_najdbe.sections   AS s  ON s.id = se.section_id
      JOIN rag_najdbe.files      AS f  ON f.id = s.file_id
      JOIN rag_najdbe.selected_files_log AS sfl
        ON f.filename = sfl.filename
      WHERE tc.text @@@ %s
      ORDER BY paradedb.score(tc.id) DESC
      LIMIT %s
    ),
    bm25_ranked AS (
      SELECT bc.id,
             RANK() OVER (ORDER BY paradedb.score(bc.id) DESC) AS rank
      FROM bm25_candidates bc
    ),
    semantic_candidates AS (
      SELECT e.text_chunk_id AS id
      FROM rag_najdbe.embeddings AS e
      JOIN rag_najdbe.text_chunks  AS tc ON e.text_chunk_id = tc.id
      JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
      JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
      JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
      JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
      JOIN rag_najdbe.selected_files_log AS sfl
        ON f.filename = sfl.filename
      -- only embeddings tied to text chunks
      WHERE e.text_chunk_id IS NOT NULL
      ORDER BY e.vector <=> %s
      LIMIT %s
    ),
    semantic_ranked AS (
      SELECT sc.id,
             RANK() OVER (ORDER BY e.vector <=> %s) AS rank
      FROM semantic_candidates sc
      JOIN rag_najdbe.embeddings AS e
        ON e.text_chunk_id = sc.id
    )
    SELECT
      COALESCE(sr.id, br.id)       AS id,
      COALESCE(1.0/(60+sr.rank),0) +
      COALESCE(1.0/(60+br.rank),0)  AS score,
      tc.text                       AS chunk_text,
      f.filename,
      f.s3_key                      AS s3_link,
      se.page_nr                    AS page_number,
      s.summary                   AS section_summary,
      f.summary                   AS file_summary
    FROM semantic_ranked sr
    FULL  OUTER JOIN bm25_ranked br ON sr.id = br.id
    JOIN rag_najdbe.text_chunks  AS tc ON tc.id = COALESCE(sr.id, br.id)
    JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
    JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
    ORDER BY score DESC, tc.text
    LIMIT %s;
    """

    cursor.execute(sql, (
      safe_q, lexical_k,
      q_vec, semantic_k,
      q_vec, k
    ))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results