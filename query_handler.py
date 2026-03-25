import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
from pgvector.psycopg2 import register_vector
import re
import logfire

 # Configure logfire to log psycopg2 queries or individual SQL commands.
logfire.configure(
        send_to_logfire=False, 
        service_name="zrsvn-rag"
    )
logfire.instrument_psycopg()

from model_handling import embedding_model

def escape_special_chars(query):
    return re.sub(r'([^\w\s])', r'\\\1', query)

register_adapter(np.ndarray, lambda arr: AsIs(arr.tolist()))
register_adapter(np.float32, lambda val: AsIs(val))
register_adapter(np.float64, lambda val: AsIs(val))

 # BM25 lexical search using the ParadeDB operator @@@ over the content of all text_chunks that are part of files listed 
 # in the selected_files_log table.
def lexical_search_limited_scope(query, k=5, db_params=None):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    safe_q = escape_special_chars(query)

    sql = """
    SELECT *
    FROM (
    SELECT DISTINCT ON (tc.id)
      tc.id,
      -- BM25 result, generated based on the ParadeDB function.
      paradedb.score(tc.id)      AS score,
      -- Text from the text_chunks table.
      tc.text                     AS chunk_text,
      -- File name from the files table.
      f.filename,
      -- Path or key to the object (file) in S3 storage.
      f.s3_key                    AS s3_link,
      -- Page number where the text_chunk is located.
      se.page_nr                  AS page_number,
      -- Section summary to which the text_chunk belongs.
      s.summary                   AS section_summary,
      -- Summary of the entire file to which the text_chunk belongs.
      f.summary                   AS file_summary
    FROM rag_najdbe.text_chunks AS tc
    -- Povezava besedilnega bloka (ang. text chunk) s pripadajočim odstavkom prek prepared_text_id.
    JOIN rag_najdbe.paragraphs  AS p  ON p.prepared_text_id = tc.prepared_text_id
    -- Povezava odstavka z elementom odseka (odstavek, slika, tabela).
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    -- Povezava elementa odseka z odsekom.
    JOIN rag_najdbe.sections   AS s  ON s.id = se.section_id
    -- Povezava odseka z datoteko, katerega sestavni del je.
    JOIN rag_najdbe.files      AS f  ON f.id = s.file_id
    -- Limit results only to files that are currently selected.
    JOIN rag_najdbe.selected_files_log AS sfl
      ON f.id = sfl.file_id
    -- Lexical search over the content of text_chunks using the ParadeDB operator @@@.
    WHERE tc.text @@@ %s
    -- DISTINCT ON (tc.id) selects the first row for each tc.id according to the order specified in ORDER BY.
    -- Therefore, we must first sort the results by tc.id and then by search result (score) descending,
    -- to get the rows with the highest search result for each tc.id.
    ORDER BY tc.id, score DESC
    ) sub
    -- The main SELECT result of the inner subquery is sorted by relevance score in descending order.
    -- Without this secondary sorting, the user would receive results in arbitrary order or determined by tc.id, instead of being sorted by relevance.
    ORDER BY score DESC
    LIMIT %s;
    """

    cursor.execute(sql, (safe_q, k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def semantic_search_limited_scope(query, k=5, db_params=None):
    q_out = embedding_model.encode([query])
    q_vec = np.array(q_out["dense_vecs"], dtype=np.float32).flatten()

    conn   = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()

    sql = """
    SELECT *
    FROM (
    -- DISTINCT ON (tc.id) selects only one row for each tc.id, which prevents duplication of text blocks in the results.
    SELECT DISTINCT ON (tc.id)
      tc.id,
      -- The <=> operator returns the distance or cosine distance between vectors. Since a smaller result (0 = perfect match) indicates higher similarity, we subtract the distance from 1. This way, a better match is associated with a higher result value, which is more intuitive and consistent with the lexical search implementation.
      1 - (e.vector <=> %s)        AS score,
      tc.text                      AS chunk_text,
      f.filename,
      f.s3_key                     AS s3_link,
      se.page_nr                   AS page_number,
      s.summary                   AS section_summary,
      f.summary                   AS file_summary
    FROM rag_najdbe.embeddings   AS e
    -- Embeddings are linked to the corresponding text blocks (text_chunks).
    JOIN rag_najdbe.text_chunks  AS tc ON e.text_chunk_id = tc.id
    -- Text blocks with paragraphs.
    JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
    -- Paragraphs with section elements (paragraph, image, table).
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    -- Section elements with sections.
    JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
    -- Sections with files they are part of.
    JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
    -- Limit the selection only to files that are currently selected (i.e., present in the selected_files_log table).
    JOIN rag_najdbe.selected_files_log AS sfl
      ON f.id = sfl.file_id
    -- Consider only those vectors that are tied to text blocks.
    WHERE e.text_chunk_id IS NOT NULL
    -- First, sort the results by tc.id and then by score descending. This way, we find out which is the most relevant row for each text block.
    ORDER BY tc.id, score DESC
    ) sub
    -- The outer query sorts the results by score descending. This ensures that the user gets results sorted by relevance.
    ORDER BY score DESC
    LIMIT %s;
    """

    cursor.execute(sql, (q_vec, k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

 # Hybrid search: combination of BM25 and semantic search.
 # First, take the top lexical_k candidates based on the lexical search result,
 # then the top semantic_k, considering semantic similarity.
 # (Both searches are limited via selected_files_log).
 # Results are combined with weights (1/(60+rank)) and the top k are selected based on the combined result.
def hybrid_search_limited_scope(query, k=5, lexical_k=20, semantic_k=20, db_params=None):
    q_out = embedding_model.encode([query])
    q_vec = np.array(q_out["dense_vecs"], dtype=np.float32).flatten()
    safe_q = escape_special_chars(query)

    conn   = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()

    sql = """
    -- CTE (Common Table Expression or temporary query table) bm25_candidates finds the IDs
    -- of text blocks that are most relevant according to the BM25 lexical score.
    -- Results are limited to currently selected files via selected_files_log.
    WITH bm25_candidates AS (
      SELECT tc.id
      FROM rag_najdbe.text_chunks AS tc
      JOIN rag_najdbe.paragraphs  AS p  ON p.prepared_text_id = tc.prepared_text_id
      JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
      JOIN rag_najdbe.sections   AS s  ON s.id = se.section_id
      JOIN rag_najdbe.files      AS f  ON f.id = s.file_id
      JOIN rag_najdbe.selected_files_log AS sfl
        ON f.id = sfl.file_id
      WHERE tc.text @@@ %s
      ORDER BY paradedb.score(tc.id) DESC
      LIMIT %s
    ),
    -- CTE bm25_ranked assigns each candidate a rank according to its BM25 result.
    -- Lower rank means higher relevance.
    bm25_ranked AS (
      SELECT bc.id,
             RANK() OVER (ORDER BY paradedb.score(bc.id) DESC) AS rank
      FROM bm25_candidates bc
    ),
    -- CTE semantic_candidates finds the IDs of text blocks that are most semantically similar to the user's query.
    -- Results are limited to currently selected files via selected_files_log.
    semantic_candidates AS (
      SELECT e.text_chunk_id AS id
      FROM rag_najdbe.embeddings AS e
      JOIN rag_najdbe.text_chunks  AS tc ON e.text_chunk_id = tc.id
      JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
      JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
      JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
      JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
      JOIN rag_najdbe.selected_files_log AS sfl
        ON f.id = sfl.file_id
      -- only embeddings tied to text chunks
      WHERE e.text_chunk_id IS NOT NULL
      ORDER BY (1 - (e.vector <=> %s)) DESC
      LIMIT %s
    ),
    -- CTE semantic_ranked assigns each candidate a rank according to its semantic similarity score.
    -- Lower rank means higher relevance.
    semantic_ranked AS (
      SELECT sc.id,
             RANK() OVER (ORDER BY (1 - (e.vector <=> %s)) DESC) AS rank
      FROM semantic_candidates sc
      JOIN rag_najdbe.embeddings AS e
        ON e.text_chunk_id = sc.id
    )
    SELECT *
    FROM (
    -- Ensure that each ID appears only once in the result, even if it appears in both lists.
    SELECT DISTINCT ON (COALESCE(sr.id, br.id))
      COALESCE(sr.id, br.id)       AS id,
      -- By using the number 60 in the denominator, we reduce the impact of individual ranks on the final result or reduce the differences between ranked values.
      -- Here we use COALESCE to return 0 when an ID does not appear in one of the lists (thus preventing NULL as the score).
      COALESCE(1.0/(60+sr.rank),0) +
      COALESCE(1.0/(60+br.rank),0)  AS score,
      tc.text                       AS chunk_text,
      f.filename,
      f.s3_key                      AS s3_link,
      se.page_nr                    AS page_number,
      s.summary                   AS section_summary,
      f.summary                   AS file_summary
    FROM semantic_ranked sr
    -- Combine the ranked results of semantic and lexical search so that we get all unique IDs that appeared in at least one of the searches.
    FULL  OUTER JOIN bm25_ranked br ON sr.id = br.id
    JOIN rag_najdbe.text_chunks  AS tc ON tc.id = COALESCE(sr.id, br.id)
    JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
    JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
    -- Results are first sorted by ID and then by combined score.
    ORDER BY COALESCE (sr.id, br.id), score DESC
    ) AS sub
    -- The final output is sorted descending by combined score.
    ORDER BY score DESC
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