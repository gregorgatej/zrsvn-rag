import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
# Podpora za pgvector.
from pgvector.psycopg2 import register_vector
import re
import logfire

# Konfiguriramo logfire, da beleži psycopg2 poizvedbe oz. posamične SQL ukaze.
logfire.configure(
        send_to_logfire=False, 
        service_name="zrsvn-rag"
    )
logfire.instrument_psycopg()

# Model (BGE-M3) za generiranje vložitev (ang. embeddings).
from model_handling import embedding_model

# Posebnim znakom (tj. vsem ne-alfanumeričnim znakom) v poizvedbi se izognemo (tj. pred znak dodamo '\'), da preprečimo 
# ParseError pri leksičnem ali hibridnem iskanju.
def escape_special_chars(query):
    """Escapes special characters for full-text search in ParadeDB/PostgreSQL so we don't receive ParseError when running lexical or hybrid search."""
    return re.sub(r'([^\w\s])', r'\\\1', query)  # Escape all non-alphanumeric characters

# Register numpy adapters so we can easily store/fetch arrays
register_adapter(np.ndarray, lambda arr: AsIs(arr.tolist()))
register_adapter(np.float32, lambda val: AsIs(val))
register_adapter(np.float64, lambda val: AsIs(val))

# BM25 leksično iskanje s ParadeDB operatorjem @@@ prek vsebine vseh text_chunks, ki so del datotek, navedenih
# v tabeli selected_files_log. 
# Vrne nabor vrednosti:
# (chunk_id, chunk_text, filename, s3_link, page_number, score, section_summary, file_summary).
# Parametri:
  # - query: besedilo poizvedbe.
  # - k: največje število vrnjenih rezultatov.
  # - db_params: slovar za psycopg2.connect (dbname, user, password, host, port, options).
def lexical_search_limited_scope(query, k=5, db_params=None):
    """
    BM25-like lexical search over text_chunks → paragraphs → section_elements → sections → files,
    restricted to selected_files_log.
    Returns tuples: (chunk_id, chunk_text, filename, page_number, score).
    """
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    safe_q = escape_special_chars(query)

    sql = """
    SELECT *
    FROM (
    SELECT DISTINCT ON (tc.id)
      tc.id,
      -- BM25 rezultat, generiran na osnovi ParadeDB funkcije.
      paradedb.score(tc.id)      AS score,
      -- Besedilo iz tabele text_chunks.
      tc.text                     AS chunk_text,
      -- Ime datoteke iz tabele files.
      f.filename,
      -- Pot ali ključ do objekta (datoteke) v S3 shrambi.
      f.s3_key                    AS s3_link,
      -- Številka strani, kjer se text_chunk nahaja.
      se.page_nr                  AS page_number,
      -- Povzetek odseka v katerega text_chunk spada.
      s.summary                   AS section_summary,
      -- Povzetek celotne datoteke v katero spada text_chunk.
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
    -- Omejitev rezultatov samo na datoteke, ki so trenutno izbrane.
    JOIN rag_najdbe.selected_files_log AS sfl
      ON f.id = sfl.file_id
    -- Leksično iskanje po vsebini text_chunks z uporabo ParadeDB operatorja @@@.
    WHERE tc.text @@@ %s
    -- DISTINCT ON (tc.id) izbere prvo vrstico za vsak tc.id glede na vrstni red podan v ORDER BY.
    -- Zato moramo rezulte najprej razvrstiti po tc.id in nato po rezultatu iskanja (score) padajoče,
    -- da pridemo do vrstic z najvišjim rezultatom iskanja za vsak tc.id.
    ORDER BY tc.id, score DESC
    ) sub
    -- Glavni SELECT rezultat notranje poizvedbe (subquery) uredi glede na oceno relevantnosti (score), v 
    -- padajočem vrstem redu.
    -- Brez tega sekundarnega urejanja rezultatov bi uporabnik prejel rezultate v poljubnem vrstem redu oz.
    -- določene glede na tc.id, namesto da bi bili sortirani po relevantnosti.
    ORDER BY score DESC
    -- Število vrnjenih rezultatov omejimo glede na podano vrednost.
    LIMIT %s;
    """

    cursor.execute(sql, (safe_q, k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

# Semantično iskanje s pomočjo razširitve pgvector.
# Vrne nabor vrednosti:
# (chunk_id, score, chunk_text, filename, s3_link, page_number, section_summary, file_summary).
# Parametri:
  # - query: besedilo poizvedbe.
  # - k: največje število vrnjenih rezultatov.
  # - db_params: slovar za psycopg2.connect (dbname, user, password, host, port, options).
def semantic_search_limited_scope(query, k=5, db_params=None):
    """
    Semantic search via pgvector over embeddings → text_chunks → … → files,
    restricted to selected_files_log.
    Returns tuples: (chunk_id, chunk_text, filename, page_number, similarity_score).
    """
    # Najprej na podlagi poizvedbe generiramo vložitev.
    q_out = embedding_model.encode([query])
    q_vec = np.array(q_out["dense_vecs"], dtype=np.float32).flatten()

    conn   = psycopg2.connect(**db_params)
    register_vector(conn)
    cursor = conn.cursor()

    sql = """
    SELECT *
    FROM (
    -- DISTINCT ON (tc.id) izbere samo eno vrstico za vsak tc.id, kar prepreči podvajanje 
    -- tekstovnih blokov v rezultatih.
    SELECT DISTINCT ON (tc.id)
      tc.id,
      -- Operator <=> vrača razdaljo oz. kosinusno razdaljo med vektorjema. Ker manjši rezultat (0 = popolno ujemanje) 
      -- izraža višjo podobnost, razdaljo odštejemo od 1. S tem povzročimo, da je z boljšim ujemanjem povezana višja 
      -- vrednost rezultata, kar
      -- deluje bolj intuitivno in je v skladu z implementacijo pri leksičnem iskanju. 
      1 - (e.vector <=> %s)        AS score,
      tc.text                      AS chunk_text,
      f.filename,
      f.s3_key                     AS s3_link,
      se.page_nr                   AS page_number,
      s.summary                   AS section_summary,
      f.summary                   AS file_summary
    FROM rag_najdbe.embeddings   AS e
    -- Vložitve povežemo s pripadajočimi tekstovnimi bloki (text_chunks).
    JOIN rag_najdbe.text_chunks  AS tc ON e.text_chunk_id = tc.id
    -- Tekstovne bloke z odstavki (paragraphs).
    JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
    -- Odstavke z elementi odseka (odstavek, slika, tabela).
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    -- Elemente odseka z odseki.
    JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
    -- Odseke z datotekami katerih del so.
    JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
    -- Izbor omejimo samo na datoteke, ki so trenutno izbrane (tj. se nahajajo v
    -- tabeli selected_files_log).
    JOIN rag_najdbe.selected_files_log AS sfl
      ON f.id = sfl.file_id
    -- Upoštevamo samo tiste vektorje, ki so vezani na tekstovne bloke.
    WHERE e.text_chunk_id IS NOT NULL
    -- Rezultate najprej uredimo po tc.id in nato po oceni (score) padajoče. 
    -- Na ta način izvemo katera je najbolj relevantna vrstica za vsak tekstovni blok.
    ORDER BY tc.id, score DESC
    ) sub
     -- Zunanja poizvedba rezultate uredi po oceni (score) padajoče.
    -- To zagotovi, da uporabnik dobi rezultate, ki so sortirani po relevantnosti.
    ORDER BY score DESC
    LIMIT %s;
    """

    cursor.execute(sql, (q_vec, k))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

# Hibridno iskanje: kombinacija BM25 in semantičnega iskanja. 
# Najprej vzamemo najvišje uvrščene lexical_k kandidate glede na rezultat leksičnega iskanja, 
# nato top semantic_k, upoštevajoč semantično podobnost.
# (Obe iskanje sta omejeni prek selected_files_log).
# Rezultate združimo z utežmi (1/(60+rank)) in izberemo top k glede na kombiniran rezultat.
# Vrne nabor vrednosti:
# (chunk_id, score, chunk_text, filename, s3_link, page_number, section_summary, file_summary).
# Parametri:
  # - query: besedilo poizvedbe.
  # - k: koliko rezultatov vrnemo v končni fazi.
  # - lexical_k: koliko je kandidatov, pridobljenih na podlagi leksičnega iskanja.
  # - semantic_k: koliko je kandidatov, pridobljenih na podlagi semantičnega iskanja.
  # - db_params: slovar za psycopg2.connect (dbname, user, password, host, port, options).
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

    sql = """
    -- CTE (Common Table Expression oz. začasna poizvedbena tabela) bm25_candidates poišče IDje 
    -- tekstovnih blokov, ki so najbolj relevantni glede na BM25 leksično oceno.
    -- Rezultate prek selected_files_log omejimo na trenutno izbrane datoteke.
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
    -- CTE bm25_ranked dodeli vsakemu kandidatu rang glede na njegov BM25 rezultat.
    -- Nižji rang pomeni višjo relevantnost.
    bm25_ranked AS (
      SELECT bc.id,
             RANK() OVER (ORDER BY paradedb.score(bc.id) DESC) AS rank
      FROM bm25_candidates bc
    ),
    -- CTE semantic_candidates poišče IDje tekstovnih blokov, ki so si z uporabnikovo poizvedbo
    -- najbolj semantično podobni.
    -- Rezultate prek selected_files_log omejimo na trenutno izbrane datoteke.
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
    -- CTE semantic_ranked dodeli vsakemu kandidatu rang glede na njegov rezultat semantične podobnosti.
    -- Nižji rang pomeni višjo relevantnost.
    semantic_ranked AS (
      SELECT sc.id,
             RANK() OVER (ORDER BY (1 - (e.vector <=> %s)) DESC) AS rank
      FROM semantic_candidates sc
      JOIN rag_najdbe.embeddings AS e
        ON e.text_chunk_id = sc.id
    )
    SELECT *
    FROM (
    -- Zagotovimo, da se vsak ID pojavi le enkrat v rezultatu, tudi če se pojavi v obeh seznamih.
    SELECT DISTINCT ON (COALESCE(sr.id, br.id))
      COALESCE(sr.id, br.id)       AS id,
      -- Z uporaba števila 60 v imenovalcu zmanjšamo vpliv posameznega ranga na končni
      -- rezultat oz. zmanjšamo razlike med rangiranimi vrednostmi.
      -- Tu uporabimo COALESCE za to, da vrnemo 0, ko se nek ID pojavi ne pojavi v enem od
      -- seznamov (in s tem preprečimo, da bi kot score dobili NULL). 
      COALESCE(1.0/(60+sr.rank),0) +
      COALESCE(1.0/(60+br.rank),0)  AS score,
      tc.text                       AS chunk_text,
      f.filename,
      f.s3_key                      AS s3_link,
      se.page_nr                    AS page_number,
      s.summary                   AS section_summary,
      f.summary                   AS file_summary
    FROM semantic_ranked sr
    -- Rangirane rezultate semantičnega in leksičnega iskanja združimo na način, da
    -- dobimo vse unikatne IDje, ki so se pojavili v vsaj enem od iskanj.
    FULL  OUTER JOIN bm25_ranked br ON sr.id = br.id
    JOIN rag_najdbe.text_chunks  AS tc ON tc.id = COALESCE(sr.id, br.id)
    JOIN rag_najdbe.paragraphs   AS p  ON p.prepared_text_id = tc.prepared_text_id
    JOIN rag_najdbe.section_elements AS se ON se.id = p.section_element_id
    JOIN rag_najdbe.sections    AS s  ON s.id = se.section_id
    JOIN rag_najdbe.files       AS f  ON f.id = s.file_id
    -- Rezultati so tu najprej urejeni po IDju in nato po skupnem rezultatu.
    ORDER BY COALESCE (sr.id, br.id), score DESC
    ) AS sub
    -- Končni izpis uredimo padajoče glede na skupni rezultat.
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

    # TODO: FULL OUTER JOIN in RANK() se lahko izkažeta za počasni pri večjih tabelah.
    #    - Optimizirati s predhodno shranjenimi rang-kandidatnimi tabelami.