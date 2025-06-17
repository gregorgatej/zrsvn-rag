# ZRSVN RAG

Spletna aplikacija z možnostjo leksičnega, semantičnega in hibridnega iskanja ter generiranja odgovorov na podlagi izbranih dokumentov.

## O projektu

Sistem implementira RAG (Retrieval Augmented Generation) pristop za iskanje po dokumentih, ki vsebuje naslednje funkcionalnosti:

- **Geografsko filtriranje**: Interaktivni zemljevid za prostorsko zamejitev iskanja.
- **Večvrsten način iskanja**: Leksično, semantično in hibridno iskanje  
- **Pogovorni vmesnik**: Gradio ChatInterface z Azure OpenAI integracijo.  
- **Vektorsko iskanje**: pgvector podpora za semantično podobnost.  

## Funkcionalnosti

- Interaktivni Leaflet zemljevid za risanje robnih okvirjev.
- Trije načini iskanja: leksični (BM25), semantični (vektorski) in hibridni (RRF).  
- Pogovorni vmesnik prek Azure OpenAI (privzeto je izbran GPT-4o-mini). 
- Generiranje presigned URLjev za pregled kontekstualnih PDF dokumentov.  
- Beleženje povratnih informacij uporabnikov.  
- Observability z Logfire integracijo.

## Tehnične zahteve

- Python 3.8+.  
- PostgreSQL z razširitvama pgvector in ParadeDB. 
- MinIO/S3 shramba za PDF dokumente.  
- Azure OpenAI API dostop.  

## Namestitev

1. Klonirajte repozitorij:
    ```bash
    git clone https://github.com/gregorgatej/zrsvn-rag.git
    cd zrsvn-rag
    ```

2. Namestite odvisnosti:
    ```bash
    pip install -r requirements.txt
    ```

3. Ustvarite `.env` datoteko z naslednjimi spremenljivkami:
    ```env
    POSTGRES_PASSWORD=tvoj_postgres_password
    S3_ACCESS_KEY=tvoj_s3_access_key
    S3_SECRET_ACCESS_KEY=tvoj_s3_secret_key
    ZRSVN_AZURE_OPENAI_ENDPOINT=tvoj_azure_endpoint
    ZRSVN_AZURE_OPENAI_KEY=tvoj_azure_openai_key
    ```

4. Pripravite PostgreSQL bazo:
    - Ustvarite bazo `zrsvn` s shemo `rag_najdbe`.
    - Namestite razširitvi `pgvector` in `paradedb`.

## Zagon aplikacije

```bash
python app.py
