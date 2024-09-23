# This module should focus on data preprocessing tasks that are performed 
# before runtime. This includes reading files, cleaning and formatting 
# text, splitting text into chunks, generating embeddings, and saving 
# preprocessed data to disk.

import os
import fitz  # PyMuPDF for handling PDFs
from tqdm.auto import tqdm
import re
from sentence_transformers import SentenceTransformer
import faiss
import torch
import numpy as np
from rank_bm25 import BM25Okapi
import pickle

_global_embedding_model = None
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

def text_formatter(text: str, patterns: list[str]) -> str: 
    """Performs minor formatting on text."""
    #Uncomment to remove chapter names and page numbers at the beginning of the text
    #text = re.sub(r'^\s*(?:\d+\s*)?[A-Za-z\s:]+(?:\s*\d+)?\s*$', '', text, flags=re.MULTILINE).strip()
    combined_pattern = '|'.join([rf'{re.escape(p)}.*' for p in patterns])
    #example_pattern = r'Acknowledgments .*|Castelfranchi.*|Marsella.*'
    text = re.sub(combined_pattern, '', text, flags=re.DOTALL)
    cleaned_text = text.replace("\n", " ").strip()
    #Remove multiple spaces.
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    # Replace '- ' with ''
    cleaned_text = re.sub(r'-\s', '', cleaned_text)
    return cleaned_text

def open_and_read_pdf(
    pdf_path: str, 
    patterns: list[str]=['(?!).*']
) -> list[dict]:
    """
    Reads a pdf file and extracts text from each page, applying specified regex patterns to clean the text.
    
    Args:
        pdf_path (str): The path to the PDF file to be read.
        patterns (List[str]): A list of regex patterns to remove unwanted text. Each pattern will be applied 
                              to the text to remove matching parts.

    Returns:
    List[Dict]: A list of dictionaries, each containing information about a page in the PDF.
                The dictionary keys are:
                - 'page_number': The page number (starting from 1).
                - 'page_char_count': The number of characters on the page.
                - 'page_word_count': The number of words on the page.
                - 'page_sentence_count_raw': The number of sentences (based on periods).
                - 'page_token_count': The estimated number of tokens (1 token = ~4 characters).
                - 'page_text': The cleaned text of the page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = [] 
    for page_number, page in tqdm(enumerate(doc), total=len(doc), desc=f"Processing {os.path.basename(pdf_path)}"):
        text = page.get_text()
        text = text_formatter(text=text, patterns=patterns)
        if text.strip():  # Check if there is any meaningful content left
            pages_and_texts.append({
                "file_name": os.path.basename(pdf_path),
                "page_number": page_number + 1,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4, # 1 token = ~4 characters
                "page_text": text
            })
    return pages_and_texts

def open_and_read_txt(
    txt_path: str, 
    patterns: list[str]=['(?!).*'], 
    tokens_per_page: int = 500
) -> list[dict]:
    """Reads a TXT file and splits the text into pages based on the specified number of tokens per page."""
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = text_formatter(text=text, patterns=patterns)
    if text.strip():  # Check if there is any meaningful content left
        pages = split_text_into_pages(text, tokens_per_page)
        return [{
            "file_name": os.path.basename(txt_path),
            "page_number": page_number + 1,
            "page_char_count": len(page),
            "page_word_count": len(page.split(" ")),
            "page_sentence_count_raw": len(page.split(". ")),
            "page_token_count": len(page) / 4, # 1 token = ~4 characters
            "page_text": page
        } for page_number, page in enumerate(pages)]
    return []

def process_files_in_folder(
    folder_path: str, 
    patterns: list[str]=['(?!).*']
    ) -> list[dict]:
    """
    Processes all PDF and TXT files in a folder and combines the extracted text from all files.

    Args:
        folder_path (str): The path to the folder containing PDF files.
        patterns (List[str]): A list of regex patterns to remove unwanted text.

    Returns:
        List[Dict]: A combined list of dictionaries with extracted text information from all PDFs.
    """
    all_pages_and_texts = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.pdf'):
                pages_and_texts = open_and_read_pdf(file_path, patterns)
                all_pages_and_texts.extend(pages_and_texts)
            elif file.lower().endswith('.txt'):
                pages_and_texts = open_and_read_txt(file_path, patterns)
                all_pages_and_texts.extend(pages_and_texts)
    return all_pages_and_texts

def split_text_into_pages(
    text: str, 
    tokens_per_page: int
) -> list[str]:
    """
    Splits a text into pages based on a specified number of tokens (words in this context) per page.

    Args:
        text (str): The input text to be split.
        tokens_per_page (int): The number of tokens per page.

    Returns:
        list[str]: A list of text pages.
    """
    words = text.split()
    pages = []
    for i in range(0, len(words), tokens_per_page):
        page = ' '.join(words[i:i + tokens_per_page])
        pages.append(page)
    return pages

class CharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap=0):
        """
        Initializes the splitter with specified chunk size and overlap.

        Parameters:
        - chunk_size: The number of characters each chunk should contain.
        - chunk_overlap: The number of characters to overlap between adjacent chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        """
        Splits the given text into chunks according to the initialized chunk size and overlap.
        Note: The splitter is greedy in the sense that the chunk_overlap doesn't count into the chunk_size.

        Parameters:
        - text: The string of text to be split.

        Returns:
        A list of text chunks.
        """
        chunks = [] 
        start_index = 0 

        # Loop to split the text until the end of the text is reached.
        while start_index < len(text):
            end_index = start_index + self.chunk_size  # Calculate the end index for the current chunk.
            
            # Extend the end index to avoid cutting off a word in the middle.
            while end_index < len(text) and text[end_index] not in [' ']:
                end_index += 1

            # If it's not the first chunk and there are words to pass on, append them to the chunk list.
            if start_index != 0 and len(words_carried_on) != 0:
                chunks.append(' '.join(words_carried_on))

            current_chunk = text[start_index:end_index].strip() # Extract the current chunk and strip leading/trailing spaces.
            
            #For the first chunk, just append it. For subsequent chunks, add it to the last chunk with a space.
            if start_index == 0:
                chunks.append(current_chunk)
            else:
                chunks[-1] += f" {current_chunk}"

            # Prepare words to pass on to the next chunk, based on the overlap.
            words_carried_on = [] # List to store words that will be carried over to the next chunk.
            words = chunks[-1].split() # Split the last chunk into words.

            # Iterate over the words in reverse to determine which to carry over.
            words_len_sum = 0
            for word in reversed(words):
                words_carried_on.insert(0, word)
                words_len_sum += len(word) + 1
                if words_len_sum > self.chunk_overlap:
                    break

            # Update the start index for the next chunk.
            start_index = end_index

        return chunks
    
def make_pages_and_chunks(
        pages_and_texts: list[dict], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150
) -> list[dict]:
    """
    Creates chunks out of pages of text extracted from PDF and TXT files.

    Args:
        pages_and_texts (list[dict]): A list of dictionaries, each containing information about a page in the PDF or TXT.
                                      The dictionary keys should include:
                                      - 'file_name': The name of the file.
                                      - 'page_number': The page number (starting from 1).
                                      - 'page_text': The cleaned text of the page.
        chunk_size (int): The size of each chunk in characters. Default is 1000.
        chunk_overlap (int): The overlap between chunks in characters. Default is 150.

    Returns:
        list[dict]: A list of dictionaries, each containing information about a chunk of text.
                    The dictionary keys are:
                    - 'file_name': The name of the file.
                    - 'page_number': The page number (starting from 1).
                    - 'chunk_text': The text of the chunk.
                    - 'chunk_char_count': The number of characters in the chunk.
                    - 'chunk_word_count': The number of words in the chunk.
                    - 'chunk_token_count': The estimated number of tokens in the chunk.
    """
    pages_and_chunks = []
    
    # Define text splitter with the specified chunk size and overlap
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Iterate over each page's text to split into chunks
    for item in tqdm(pages_and_texts, desc="Creating chunks"):
        text = item['page_text']
        
        # Split the text into chunks
        chunks = splitter.split_text(text)
        
        # Iterate over each chunk to gather stats and store the result
        for chunk in chunks: 
            chunk_dict = {
                "file_name": item["file_name"],
                "page_number": item["page_number"],
                "chunk_text": chunk,
                "chunk_char_count": len(chunk),
                "chunk_word_count": len(chunk.split(" ")),
                "chunk_token_count": len(chunk) / 4  # 1 token is approximately 4 characters
            }
            
            # Append the chunk dictionary to the list
            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks

def init_embedding_model(model_name_or_path=EMBEDDING_MODEL_NAME):
    """
    Initializes and returns an embedding model using the SentenceTransformer class.

    Parameters:
    -----------
    model_name_or_path : str, optional
        The name or path of the pretrained model to load. Defaults to the value of EMBEDDING_MODEL_NAME.

    Returns:
    --------
    SentenceTransformer
        An instance of `SentenceTransformer` initialized with the specified model.

    Raises:
    -------
    ValueError
        If `model_name_or_path` is not a valid model for SentenceTransformer.
    """

    global _global_embedding_model
    
    if _global_embedding_model is not None and _global_embedding_model[0] == model_name_or_path:
        return _global_embedding_model[1]

    try:
        embedding_model = SentenceTransformer(model_name_or_path=model_name_or_path)
    except Exception as e:
        raise ValueError(f"The specified model_name_or_path '{model_name_or_path}' is not a valid model for SentenceTransformer. Error: {e}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model.to(device)

    # Store the initialized model in the global variable
    _global_embedding_model = (model_name_or_path, embedding_model)

    return embedding_model

def generate_embeddings(pages_and_chunks, model_name_or_path=EMBEDDING_MODEL_NAME, output_file="embeddings.index"):
    """
    Generates embeddings for the given text data and creates a FAISS index.

    Parameters:
    -----------
    pages_and_chunks : list of dict
        A list of dictionaries where each dictionary contains text data under the key "chunk_text".
    
    model_name_or_path : str, optional
        The name or path of the pretrained model to load. Defaults to the value of EMBEDDING_MODEL_NAME.

    output_file : str, optional
        The file path where the FAISS index will be saved. Defaults to "embeddings.index".

    Returns:
    --------
    faiss.Index
        The FAISS index containing the embeddings.
    """
    
    # Initialize the embedding model
    embedding_model = init_embedding_model(model_name_or_path)
    
    # Generate embeddings and store them in a list
    embeddings_list = []
    for item in tqdm(pages_and_chunks):
        item["embedding"] = embedding_model.encode(item["chunk_text"])
        embeddings_list.append(item["embedding"])
    
    # Convert list of embeddings to numpy array for FAISS
    embeddings_array = np.stack(embeddings_list).astype('float32')
    
    # Normalize embeddings because we are using dot product
    embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # Initialize FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to the FAISS index
    index.add(embeddings_array)
    
    # Save the index to a file
    faiss.write_index(index, output_file)
    
    return index

def preprocess_chunks_for_bm25(pages_and_chunks):
    """
    Preprocesses the chunks of text for BM25 by cleaning and tokenizing the text.
    
    Parameters:
    -----------
    pages_and_chunks : list of dict
        A list of dictionaries where each dictionary contains text data under the key "chunk_text".
    
    Returns:
    --------
    bm25_model : BM25Okapi
        The BM25 index for lexical search.
    
    tokenized_chunks : list of list of str
        Tokenized versions of the chunks ready to be used in BM25 search.
    """
    # Tokenize and clean each chunk text for BM25
    tokenized_chunks = [re.sub(r"[^a-zA-Z0-9]", " ", item["chunk_text"]).lower().split() for item in pages_and_chunks]
    
    # Create the BM25 model
    bm25_model = BM25Okapi(tokenized_chunks)
    
    return bm25_model, tokenized_chunks

def preprocess_and_save(
        folder_path, 
        patterns, 
        artifacts_dir="artifacts",
        output_embedding_file="embeddings.index", 
        output_bm25_file="bm25.pkl",
        pages_and_chunks_file="pages_and_chunks.pkl"
):
    """
    This function processes files, creates chunks, generates embeddings, and saves them to disk.
    It only runs if the preprocessed data doesn't already exist.
    """

    print("Entered preprocess_and_save...")

    # Ensure the artifacts directory exists
    if not os.path.exists(artifacts_dir):
        print("artifacts folder doesn't exist, creating it...")
        os.makedirs(artifacts_dir)

    # Construct full file paths
    output_embedding_file = os.path.join(artifacts_dir, output_embedding_file)
    output_bm25_file = os.path.join(artifacts_dir, output_bm25_file)
    pages_and_chunks_file = os.path.join(artifacts_dir, pages_and_chunks_file)

    # Check if embeddings and BM25 files already exist
    if not os.path.exists(output_embedding_file) or not os.path.exists(output_bm25_file) or not os.path.exists(pages_and_chunks_file):
        print("Preprocessed data not found. Running preprocessing...")

        # Process files
        pages_and_texts = process_files_in_folder(folder_path, patterns)
        pages_and_chunks = make_pages_and_chunks(pages_and_texts)

        # Save pages_and_chunks to disk
        with open(pages_and_chunks_file, 'wb') as f:
            pickle.dump(pages_and_chunks, f)

        # Generate and save embeddings (FAISS index)
        print("Generating embeddings...")
        index = generate_embeddings(pages_and_chunks, output_file=output_embedding_file)

        # Generate and save BM25 index
        bm25_model, tokenized_chunks = preprocess_chunks_for_bm25(pages_and_chunks)
        with open(output_bm25_file, 'wb') as bm25_file:
            pickle.dump((bm25_model, tokenized_chunks), bm25_file)

    else:
        print("Preprocessed data already exists. Loading from disk...")

        # Load pages_and_chunks from disk
        with open(pages_and_chunks_file, 'rb') as f:
            pages_and_chunks = pickle.load(f)
        # Load the FAISS index
        index = faiss.read_index(output_embedding_file)

        # Load the BM25 model and tokenized chunks
        with open(output_bm25_file, 'rb') as bm25_file:
            bm25_model, tokenized_chunks = pickle.load(bm25_file)

    return index, bm25_model, tokenized_chunks, pages_and_chunks