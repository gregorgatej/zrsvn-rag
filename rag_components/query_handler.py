# This module handles tasks that occur at runtime when a user 
# query is processed. This includes retrieving relevant resources, 
# performing searches, reranking documents, formatting prompts, and 
# generating responses.

import os
from time import perf_counter as timer
from .model_handling import LLM, initialize_local_llm, query_local_llm_server
from scripts.preprocess import init_embedding_model
import openai
from openai import AzureOpenAI
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

openai.api_key = os.getenv("OPENAI_API_KEY")

def retrieve_relevant_resources(
    query: str,
    index: faiss.Index = None,  # FAISS index for semantic search, TODO: needs to be passed
    bm25_model = None,  # BM25 model for lexical search, TODO: needs to be passed
    num_resources_to_return: int = 5,
    #TODO: check if embedding_model is needed here, if yes, probably needs
    #to get passed into generate_response
    embedding_model: SentenceTransformer = None,
    use_lexical_search: bool = False,
    tokenized_chunks: list[list[str]] = None,  # Preprocessed chunks for BM25, TODO: needs to be passed
    print_time: bool = False
):
    """
    Retrieves top resources based on the query using either semantic (FAISS) or lexical (BM25) search.
    """
    if use_lexical_search:
        # Perform lexical search using BM25
        if bm25_model is None or tokenized_chunks is None:
            raise ValueError("BM25 model and tokenized chunks are required for lexical search.")
        
        scores, indices = lexical_search(query=query, bm25_model=bm25_model, tokenized_chunks=tokenized_chunks, k=num_resources_to_return)

    else:
        # Perform semantic search using FAISS
        if embedding_model is None:
            embedding_model = init_embedding_model()

        # Embed the query
        query_embedding = embedding_model.encode(query, convert_to_tensor=False).astype('float32').reshape(1, -1)

        # Normalize query embedding if using dot product
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Get FAISS search results
        start_time = timer()
        distances, indices = index.search(query_embedding, num_resources_to_return)
        end_time = timer()

        if print_time:
            print(f"[INFO] Time taken to get scores on {index.ntotal} embeddings: {end_time - start_time:.5f} seconds.")

        scores = distances[0]

    return scores, indices

def prompt_formatter(
    query: str, 
    context_items: list[dict], 
    llm: str, 
    tokenizer=None
) -> str:
    
    context = "- " + "\n- ".join([item["chunk_text"] for item in context_items])

    base_prompt = """with your general knowledge and with the help of the following context items, please answer the query.
give yourself room to think by extracting relevant passages from the context before answering the query.
don't return the thinking, only return the answer.
make sure your answers are as explanatory as possible.

\ncontext items:
{context}
\nrelevant passages: <extract relevant passages from the context (if any) here>
user query: {query}
answer:""" 
    base_prompt = base_prompt.format(context=context, query=query)

    # create prompt template for instruction-tuned model 
    dialogue_template = [
        {"role": "user", "content": base_prompt}
    ]

    if llm == "hugging_face":
        # apply the chat template for hugging face
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,
                                               add_generation_prompt=True)
    else:
        # manually format the prompt for openai
        prompt = dialogue_template[0]["content"]

    return prompt

def generate_response(
    llm: LLM, 
    query: str,
    pages_and_chunks: list[dict],
    #TODO: Add types
    tokenized_chunks,
    bm25_model,
    index,
    system_content: str="You are a helpful assistant",
    temperature: float=0.7, 
    max_new_tokens: int=256,
    num_context_items: int=5,
    use_lexical_search: bool = False,
    use_reranking: bool = False,
    format_response_text=True, 
):
    """
    takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.

    nr_context_items: determines the number of context items that are taken in account in the generation of answers.
    If set to 0 the query is used directly, without any additional context.
    format_response_text: determines whether the output text should be cleaned by removing the prompt and special tokens 
    like <bos> (beginning of sequence) and <eos> (end of sequence). Set to True to clean the output text and ensure it 
    doesn't include these control tokens, or False to return the raw model output.
    """

    print("Entered generate_response...")

    provider = llm.get_provider()
    model_name = llm.get_model_name()

    if provider == "hugging_face":
        tokenizer, llm = initialize_local_llm(model_name)
    elif provider == "azure":
        azure_oai_key = os.getenv("AZURE_OAI_KEY")
        azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
        client = AzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version="2024-02-15-preview"
        )
    else:
        client = openai.OpenAI(api_key=openai.api_key)

    if num_context_items > 0:
        # retrieval
        # get just the scores and indices of top related results
        scores, indices = retrieve_relevant_resources(
            query=query, 
            index=index,
            bm25_model=bm25_model,
            tokenized_chunks=tokenized_chunks, 
            num_resources_to_return=num_context_items, 
            use_lexical_search=use_lexical_search
        )

        # ENSURE INDICES ARE FLATTENED AND CONVERTED TO A LIST OF INTEGERS
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()  # FLATTEN AND CONVERT TO LIST
        elif isinstance(indices[0], list):
            indices = [i for sublist in indices for i in sublist]  # FLATTEN LIST OF LISTS

        if use_reranking:
            context_items = rerank_documents(query, indices, pages_and_chunks)
        else:
            # Use context items directly without reranking
            context_items = [pages_and_chunks[i] for i in indices]

        # add score to context item
        #Note: score gets added to the original pages_and_chunks entry!
        for i, item in enumerate(context_items): 
            item["score"] = float(scores[i])  # convert score to float directly if openai, else convert to tensor

        # augmentation
        # create the prompt and format it with context items
        if provider == "hugging_face":
            prompt = prompt_formatter(query=query, context_items=context_items, llm=provider, tokenizer=tokenizer)
        else:
            prompt = prompt_formatter(query=query, context_items=context_items, llm=provider)

    else:
        # If no context items are requested, use the query directly as the prompt
        prompt = query
    
    if provider == "hugging_face":
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = llm.generate(**input_ids,
                                     temperature=temperature,
                                     do_sample=True,
                                     max_new_tokens=max_new_tokens)
        output_text = tokenizer.decode(outputs[0])
    elif provider == "local":
        output_text = query_local_llm_server(query)
    #The same if provider == "azure" or "openai"
    else:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        output_text = response.choices[0].message.content  # accessed response content

    if format_response_text:
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "")

    if num_context_items > 0:
        final_output = {
            "question": query,
            "answer": output_text,
            "llm": model_name,
            "context_source": list(set([item["file_name"] for item in context_items])),
            "context_content": list(set([item["chunk_text"] for item in context_items]))
        }
    else:
        final_output = {
            "question": query,
            "answer": output_text,
            "llm": model_name,
            "context_source": None,
            "context_content": None
        }

    return final_output

def format_response(
        response: dict, 
        return_answer_only=True
) -> str:
    if return_answer_only:
        return response['answer']
    else:
        return response

def lexical_search(
        query: str, 
        bm25_model, 
        tokenized_chunks: list[list[str]], 
        k: int
):
    """
    Performs lexical search using BM25 on the given query and returns the top k results.
    """
    # Preprocess the query (tokenization, lowercasing, etc.)
    query_tokens = query.lower().split()

    # Get BM25 scores for each chunk based on the query
    scores = bm25_model.get_scores(query_tokens)

    # Sort the scores and get top k indices
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    # Return the indices and corresponding scores
    return [scores[i] for i in indices], indices

# CrossEncoder for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(
        query, 
        indices, 
        pages_and_chunks
):
    """
    Prepares context items, reranks them using CrossEncoder, and returns reranked context items.
    
    Parameters:
    - query: The user query.
    - indices: The indices of the documents retrieved by FAISS or BM25.
    - pages_and_chunks: The original documents/chunks from which context is drawn.
    
    Returns:
    - reranked_context_items: The context items sorted by relevance after reranking.
    """

    # Prepare context items for reranking
    context_items = [pages_and_chunks[i] for i in indices]
    document_texts = [item['chunk_text'] for item in context_items]

    # CrossEncoder reranking
    pairs = [[query, doc_text] for doc_text in document_texts]  # Query-document pairs
    rerank_scores = cross_encoder.predict(pairs)

    # Sort context items based on rerank scores
    reranked_items = sorted(zip(rerank_scores, context_items), key=lambda x: x[0], reverse=True)
    reranked_context_items = [item for _, item in reranked_items]

    return reranked_context_items