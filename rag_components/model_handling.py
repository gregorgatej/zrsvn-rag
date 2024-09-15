from enum import Enum
import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
import torch

# Enum for model handling
class LLM(Enum):
    GPT_4O_MINI = ("openai", "gpt-4o-mini")
    GPT_35_TURBO = ("openai", "gpt-3.5-turbo")
    GEMMA_7B_IT = ("hugging_face", "google/gemma-7b-it")
    LLAMA_2_7B_CHAT = ("hugging_face", "meta-llama/Llama-2-7b-chat-hf")
    MISTRAL_7B_INSTRUCT = ("hugging_face", "mistralai/Mistral-7B-Instruct-v0.1")
    LOCAL_LLM_SERVER = ("local", "llama-2-7b-local")

    def get_provider(self):
        return self.value[0]

    def get_model_name(self):
        return self.value[1]

# Model initialization for Hugging Face
hugging_face_tokenizer = None
hugging_face_model = None
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

def initialize_local_llm(model_id):
    global hugging_face_tokenizer, hugging_face_model

    #TODO: Optimize this part of code, returning None probably not the best option
    # Check if it's the local LLM server case
    if model_id == "llama-2-7b-local":
        # In the case of the local model, we do not need to initialize a model/tokenizer locally
        # Instead, we'll return None and rely on a separate function to make HTTP requests
        return None, None

    if hugging_face_tokenizer is not None and hugging_face_model is not None:
        return hugging_face_tokenizer, hugging_face_model

    use_quantization_config = True

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    attn_implementation = "flash_attention_2" if (is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8) else "sdpa"
    print(f"using attention implementation: {attn_implementation}") 

    hugging_face_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, use_auth_token=hf_token)
    hugging_face_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype=torch.float16, quantization_config=quantization_config if use_quantization_config else None, low_cpu_mem_usage=False, attn_implementation=attn_implementation, use_auth_token=hf_token)

    if not use_quantization_config:
        hugging_face_model.to("cuda")

    return hugging_face_tokenizer, hugging_face_model

def query_local_llm_server(query: str, server_url: str = "http://192.168.20.220:5000/generate"):
    """
    Send a query to the local LLM server and return the response.
    """
    try:
        response = requests.post(server_url, json={"input_text": query})
        response.raise_for_status()
        return response.json().get("answer", "Error: No response from server")
    except requests.exceptions.RequestException as e:
        return f"Error querying local LLM server: {e}"