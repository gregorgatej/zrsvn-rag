
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

app = Flask(__name__)

# Model ID for the Llama-2-7b-chat-hf with 4-bit quantization
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Set the compute type to FP16 to match the input data type
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Use FP16 for computations
)

# Load the model with 4-bit precision and move it to GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"  # Automatically map model layers to GPU if available
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('input_text', '')

    # Tokenize the input and move it to the GPU
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')

    # Generate a response using the model
    with torch.no_grad():  # Avoids tracking gradients since we're only doing inference
        output = model.generate(
            inputs['input_ids'],
            max_length=1000,  # You can adjust this based on your needs
            no_repeat_ngram_size=3
        )

    # Decode the generated response
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'answer': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
