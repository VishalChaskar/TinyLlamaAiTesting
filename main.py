from llama_cpp import Llama
import os

# Define the model path
MODEL_DIR = "models"
MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Load TinyLLaMA
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024, # Context window size
        n_threads=4, # Number of CPU threads to use
        verbose=False # Set to True for more detailed logging from llama.cpp
    )
    print(f"Successfully loaded TinyLLaMA model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading TinyLLaMA model: {e}")
    print(f"Please ensure the model file '{MODEL_NAME}' is in the '{MODEL_DIR}' directory.")
    exit()

print("\n--- TinyLLaMA Assistant ---")
print("Type your questions below. Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    # Construct the prompt for the LLM
    # We use a simple instruction format here. For more complex interactions,
    # you might use specific chat formats like "Llama-2-chat" or "ChatML".
    prompt = f"Question: {user_input}\nAnswer:"

    print("TinyLLaMA is thinking...")
    try:
        # Generate a response from the LLM
        # max_tokens: The maximum number of tokens to generate in the response.
        # stop: A list of sequences that, if encountered, will stop the generation.
        #       "Question:" is used here to prevent the model from generating another question.
        response = llm(prompt, max_tokens=256, stop=["Question:", "\n\n"])

        # Extract and print the generated text
        ai_response = response["choices"][0]["text"].strip()
        print(f"TinyLLaMA: {ai_response}")
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        print("This might happen if the model is not loaded correctly or if there are issues with the prompt/response.")