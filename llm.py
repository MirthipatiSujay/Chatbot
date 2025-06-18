from llama_cpp import Llama

# Load once
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4
)


def generate_response(user_input, emotion):
    prompt = (
        f"You are a compassionate assistant. A user is feeling {emotion} and said:\n"
        f"\"{user_input}\"\n\n"
        f"Respond empathetically with kindness, support, and emotional intelligence."
    )

    output = llm(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.95,
        stop=["</s>"]
    )

    return output["choices"][0]["text"].strip()
