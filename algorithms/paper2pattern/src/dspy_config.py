import dspy

def configure_dspy(model_name="gemma3:12b", ollama_base_url="http://localhost:11434"):
    """
    Configures dspy to use a local Ollama model.
    """
    # Configure the language model using the 'ollama_chat/' prefix
    # as per the official DSPy documentation.
    # See: https://dspy.ai/learn/programming/language_models/#__tabbed_1_6
    ollama_lm = dspy.LM(
        f'ollama_chat/{model_name}',
        api_base=ollama_base_url,
        api_key='' # Ollama doesn't require an API key
    )

    # Configure the retrieval model (using default SentenceTransformer)
    # This will download a model on first use
    #retriever_model = dspy.SentenceTransformers("all-MiniLM-L6-v2")

    # Set the configured models in dspy
    dspy.configure(
        lm=ollama_lm,
        #rm=retriever_model
    )

    print(f"DSPy configured to use Ollama model '{model_name}' and SentenceTransformers retriever.")

if __name__ == '__main__':
    # Example of how to use it
    configure_dspy()
    
    # You can verify the language model is working
    try:
        response = dspy.settings.lm("Who is the CEO of Apple?")
        print("\nTest query response:")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred while testing the LM: {e}")
        print("Please ensure your Ollama server is running and the model is available.") 