from paperqa import Settings
from paperqa.settings import AgentSettings, IndexSettings
import os

#llm_model = "ollama/gemma3:27b"
#embedding_model = "ollama/nomic-embed-text:latest"

#llm_model = "ollama/gemma3:27b"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_local_llm_settings(llm_model, embedding_model) -> Settings:
    # This is the prompt that was causing the question to be repeated.
    # By defining our own, we can control the output format.
    qa_prompt = (
        "Answer the question '{question}' based on the provided context.\n"
        "Cite the sources using the keys provided in the context.\n"
        "If the context is insufficient, respond with 'I cannot answer this question based on the provided context.'\n\n"
        "Context:\n{context}"
    )

    local_llm_config = {
    "model_list": [
        {
            "model_name": llm_model,
            "litellm_params": {
                "model": llm_model,
                "api_base": "http://localhost:11434",
            },
        }
    ]
    }

    local_embedding_config = {
        "api_base": "http://localhost:11434",
    }

    my_settings=Settings(
        llm=llm_model,
        prompts={'use_json': False, 'qa': qa_prompt},
        parsing={'use_doc_details': False},
        llm_config=local_llm_config,
        summary_llm=llm_model,
        summary_llm_config=local_llm_config,
        embedding = embedding_model,
        embedding_config=local_embedding_config,
        agent= AgentSettings(
            agent_llm=llm_model, 
            agent_config=local_llm_config,
            index = IndexSettings(
                index_directory=os.path.join(PROJECT_ROOT, "data", "indexes"),
                paper_directory=os.path.join(PROJECT_ROOT, "data", "papers"),
                manifest_file=os.path.join(PROJECT_ROOT, "data", "papers_manifest", "generated_manifest.csv"),
            ),
        ),

        index_directory=os.path.join(PROJECT_ROOT, "data", "indexes"),
        paper_directory=os.path.join(PROJECT_ROOT, "data", "papers"),
        manifest_file=os.path.join(PROJECT_ROOT, "data", "papers_manifest", "generated_manifest.csv"),

    )
    return my_settings
# Placeholder for utility functions and constants 