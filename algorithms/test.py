import asyncio
from paperqa import Docs, Settings
from paperqa import Settings
from paperqa.settings import AgentSettings, IndexSettings
from paperqa.agents.main import agent_query
from paperqa.agents.search import get_directory_index
from paperqa.agents import build_index
import os
from time import sleep
llm_model ="ollama/gemma3:4b"
embedding_model = "ollama/nomic-embed-text:latest"

from core.utils import get_local_llm_settings
my_settings=get_local_llm_settings(llm_model, embedding_model)

'''
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
    llm_config=local_llm_config,
    summary_llm=llm_model,
    summary_llm_config=local_llm_config,
    embedding = embedding_model,
    embedding_config=local_embedding_config,
    agent= AgentSettings(
        agent_llm=llm_model, 
        agent_config=local_llm_config,
        index = IndexSettings(
            index_directory="./indexes",
            paper_directory="./papers",
        ),
    ),
)

'''
# main is now an async function
async def main():
    print("Starting main function...")



    docs = Docs() # Initialize Docs without settings argument
    print("Docs object created.")
    
    print("Adding document...")    
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "papers", "2502.16111v1.pdf")
    pdf_path = os.path.abspath(pdf_path)  # Convert to absolute path

    p1 = await docs.aadd(
        pdf_path,
        summary="This is a test summary",
        metadata={"author": "John Doe"},
        doi="10.48550/arXiv.2502.16111",
        settings=my_settings # Pass settings to aadd method
    )
    print("Document added.")
    print(p1)
    
    
    sleep(10)
    print("Adding document...")    
    pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "papers", "2502.16645v1.pdf")
    pdf_path = os.path.abspath(pdf_path)  # Convert to absolute path

    p2 = await docs.aadd(
        pdf_path,
        summary="This is a test summary",
        metadata={"author": "John Doe"},
        doi="10.48550/arXiv.2502.16645",
        settings=my_settings # Pass settings to aadd method
    )
    print("Document added.")
    print(p2)

    sleep(5)
    print("Querying document...")
    response = await docs.aquery(
        "What is the main topic of this paper?",
        settings=my_settings # Pass settings to aquery method
    )
    print("Query finished.")
    print(response)

if __name__ == "__main__":
    print("Script starting...")
    # Run the async main function
    asyncio.run(main())
    print("Script finished.")
