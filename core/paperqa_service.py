# core/paperqa_service.py
import asyncio
import tempfile
from pathlib import Path
import os
import html
from typing import List, Dict, Any, Tuple

from paperqa import Docs, Settings
from paperqa.settings import AgentSettings, IndexSettings
from core.data_models import Article
from time import sleep
# --- Global PaperQA Configuration (base settings) ---

llm_model = "ollama/gemma3:27b"
embedding_model = "ollama/nomic-embed-text:latest"

from core.utils import get_local_llm_settings
my_settings=get_local_llm_settings(llm_model, embedding_model)


def extract_pdf_path_from_notes(notes: str) -> str | None:
    if not notes:
        return None
    lines = notes.splitlines()
    for line in lines:
        if line.startswith("Local PDF: "):
            return line.replace("Local PDF: ", "").strip()
    return None

class PaperQAService:

    async def query_documents(
        self, articles: List[Article], question: str
    ) -> Dict[str, Any]:
        """
        Processes a list of Article objects with PaperQA and answers a question.
        Each Article should have a local PDF path (in notes or a dedicated field).
        Returns a dictionary with 'answer_text', 'formatted_evidence', and 'error'.
        """
        if not articles:
            return {"answer_text": "No articles provided.", "formatted_evidence": "", "error": None}

        try:
            docs = Docs()
            print(f"Adding {len(articles)} articles to PaperQA Docs...")
            added_count = 0
            for i, article in enumerate(articles):
                pdf_path = extract_pdf_path_from_notes(getattr(article, 'notes', ''))
                if not pdf_path or not os.path.exists(pdf_path):
                    print(f"File does not exist, skipping: {pdf_path}")
                    continue
                arxiv_id = getattr(article, 'id', None)
                if arxiv_id and arxiv_id.endswith(('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9')):
                    arxiv_id = arxiv_id[:-2]
                # Compose metadata and summary
                metadata = {
                    "arxiv_id": arxiv_id,
                    "title": getattr(article, 'title', None),
                    "authors": getattr(article, 'authors', None),
                    "tags": getattr(article, 'tags', None),
                    "abstract": getattr(article, 'abstract', None),
                    "publication_date": str(getattr(article, 'publication_date', '')),
                }
                summary = article.abstract or article.title or ""
                doi= f"10.48550/arXiv.{arxiv_id}"
                try:
                    await docs.aadd(
                        pdf_path,
                        citation=f"arXiv:{arxiv_id}",
                        docname=f"arXiv:{arxiv_id}",
                        dockey=f"arXiv:{arxiv_id}",
                        title=article.title,
                        authors=article.authors,
                        doi=doi,
                        settings=my_settings,
                    )
                    print(f"Document added: {pdf_path}")
                    
                    sleep(10)
                    print("HACK")
                    added_count += 1
                except Exception as e:
                    print(f"Error adding document {pdf_path}: {str(e)}")
                    continue

            if added_count == 0:
                return {"answer_text": "No valid PDF documents could be processed.", "formatted_evidence": "", "error": None}

            print(f"Querying PaperQA with: '{question}'")
            response = await docs.aquery(question, settings=my_settings)

            print("PaperQA query finished.")

            answer_text = response.formatted_answer if response and response.formatted_answer else "No answer found by PaperQA."
            
            contexts_md = ""
            #if response and response.contexts:
            #    for i, ctx in enumerate(response.contexts):
            #        contexts_md += f"\n{i + 1}. **Source:** {ctx.citation} (Score: {ctx.score:.2f})\n"
            #        contexts_md += f"> {ctx.context}\n\n" # Raw context
            #else:
            #    contexts_md += "_No specific evidence found by PaperQA._\n"
            
            return {"answer_text": answer_text, "formatted_evidence": contexts_md, "error": None}

        except Exception as e:
            error_message = f"Error during PaperQA processing: {str(e)}"
            print(error_message)
            return {"answer_text": "", "formatted_evidence": "", "error": error_message} 