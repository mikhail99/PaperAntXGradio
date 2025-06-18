import arxiv
import chromadb
from chromadb.utils import embedding_functions
from core.proposal_agent.state import Paper

class PaperSearchTool:
    def __init__(self, db_path="chroma_db", collection_name="papers"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def search_arxiv(self, query: str, max_results: int = 5) -> list[Paper]:
        """Searches ArXiv for papers and adds them to the ChromaDB."""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers_to_add = []
        for result in search.results():
            paper_id = result.entry_id.split('/')[-1]
            # Check if paper already exists
            if self.collection.get(ids=[paper_id])['ids']:
                print(f"Paper {paper_id} already in DB, skipping.")
                continue

            paper_data = {
                "id": paper_id,
                "title": result.title,
                "summary": result.summary,
                "authors": [author.name for author in result.authors],
                "url": result.pdf_url
            }
            # Use a more structured document and metadata
            document_content = f"Title: {result.title}\nAbstract: {result.summary}"
            
            self.collection.add(
                documents=[document_content],
                metadatas={
                    "title": result.title,
                    "url": result.pdf_url,
                    "summary": result.summary,
                    "authors": ", ".join(paper_data["authors"]) # Store authors as a comma-separated string
                },
                ids=[paper_id]
            )
            papers_to_add.append(Paper(**paper_data))

        return papers_to_add

    def get_relevant_papers_from_db(self, query: str, n_results: int = 10) -> list[Paper]:
        """Retrieves relevant papers from ChromaDB based on a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        papers = []
        if results and results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                paper = Paper(
                    id=results['ids'][0][i] if results['ids'] and results['ids'][0] else f"db_{i}",
                    title=metadata.get('title', 'Unknown Title'),
                    summary=metadata.get('summary', 'No summary available.'),
                    authors=metadata.get('authors', '').split(', '),
                    url=metadata.get('url', '')
                )
                papers.append(paper)
        
        return papers

    def find_similar_papers(self, research_plan: str, n_results: int = 3) -> list[dict]:
        """Queries ChromaDB to find papers similar to the research plan."""
        results = self.collection.query(
            query_texts=[research_plan],
            n_results=n_results
        )
        
        similar_papers = []
        if results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                similar_papers.append({
                    "title": metadata.get('title'),
                    "url": metadata.get('url')
                })
        return similar_papers
