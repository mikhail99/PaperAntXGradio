from core.proposal_agent.tools import PaperSearchTool

if __name__ == "__main__":
    print("Testing find_similar_papers...")
    tool = PaperSearchTool()
    sample_plan = "Use LLMs to guide parameter selection for ODE solvers and benchmark against ode45."
    collection_id = "99a6b6d7-986f-4b95-a6aa-16c75ab0cc55"
    similar = tool.find_similar_papers(sample_plan, n_results=3, collection_id=collection_id)
    print("Similar papers found:")
    for i, paper in enumerate(similar, 1):
        print(f"{i}. {paper['title']} ({paper['url']})")