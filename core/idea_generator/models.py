from pydantic import BaseModel

class IdeaTemplate(BaseModel):
    research_question: str
    hypothesis: str
    methodology: str
    data: str
    experiments: str
    key_findings: str
    applications: str


class Candidate(BaseModel):
    id: str
    idea: IdeaTemplate
    win_count: int = 0

    def __repr__(self):
        return f"Candidate(win_count={self.win_count}, idea={self.idea})"