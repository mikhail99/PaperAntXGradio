from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime

# Define BASE_URL here as it's needed for link reconstruction
BASE_URL = "https://huggingface.co"

class HFPaperInfo(BaseModel):
    fetch_date: str
    title: str
    hf_link: str
    arxiv_link: Optional[str] = None
    votes: int = 0
    
    @validator('fetch_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("fetch_date must be in YYYY-MM-DD format")
    
    @property
    def hf_id(self) -> str:
        """Extract Hugging Face paper ID from the link"""
        # Handle potential edge case where link might not be standard
        parts = self.hf_link.split('/')
        return parts[-1] if parts else ""
    
    @property
    def arxiv_id(self) -> Optional[str]:
        """Extract arXiv ID from the link if it exists"""
        if not self.arxiv_link:
            return None
        # Handle potential edge case where link might not be standard
        parts = self.arxiv_link.split('/')
        return parts[-1] if parts else None
    
    def to_csv_dict(self) -> dict:
        """Convert to dictionary for CSV storage"""
        return {
            'fetch_date': self.fetch_date,
            'title': self.title,
            'hf_id': self.hf_id,
            'arxiv_id': self.arxiv_id,
            'votes': self.votes
        }
    
    @classmethod
    def from_csv_dict(cls, csv_row: dict) -> 'HFPaperInfo':
        """Create HFPaperInfo instance from a CSV row dictionary"""
        hf_id = csv_row.get('hf_id', '')
        hf_link = f"{BASE_URL}/papers/{hf_id}" if hf_id else ""
        
        arxiv_id = csv_row.get('arxiv_id')
        arxiv_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None
            
        return cls(
            fetch_date=csv_row['fetch_date'],
            title=csv_row['title'],
            hf_link=hf_link,
            arxiv_link=arxiv_link,
            # Ensure votes are handled safely
            votes=int(csv_row.get('votes', 0) or 0)
        ) 