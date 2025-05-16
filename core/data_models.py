from typing import List, Dict, Optional, Union
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field, validator

class Tag(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    color: str = "#1890ff"  # Default blue color
    parent_id: Optional[str] = None

class Article(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    authors: List[str]
    abstract: str
    publication_date: Union[datetime, str]
    tags: List[str] = []
    favorite: bool = False
    rating: Optional[str] = None  # "accept", "reject", or None
    citation_count: int = 0
    notes: str = ""  # User notes, separate from abstract

    @validator("publication_date", pre=True, always=True)
    def ensure_datetime(cls, v):
        if isinstance(v, datetime):
            return v
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return datetime.now()

class Collection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    tags: Dict[str, Tag] = {}
    articles: Dict[str, Article] = {}
    archived: bool = False
