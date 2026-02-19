from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SourceType(str, Enum):
    WEBPAGE = "webpage"
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    REPORT = "report"
    PREPRINT = "preprint"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    DATASET = "dataset"
    SOFTWARE = "software"
    VIDEO = "video"
    COURT_OPINION = "court_opinion"
    UNKNOWN = "unknown"


class Author(BaseModel):
    name: str = Field(..., description="Full name of the author")
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    orcid: Optional[str] = None
    affiliation: Optional[str] = None
    email: Optional[str] = None


class CitationMetadata(BaseModel):
    title: str = Field(..., description="Title of the document")
    authors: List[Author] = Field(default_factory=list)
    date_published: Optional[str] = None
    year: Optional[int] = None
    source_type: SourceType = SourceType.UNKNOWN

    # Publication info
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None

    # Identifiers
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    isbn: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None

    abstract: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    language: Optional[str] = "en"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
