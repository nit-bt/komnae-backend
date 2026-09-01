"""
Request and response schemas.

The `Issue` shape is the contract with the Lovable frontend. `start` and `end`
are character offsets into the exact string the client sent, so the client can
render underlines with text.slice(start, end) and nothing else.
"""

from typing import Literal

from pydantic import BaseModel, Field

IssueType = Literal["spelling", "grammar", "style"]
IssueSource = Literal["dictionary", "ai"]


class Issue(BaseModel):
    start: int = Field(..., description="Character offset into the submitted text")
    end: int = Field(..., description="Exclusive end offset")
    original: str = Field(..., description="The flagged substring")
    suggestion: str = Field("", description="Proposed replacement, may be empty")
    alternatives: list[str] = Field(default_factory=list)
    reason: str = Field("", description="Short explanation, in Khmer")
    type: IssueType = "spelling"
    source: IssueSource = "dictionary"
    definition: str = Field("", description="Dictionary gloss of the suggestion")
    pos: str = Field("", description="Part of speech of the suggestion")
    confidence: float = 1.0


class CheckRequest(BaseModel):
    text: str = Field(..., max_length=20_000)
    use_ai: bool = Field(True, description="Run the Gemini pass as well")


class CheckResponse(BaseModel):
    issues: list[Issue]
    tokens: int
    backend: str = Field("", description="Which segmenter served the request")
    ai: Literal["ok", "skipped", "no_key", "error", "timeout"] = "skipped"
    ai_error: str = ""


class RefineRequest(BaseModel):
    text: str = Field(..., max_length=20_000)
    issues: list[Issue] = Field(default_factory=list)


class KeyCheckRequest(BaseModel):
    api_key: str


class KeyCheckResponse(BaseModel):
    valid: bool
    model: str = ""
    error: str = ""
