"""FastAPI endpoint for single-repo grounded chat."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.repo_chat import RepoChatService
from src.snowflake_client import SnowflakeClient
from src.store import LocalStore, SnowflakeStore


app = FastAPI(title="OpenSourceOps Chat API", version="1.0.0")
service = RepoChatService(SnowflakeStore(SnowflakeClient(), fallback=LocalStore()))


class IndexRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    ref: Optional[str] = Field(default=None, description="Optional tag/branch/commit")


class ChatRequest(BaseModel):
    repoId: str = Field(..., description="Indexed repo id")
    question: str = Field(..., min_length=3)
    mode: Literal["grounded_qa", "scenario_brainstorm", "risk_review"] = "grounded_qa"


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/index")
def index_repo(req: IndexRequest) -> dict:
    try:
        return service.build_index(req.repo_url, req.ref)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    try:
        return service.chat(repo_id=req.repoId, question=req.question, mode=req.mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

