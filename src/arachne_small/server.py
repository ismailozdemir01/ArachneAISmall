from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .artifacts import load_model, verify_artifact

LOGGER = logging.getLogger("arachne_small.server")
FRONTEND_DIR = Path(__file__).with_name("frontend")
MAX_PROMPT_CHARS = 20_000


class Runtime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model = None
        self.tokenizer = None
        self.artifact_dir: Path | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_error: str | None = None

    def load(self, artifact_dir: str) -> None:
        path = Path(artifact_dir).expanduser().resolve()
        with self.lock:
            try:
                verify_artifact(path)
                model, tokenizer = load_model(path, self.device)
                self.model, self.tokenizer, self.artifact_dir = model, tokenizer, path
                self.load_error = None
            except Exception as exc:
                self.model = self.tokenizer = None
                self.artifact_dir = None
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise

    def status(self) -> dict:
        with self.lock:
            ready = self.model is not None and self.tokenizer is not None
            result = {
                "ready": ready,
                "state": "ready" if ready else ("error" if self.load_error else "artifact_missing"),
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "version": __version__,
                "artifact": str(self.artifact_dir) if self.artifact_dir else None,
                "load_error": "Artifact validation or model loading failed. Check server logs." if self.load_error else None,
            }
            if ready:
                result.update({
                    "parameter_count": self.model.config.parameter_count(),
                    "vocab_size": self.model.config.vocab_size,
                    "context_length": self.model.config.max_seq_len,
                })
            return result

    def generate(self, prompt: str, history: list["ChatMessage"], max_new_tokens: int,
                 temperature: float, top_k: int,
                 language: Literal["auto", "tr", "en"]) -> dict:
        with self.lock:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("No verified model artifact is loaded.")
            current_prompt = prompt.strip()
            if not current_prompt:
                raise ValueError("Prompt cannot be empty.")
            language_prefix = "Türkçe konuş. " if language == "tr" else "Respond in English. " if language == "en" else ""
            turns = []
            for message in history:
                role = "User" if message.role == "user" else "Assistant"
                turns.append(f"{role}: {message.content.strip()}")
            turns.append(f"User: {current_prompt}")
            text = language_prefix + "\\n".join(turns) + "\\nAssistant:"
            token_ids = self.tokenizer.encode(text, out_type=int, add_bos=True)
            if not token_ids:
                raise ValueError("Prompt produced no tokens.")
            max_context = self.model.config.max_seq_len
            if len(token_ids) > max_context:
                token_ids = token_ids[-max_context:]
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            eos_id = self.tokenizer.eos_id()
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_id if eos_id >= 0 else None,
            )
            generated_ids = output[0, len(token_ids):].tolist()
            answer = self.tokenizer.decode(generated_ids).strip()
            return {
                "text": answer,
                "prompt_tokens": len(token_ids),
                "completion_tokens": len(generated_ids),
                "finish_reason": "eos" if generated_ids and eos_id in generated_ids else "length",
            }


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Conversation messages cannot be blank.")
        return value


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_k: int = Field(default=50, ge=0, le=500)
    language: Literal["auto", "tr", "en"] = "auto"

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be blank.")
        return value

    @field_validator("history")
    @classmethod
    def history_within_character_budget(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if sum(len(message.content) for message in value) > MAX_PROMPT_CHARS:
            raise ValueError("Conversation history exceeds the total character budget.")
        return value


runtime = Runtime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = os.environ.get("ARACHNE_SMALL_ARTIFACT")
    if artifact:
        try:
            runtime.load(artifact)
            LOGGER.info("Loaded verified Arachne Small artifact from %s", artifact)
        except Exception:
            LOGGER.exception("Arachne Small artifact could not be loaded")
    yield


app = FastAPI(
    title="ArachneAISmall",
    description="Local inference API for verified ArachneAISmall model artifacts.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "arachne-small"}


@app.get("/api/status")
def status() -> dict:
    return runtime.status()


@app.post("/api/generate")
def generate(payload: GenerateRequest, request: Request) -> dict:
    if not runtime.status()["ready"]:
        raise HTTPException(status_code=503, detail="No trained model artifact is loaded. Set ARACHNE_SMALL_ARTIFACT to a verified artifact directory.")
    try:
        result = runtime.generate(
            payload.prompt, payload.history, payload.max_new_tokens,
            payload.temperature, payload.top_k, payload.language,
        )
        return {"model": "arachne-small", **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Inference runtime failure for client %s", request.client.host if request.client else "unknown")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected inference failure")
        raise HTTPException(status_code=500, detail="Inference failed. Check the server log for details.") from exc


def main() -> None:
    import uvicorn
    host = os.environ.get("ARACHNE_SMALL_HOST", "127.0.0.1")
    port = int(os.environ.get("ARACHNE_SMALL_PORT", "8000"))
    if host not in {"127.0.0.1", "localhost", "::1"} and os.environ.get("ARACHNE_SMALL_ALLOW_REMOTE") != "1":
        raise SystemExit("Remote binding is disabled. Set ARACHNE_SMALL_ALLOW_REMOTE=1 only behind a secured network boundary.")
    uvicorn.run("arachne_small.server:app", host=host, port=port, reload=False, access_log=True)


if __name__ == "__main__":
    main()
