"""HTTP endpoint with TypeSafe's /v1/systemone request and answer shape."""

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import MagibuMizan


def create_app(engine=None, model=None, api_key=None, max_questions=None):
    """Build an app; an injected engine keeps HTTP tests independent of model weights."""
    model = model or os.environ.get("MODEL")
    api_key = api_key if api_key is not None else os.environ.get("API_KEY")
    max_questions = max_questions if max_questions is not None else int(os.environ.get("MAX_QUESTIONS", "10"))
    if max_questions < 1:
        raise ValueError("MAX_QUESTIONS must be positive")

    @asynccontextmanager
    async def lifespan(app):
        if app.state.engine is None:
            if not model:
                raise RuntimeError("set MODEL to a local path or Hugging Face model id")
            app.state.engine = MagibuMizan(
                model,
                backend=os.environ.get("BACKEND"),
                temperature=float(os.environ.get("TEMPERATURE", "1.0")),
            )
        yield

    app = FastAPI(title="MagibuMizan", lifespan=lifespan)
    app.state.engine = engine

    def error(status, message):
        return JSONResponse({"status": status, "message": message}, status_code=status)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if api_key and not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {api_key}"
        ):
            return error(401, "invalid API key")
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            if "model" in body and not isinstance(body["model"], str):
                raise ValueError("model must be a string")
            questions = body["questions"]
            if not isinstance(questions, dict) or not questions:
                raise ValueError("questions must be a nonempty object")
            if len(questions) > max_questions:
                raise ValueError(f"at most {max_questions} questions per request")
            answers, tokens = app.state.engine.answer(body["state"], questions)
        except (KeyError, TypeError, ValueError) as exc:
            return error(422, str(exc))
        return {
            "model": model,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    return app


app = create_app()
