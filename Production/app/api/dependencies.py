"""FastAPI dependency helpers for agent access via application state."""

from fastapi import HTTPException, Request


def get_fraud_agent(request: Request):
    """Dependency to get fraud detection agent from app state."""
    agent = getattr(request.app.state, "fraud_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Fraud agent not initialized")
    return agent


def get_abuse_agent(request: Request):
    """Dependency to get abuse investigation agent from app state."""
    agent = getattr(request.app.state, "abuse_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Abuse agent not initialized")
    return agent


def get_security_agent(request: Request):
    """Dependency to get security analyst agent from app state."""
    agent = getattr(request.app.state, "security_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Security agent not initialized")
    return agent


def get_llm_judge(request: Request):
    """Dependency to get LLM-as-Judge evaluator from app state."""
    judge = getattr(request.app.state, "llm_judge", None)
    if judge is None:
        raise HTTPException(status_code=503, detail="LLM Judge not initialized")
    return judge


def get_agents_initialized(request: Request) -> bool:
    """Dependency to check whether agents are initialized."""
    return bool(getattr(request.app.state, "agents_initialized", False))