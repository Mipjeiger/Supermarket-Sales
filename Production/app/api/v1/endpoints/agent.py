from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.security_agent import security_agent_engine

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    query: str
    context: str


@router.post("/chat")
async def process_agent_interaction(payload: ChatRequest):
    try:
        reply = await security_agent_engine.chat_investigate(
            session_id=payload.session_id,
            user_query=payload.query,
            velocity_context=payload.context,
        )
        return {"status": "success", "reply": reply}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing agent interaction: {str(e)}"
        )
