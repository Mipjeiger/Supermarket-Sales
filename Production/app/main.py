"""
FastAPI Main Application with Agentic Framework Integration
Enterprise-grade API for fraud detection, abuse investigation, and security analysis
Configured for decoupled execution using HuggingFace External Cloud Inference.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

# Import configuration
from app.config.config import settings

# CRITICAL FIX: Use Endpoint instead of Pipeline for Cloud Hosted Inference
from langchain_huggingface import HuggingFaceEndpoint

# Import API routers (Safe from circular imports)
from app.api.endpoints import (
    prediction,
    monitoring
)

# Import agentic framework components
from app.agentic.agents.fraud_detector import FraudDetectionAgent
from app.agentic.agents.abuse_investigator import AbuseInvestigationAgent
from app.agentic.agents.security_analyst import SecurityAnalystAgent
from app.agentic.evaluators.llm_as_judge import LLMAsJudge

# Import monitoring and alerting
from app.monitoring.metrics import MetricsCollector # Ensure MetricsCollector is imported for monitoring
from app.monitoring.slack_notifier import slack_notifier

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initializes cloud connections and safely binds them to app state.
    """
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} with Agentic Framework...")
    app.state.agents_initialized = False
    
    try:
        try:
            logger.info("🔧 Connecting to HuggingFace Cloud Inference API...")
            model_name = "meta-llama/Llama-3.1-8B-Instruct"
            api_token = settings.HUGGINGFACE_API_KEY

            if not api_token:
                logger.error("❌ HUGGINGFACE_API_KEY is not set in environment variables.")
                raise ValueError("HUGGINGFACE_API_KEY is required for LLM initialization.")
            
            # Using external hosted endpoint inference wrapper
            remote_llm = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                max_new_tokens=512,
                temperature=0.3,
                huggingfacehub_api_token=api_token,
                timeout=30
            )
            logger.info(f"✅ Remote Cloud LLM connected successfully: {model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Cloud LLM connection failed: {str(e)}")
            logger.warning("⚠️ Running agents in fallback mode (ML-only)")
            remote_llm = None
        
        # Initialize and bind agents to app state to avoid circular dependencies
        logger.info("🔧 Injecting state into Fraud Detection Agent...")
        app.state.fraud_agent = FraudDetectionAgent(llm_model=remote_llm, vector_store=None)
        
        logger.info("🔧 Injecting state into Abuse Investigation Agent...")
        app.state.abuse_agent = AbuseInvestigationAgent(llm_model=remote_llm, vector_store=None)
        
        logger.info("🔧 Injecting state into Security Analyst Agent...")
        app.state.security_agent = SecurityAnalystAgent(llm_model=remote_llm, vector_store=None)
        
        logger.info("🔧 Injecting state into LLM-as-Judge Evaluator...")
        app.state.llm_judge = LLMAsJudge(llm_model=remote_llm)
        
        app.state.agents_initialized = True
        
        await slack_notifier.send_message(
            f"🚀 {settings.PROJECT_NAME} with Cloud Agentic Framework started successfully!",
            color="#36a64f"
        )
        logger.info("✅ All agents initialized and attached to App State successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents in app state: {str(e)}")
        app.state.agents_initialized = False
    
    yield
    
    logger.info(f"🛑 Shutting down {settings.PROJECT_NAME}...")
    await slack_notifier.send_message(
        f"🛑 {settings.PROJECT_NAME} is shutting down...",
        color="#ff4444"
    )

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    description="Enterprise-grade Fraud Detection, Abuse Investigation, and Security Analysis API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Late imports to ensure routers read cleanly from app.state dependencies
from app.api.endpoints import (
    fraud_detection,
    abuse_detection,
    security_analysis,
    agentic_investigation
)

# Include routers
app.include_router(prediction.router, prefix=f"/api/{settings.API_VERSION}/prediction", tags=["Prediction"])
app.include_router(fraud_detection.router, prefix=f"/api/{settings.API_VERSION}/fraud", tags=["Fraud Detection"])
app.include_router(abuse_detection.router, prefix=f"/api/{settings.API_VERSION}/abuse", tags=["Abuse Investigation"])
app.include_router(security_analysis.router, prefix=f"/api/{settings.API_VERSION}/security", tags=["Security Analysis"])
app.include_router(agentic_investigation.router, prefix=f"/api/{settings.API_VERSION}/investigation", tags=["Agentic Investigation"])
app.include_router(monitoring.router, prefix=f"/api/{settings.API_VERSION}/monitoring", tags=["Monitoring"])

@app.get("/")
async def root():
    """Root endpoint reading live status from safe application state mapping."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.API_VERSION,
        "status": "running",
        "agents": {
            "fraud_detection": "active" if getattr(app.state, "fraud_agent", None) else "inactive",
            "abuse_investigation": "active" if getattr(app.state, "abuse_agent", None) else "inactive",
            "security_analysis": "active" if getattr(app.state, "security_agent", None) else "inactive",
            "llm_judge": "active" if getattr(app.state, "llm_judge", None) else "inactive"
        },
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "agents": {
            "fraud_detection": getattr(app.state, "fraud_agent", None) is not None,
            "abuse_investigation": getattr(app.state, "abuse_agent", None) is not None,
            "security_analysis": getattr(app.state, "security_agent", None) is not None,
            "llm_judge": getattr(app.state, "llm_judge", None) is not None
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/system/status")
async def system_status():
    f_agent = getattr(app.state, "fraud_agent", None)
    a_agent = getattr(app.state, "abuse_agent", None)
    s_agent = getattr(app.state, "security_agent", None)
    judge = getattr(app.state, "llm_judge", None)

    return {
        "system": {
            "name": settings.PROJECT_NAME,
            "version": settings.API_VERSION,
            "environment": "production" if not settings.DEBUG else "development"
        },
        "agents": {
            "fraud_detection": {
                "status": "active" if f_agent else "inactive",
                "history_count": len(f_agent.investigation_history) if f_agent else 0
            },
            "abuse_investigation": {
                "status": "active" if a_agent else "inactive",
                "history_count": len(a_agent.investigation_history) if a_agent else 0
            },
            "security_analysis": {
                "status": "active" if s_agent else "inactive",
                "findings_count": len(s_agent.security_findings) if s_agent else 0
            },
            "llm_judge": {
                "status": "active" if judge else "inactive",
                "evaluations_count": len(judge.evaluation_history) if judge else 0
            }
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )