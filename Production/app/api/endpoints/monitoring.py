from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

@router.get("/metrics")
def get_metrics():
    """
    Expose Prometheus metrics collected accross the application.
    Prometheus will scrape this endpoint periodically to gather metrics data.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)