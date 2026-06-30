import os
import logging
from pathlib import Path
from app.config.config import settings

logger = logging.getLogger(__name__)

# Check if app.config.config.settings is accessible
if settings:
    logger.info("Settings loaded successfully.")
    logger.info(f"📂 Active Cleaned Data Storage: {settings.DATA_CLEANED}")
else:
    logger.error("Failed to load settings.")