import logging
import aiohttp
from typing import Dict, Optional
from datetime import datetime
from app.config.config import settings

logger = logging.getLogger(__name__)

class SlackNotifier:
    """Slack notification service for sending alerts and notifications."""

    def __init__(self):
        self.webhook_url = settings.SLACK_WEBHOOK_URL
        self.channel = settings.SLACK_CHANNEL
        self._enabled = bool(self.webhook_url)

        if not self._enabled:
            logger.warning("⚠️ Slack webhook URL not configured. Notifications disabled.")

    async def send_message(self, message: str, color: str = "#36a64f"):
        """Send a message to Slack."""
        if not self._enabled:
            return
        
        try:
            payload = {
                "channel": self.channel,
                "attachments": [
                    {
                        "color": color,
                        "text": message,
                        "footer": settings.PROJECT_NAME,
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:

                    if response.status == 200:
                        logger.info("✅ Slack notification sent successfully.")
                    else:
                        logger.error(f"❌ Failed to send Slack notification. Status: {response.status}, Response: {await response.text()}")

        except Exception as e:
            logger.error(f"❌ Exception occurred while sending Slack notification: {str(e)}")

    async def send_error_alert(self, error_message: str, severity: str = "error"):
        """Send an error alert to Slack."""
        colors = {
            "critical": "#ff0000",
            "error": "#ff4444",
            "warning": "#ffaa00",
            "info": "#36a64f"
        }

        message = f"🚨 *{severity.upper()} ALERT*\n{error_message}"
        await self.send_message(message, color=colors.get(severity, "#ff4444"))

    async def send_model_alert(self, model_name: str, metrics: Dict):
        """Send model performance altert to Slack."""
        if metrics.get('rmse', 0) > settings.ALERT_THRESHOLD_RMSE:
            message = f"""
            ⚠️ *Model Performance Degradation Alert*
            Model: {model_name}
            RMSE: {metrics.get('rmse', 'N/A'):.3f}
            MAE: {metrics.get('mae', 'N/A'):.3f}
            R2: {metrics.get('r2', 'N/A'):.3f}
            MAPE: {metrics.get('mape', 'N/A'):.3f}
            """
            await self.send_message(message, color="#ffaa00")

    async def send_system_alert(self, component: str, status: str, details: str = ""):
        """Send system health alert to Slack."""
        if status == "critical":
            color = "#ff0000"
            emoji = "🚨"
        elif status == "warning":
            color = "#ffaa00"
            emoji = "⚠️"
        else:
            color = "#36a64f"
            emoji = "✅"

        message = f"{emoji} *System Alert: {component}*\nStatus: {status}\nDetails: {details}"
        await self.send_message(message, color=color)

    async def send_daily_report(self, report_data: Dict):
        """Send daily performance report to Slack."""
        message = f"""
        📊 *Daily Performance Report*
        Date: {datetime.now().strftime('%Y-%m-%d')}

        *Models*
        {self._format_models_report(report_data.get('models', {}))}

        *System Health*
        API: {report_data.get('api_status', 'N/A')}
        LLM: {report_data.get('llm_status', 'N/A')}
        Database: {report_data.get('db_status', 'N/A')}

        *Predictions*
        Total: {report_data.get('predictions_total', 0)}
        Avg Latency: {report_data.get('avg_latency', 0):.2f}s
        """

        await self.send_message(message, color="#36a64f")

    def _format_models_report(self, models_data: Dict) -> str:
        """Format model metrics for report."""
        lines = []
        
        for name, metrics in models_data.items():
            lines.append(
                f"• {name}: RMSE={metrics.get('rmse', 'N/A'):.3f}, "
                f"R²={metrics.get('r2', 'N/A'):.3f}"
            )
        return "\n".join(lines)

# Singleton instance of SlackNotifier
slack_notifier = SlackNotifier()