from pydantic import BaseModel, Field
from typing import Optional

class FraudInvestigationRequest(BaseModel):
    """Request schema aligned with engineering.supermarket database features for fraud auditing."""
    order_id: Optional[str] = Field("ID-2012-79922", description="Unique identifier for the supermarket order.")
    customer_name: str = Field(..., example="John Lee", description="Target customer name.")
    investigation_date: str = Field("2012-12-10", example="2026-07-25", description="Audit or evaluation date.")
    category: str = Field("Office Supplies", example="Office Supplies", description="Product category.")
    sub_category: str = Field("Binders", example="Binders", description="Product sub-category.")
    sales: float = Field(150000.0, example=150000.0, description="Total sales value.")
    quantity: int = Field(2, example=2, description="Number of units ordered.")
    unit_price: float = Field(75000.0, example=75000.0, description="Price per individual unit.")
    discount: float = Field(0.10, example=0.10, description="Discount factor applied.")
    profit: float = Field(-10140.0, example=-10140.0, description="Net profit earned.")
    profit_margin: float = Field(-0.07, example=-0.07, description="Profit margin ratio.")
    shipping_days: int = Field(214, example=214, description="Days taken to ship the order.")
    order_priority: Optional[str] = Field("Low", example="Low", description="Priority level of the order.")
    fraud_flag: int = Field(0, example=0, description="Binary classification flag (1 = Fraud, 0 = Legitimate).")
    risk_level: str = Field("LOW", example="LOW", description="Assessed risk tier: LOW, MEDIUM, HIGH, CRITICAL.")
    abuse_score: float = Field(0.04, example=0.04, description="Calculated velocity abuse score.")
    custom_context: Optional[str] = Field(None, example="High shipping delay with negative profit margin.")

# Singleton calls
fraud_investigation_request = FraudInvestigationRequest()