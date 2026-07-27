from pydantic import BaseModel, Field, model_validator
from typing import Optional, Tuple

class FraudInvestigationRequest(BaseModel):
    """Request schema aligned with engineering.supermarket database features for fraud auditing."""
    order_id: Optional[str] = Field("ID-2012-79922", description="Unique identifier for the supermarket order.")
    customer_name: str = Field(..., example="John Lee", description="Target customer name.")
    investigation_date: str = Field("2026-07-25", example="2026-07-25", description="Audit or evaluation date.")
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

    # Computed for Non-DB Fields
    risk_level: Optional[str] = None
    abuse_score: Optional[float] = None
    custom_context: Optional[str] = None

    @model_validator(mode="after")
    def compute_non_db_fields(self):
        """Automatically calculates risk_level, abuse_score, and custom_context based on input features."""
        calc_risk, calc_score, calc_ctx = derive_fraud_metrics(
            fraud_flag=self.fraud_flag,
            profit_margin=self.profit_margin,
            discount=self.discount,
            shipping_days=self.shipping_days
        )
        if not self.risk_level:
            self.risk_level = calc_risk
        if self.abuse_score is None:
            self.abuse_score = calc_score
        if not self.custom_context:
            self.custom_context = calc_ctx
        return self

def derive_fraud_metrics(
        fraud_flag: int,
        profit_margin: float,
        discount: float,
        shipping_days: int
) -> Tuple[str, float, str]:
    """Derives risk_level, abuse_score, and custom_context from raw DB columns."""
    score = 0.0
    reasons = []

    if fraud_flag == 1:
        score += 0.50
        reasons.append("Marked positive by upstream fraud model.")
    if profit_margin < 0:
        score += 0.25
        reasons.append(f"Negative profit margin ({profit_margin:.1%})")
    if discount >= 0.30:
        score += 0.15
        reasons.append(f"High discount applied ({discount:.0%})")
    if shipping_days >= 60:
        score += 0.10
        reasons.append(f"Excessive shipping delay ({shipping_days} days)")
    elif shipping_days < 0:
        score += 0.20
        reasons.append(f"Negative shipping days ({shipping_days})")

    abuse_score = min(round(score, 2), 1.0)

    if abuse_score >= 0.70:
        risk_level = "CRITICAL"
    elif abuse_score >= 0.40:
        risk_level = "HIGH"
    elif abuse_score >= 0.20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    custom_context = " | ".join(reasons) if reasons else "Normal transaction behavior observed."
    return risk_level, abuse_score, custom_context