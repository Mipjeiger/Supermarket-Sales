from pydantic import BaseModel, Field, model_validator
from typing import Optional, Tuple

class SalesInvestigationRequest(BaseModel):
    """Request schema for market sales variance and revenue performance investigation."""
    order_id: Optional[str] = Field("ID-2012-79922", description="Unique order ID.")
    customer_target: str = Field(..., example="John Lee", description="The target customer for sales analysis.")
    investigation_date: str = Field("2026-07-25", example="2026-07-25", description="Date of sales audit.")
    category: Optional[str] = Field("Office Supplies", example="Office Supplies", description="Product category.")
    sub_category: Optional[str] = Field("Binders", example="Binders", description="Product sub-category.")
    product_name: Optional[str] = Field("Ibico Index Tab, Clear", example="Ibico Index Tab, Clear")
    market: Optional[str] = Field("APAC", example="APAC", description="Market region.")
    segment: Optional[str] = Field("Consumer", example="Consumer", description="Customer market segment.")
    predicted_sales: float = Field(..., example=524319.5, description="Model-predicted sales value.")
    actual_sales: float = Field(..., example=520000.0, description="Actual recorded sales revenue.")
    sales_difference: float = Field(..., example=4319.5, description="Absolute/Relative variance (Predicted - Actual).")
    discount: float = Field(0.05, example=0.05, description="Discount applied.")
    quantity: int = Field(2, example=2, description="Quantity sold.")
    unit_price: Optional[float] = Field(75000.0, example=75000.0)
    profit_margin: Optional[float] = Field(0.15, example=0.15)
    shipping_days: int = Field(153, example=153, description="Days taken to complete shipping.")

    # Computed / Optional Non-DB Fields
    sales_trend: Optional[str] = None
    custom_context: Optional[str] = None

    @model_validator(mode="after")
    def compute_non_db_fields(self):
        """Automatically derives sales_trend and custom_context if not explicitly provided."""
        calc_trend, calc_ctx = derive_sales_metrics(
            actual_sales=self.actual_sales,
            predicted_sales=self.predicted_sales,
            discount=self.discount,
            shipping_days=self.shipping_days
        )
        if not self.sales_trend:
            self.sales_trend = calc_trend
        if not self.custom_context:
            self.custom_context = calc_ctx
        return self

def derive_sales_metrics(
        actual_sales: float,
        predicted_sales: float,
        discount: float,
        shipping_days: int
) -> Tuple[str, str]:
    """Derives sales_trend, sales_difference, and custom_context from raw DB columns."""
    variance = actual_sales - predicted_sales
    pct_diff = (variance / predicted_sales) * 100 if predicted_sales > 0 else 0.0

    if pct_diff < -5.0:
        sales_trend = "Decreasing"
    elif pct_diff > 5.0:
        sales_trend = "Increasing"
    else:
        sales_trend = "Stable"

    reasons = []
    if pct_diff < 0:
        reasons.append(f"Actual sales fell short of prediction by {abs(pct_diff):.1f}%")
    else:
        reasons.append(f"Actual sales exceeded prediction by {pct_diff:.1f}%")

    if discount > 0.20:
        reasons.append(f"Aggressive discount applied ({discount:.0%})")
    if shipping_days > 30:
        reasons.append(f"Fulfilment delay detected ({shipping_days} days)")
    if shipping_days < 0:
        reasons.append(f"Negative fulfilment delay ({shipping_days} days)")

    custom_context = " | ".join(reasons)
    return sales_trend, custom_context