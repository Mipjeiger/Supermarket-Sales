from pydantic import BaseModel, Field
from typing import Optional


class SalesInvestigationRequest(BaseModel):
    """Request schema for market sales variance and revenue performance investigation."""

    order_id: Optional[str] = Field("ID-2012-79922", description="Unique order ID.")
    customer_target: str = Field(
        ..., example="John Lee", description="The target customer for sales analysis."
    )
    investigation_date: str = Field(
        "2012-12-10", example="2026-07-25", description="Date of sales audit."
    )
    category: Optional[str] = Field(
        "Office Supplies", example="Office Supplies", description="Product category."
    )
    sub_category: Optional[str] = Field(
        "Binders", example="Binders", description="Product sub-category."
    )
    product_name: Optional[str] = Field(
        "Ibico Index Tab, Clear", example="Ibico Index Tab, Clear"
    )
    market: Optional[str] = Field("APAC", example="APAC", description="Market region.")
    segment: Optional[str] = Field(
        "Consumer", example="Consumer", description="Customer market segment."
    )
    predicted_sales: float = Field(
        ..., example=524319.5, description="Model-predicted sales value."
    )
    actual_sales: float = Field(
        ..., example=520000.0, description="Actual recorded sales revenue."
    )
    sales_difference: float = Field(
        ...,
        example=4319.5,
        description="Absolute/Relative variance (Predicted - Actual).",
    )
    discount: float = Field(0.05, example=0.05, description="Discount applied.")
    quantity: int = Field(2, example=2, description="Quantity sold.")
    unit_price: Optional[float] = Field(75000.0, example=75000.0)
    profit_margin: Optional[float] = Field(0.15, example=0.15)
    shipping_days: int = Field(
        153, example=153, description="Days taken to complete shipping."
    )
    sales_trend: str = Field(
        "Decreasing",
        example="Decreasing",
        description="Sales trajectory: Increasing, Decreasing, Stable.",
    )
    custom_context: Optional[str] = Field(
        None, example="Discount increased during Q2 promo campaign."
    )


# Singleton calls
sales_investigation_request = SalesInvestigationRequest()
