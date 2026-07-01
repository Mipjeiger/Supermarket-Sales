import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import json
import asyncio
from functools import lru_cache

from app.config.config import settings

logger = logging.getLogger(__name__)

"""
Data Retrieval Tool for Agentic System
Retrieves transaction data, user history, and patterns from various data sources.
"""

class DataRetrievalTool:
    """Tool for retrieving data from multiple sources:
    - Transaction data
    - User history
    - Pattern data
    - Real-time streams
    """

    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300 # 5 minutes
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize connections to databases and APIs."""
        try:
            self.engine = create_engine(settings.POSTGRES_URL)
            logger.info("Database connection initialized.")

        except Exception as e:
            logger.error(f"Error initializing database connection: {e}")
            raise

    def run(self, query: str) -> Dict:
        """Main entry point for the tool.
        
        Args:
            query: JSON string or dict with query parameters.
            
        Returns:
            Dict containing retrieved data
        """
        try:
            # Parse query if string
            if isinstance(query, str):
                query = json.loads(query)

            # Determine query type
            query_type = query.get("type", "transaction")

            if query_type == "transaction":
                return self.get_transaction_data(query)
            elif query_type == "user_history":
                return self.get_user_history(query)
            elif query_type == 'patterns':
                return self.get_pattern_data(query)
            elif query_type == 'real_time':
                return self.get_real_time_data(query)
            else:
                return {'error': f'Unknown query type: {query_type}'}
                
        except Exception as e:
            logger.error(f"❌ Data retrieval failed: {str(e)}")
            return {'error': str(e)}
        
    @lru_cache(maxsize=100)
    def get_transaction_data(self, params: Dict) -> Dict:
        """
        Retrieve transaction data based on parameters.
        
        Args:
            params: Dict with transaction_id, user_id, date_range, etc.
            
        Returns:
            Dict containing transaction data
        """
        try:
            transaction_id = params.get("transaction_id")
            user_id = params.get("user_id")
            date_range = params.get("date_range", {})

            # Build SQL query
            query = "SELECT * FROM engineering.supermarket WHERE 1=1"

            if transaction_id:
                query += f" AND order_id = '{transaction_id}'"

            if user_id:
                query += f" AND customer_id = '{user_id}'"

            if date_range:
                start_date = date_range.get("start")
                end_date = date_range.get("end")
                if start_date:
                    query += f" AND order_date >= '{start_date}'"

                if end_date:
                    query += f" AND order_date <= '{end_date}'"

            query += " ORDER BY oder_date DESC LIMIT 1000"

            # Execute query
            if self.engine:
                df = pd.read_sql(query, self.engine)
                data = df.to_dict('records')
            else:
                raise ValueError("❌ Database engine not initialized. Cannot retrieve transaction data.")

            return {
                'status': 'success',
                'data': data,
                'count': len(data),
                'query_params': params,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Transaction data retrieval failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
    def get_user_history(self, params: Dict) -> Dict:
        """
        Retrieve user history and behavior patterns.
        
        Args:
            params: Dict with user_id, timeframe, etc.
            
        Returns:
            Dict containing user history
        """
        try:
            user_id = params.get("user_id")
            timeframe = params.get("timeframe", "30d")  # Default to last 30 days

            if not user_id:
                return {'error': 'user_id is required for user history retrieval'}
            
            # Calculate date range
            days = int(timeframe.replace('d', ''))
            start_date = datetime.now() - timedelta(days=days)

            # Build SQL history user query
            query = f"""
                SELECT
                    order_id,
                    order_date,
                    sales,
                    quantity,
                    discount,
                    profit,
                    category,
                    sub_category,
                    ship_mode
                FROM engineering.supermarket
                WHERE customer_id = '{user_id}'
                    AND order_date >= '{start_date.isoformat()}'
                ORDER BY order_date DESC
            """

            if self.engine:
                df = pd.read_sql(query, self.engine)
                data = df.to_dict('records')
            else:
                raise ValueError("❌ Database engine not initialized. Cannot retrieve user history.")


            # Calculate user statistics
            stats = self._calculate_user_stats(data)

            return {
                'status': 'success',
                'user_id': user_id,
                'transactions': data,
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ User history retrieval failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
    def get_pattern_data(self, params: Dict) -> Dict:
        """
        Retrieve pattern data and trends.
        
        Args:
            params: Dict with pattern_type, timeframe, etc.
            
        Returns:
            Dict containing pattern data
        """
        try:
            pattern_type = params.get('pattern_type', 'all')
            timeframe = params.get('timeframe', '30d')
            
            # Get historical data
            days = int(timeframe.replace('d', ''))
            start_date = datetime.now() - timedelta(days=days)

            query = f"""
                SELECT
                    DATE(order_date) as date,
                    SUM(sales) as total_sales,
                    COUNT(order_id) as transaction_count,
                    AVG(sales) as avg_sales,
                    AVG(discount) as avg_discount,
                    SUM(quantity) as total_quantity
                FROM engineering.supermarket
                WHERE order_date >= '{start_date.isoformat()}'
                GROUP BY DATE(order_date)
                ORDER BY date
            """

            if self.engine:
                df = pd.read_sql(query, self.engine)
                data = df.to_dict('records')
            else:
                raise ValueError("❌ Database engine not initialized. Cannot retrieve pattern data.")

            return {
                'status': 'success',
                'pattern_type': pattern_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Pattern data retrieval failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
    def get_real_time_data(self, params: Dict) -> Dict:
        """
        Retrieve real-time data from streaming sources.
        
        Args:
            params: Dict with stream_type, limit, etc.
            
        Returns:
            Dict containing real-time data
        """
        try:
            stream_type = params.get('stream_type', 'transactions')
            limit = params.get('limit', 100)
            
            # In production, this would connect to Kafka/RabbitMQ
            # For now, return sample real-time data
            real_time_data = self._get_sample_real_time_data(limit)

            return {
                'status': 'success',
                'stream_type': stream_type,
                'data': real_time_data,
                'count': len(real_time_data),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Real-time data retrieval failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
    def _calculate_user_stats(self, transactions: List[Dict]) -> Dict:
        """Calculate user statistics from transaction data."""
        if not transactions:
            return {}

        df = pd.DataFrame(transactions)

        return {
            'total_transactions': len(df),
            'total_spent': float(df['sales'].sum()),
            'avg_spent': float(df['sales'].mean()),
            'max_spent': float(df['sales'].max()),
            'min_spent': float(df['sales'].min()),
            'favorite_category': df['category'].mode().iloc[0] if not df.empty else None,
            'avg_discount': float(df['discount'].mean()),
            'avg_quantity': float(df['quantity'].mean()),
            'purchase_frequency': len(df) / 30 # Per day
        }
    
    def _get_sample_real_time_data(self, limit: int) -> List[Dict]:
        """Generate sample real-time data"""
        return [
            {
                'transaction_id': f'TXN-{i:06d}',
                'timestamp': datetime.now().isoformat(),
                'user_id': f'USER-{np.random.randint(1, 100):03d}',
                'amount': round(np.random.uniform(10, 500), 2),
                'status': np.random.choice(['pending', 'completed', 'flagged']),
                'risk_score': round(np.random.uniform(0, 1), 2)
            }
            for i, in range(1, limit + 1)
        ]
    
    def clear_cache(self):
        """Clear the internal cache."""
        self.cache.clear()
        self.get_transaction_data.cache_clear()
        logger.info("🧹 Cache cleared")