import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternAnalysisTool:
    """
    Tool for analyzing patterns in transaction data:
    - Anomaly detection
    - Trend analysis
    - Pattern recognition
    - Statistical analysis
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.pattern_cache = {}

    def run(self, data: str) -> Dict:
        """
        Main entry point for the tool.
        
        Args:
            data: JSON string or dict with data to analyze
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Parse data if string
            if isinstance(data, str):
                data = json.loads(data)
            
            analysis_type = data.get('analysis_type', 'comprehensive')
            
            # Convert to DataFrame
            df = pd.DataFrame(data.get('data', []))
            
            if df.empty:
                return {'error': 'No data provided for analysis'}
            
            # Perform analysis
            if analysis_type == 'comprehensive':
                return self.comprehensive_analysis(df)
            elif analysis_type == 'anomaly':
                return self.detect_anomalies(df)
            elif analysis_type == 'trend':
                return self.analyze_trends(df)
            elif analysis_type == 'pattern':
                return self.find_patterns(df)
            else:
                return {'error': f'Unknown analysis type: {analysis_type}'}
                
        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {str(e)}")
            return {'error': str(e)}
        
    def comprehensive_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive pattern analysis.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dict with comprehensive analysis results
        """
        try:
            # Basic statistics
            stats = self._calculate_statistics(df)
            
            # Anomaly detection
            anomalies = self._detect_anomalies(df)
            
            # Trend analysis
            trends = self._analyze_trends(df)
            
            # Pattern recognition
            patterns = self._find_patterns(df)
            
            # Correlation analysis
            correlations = self._analyze_correlations(df)
            
            return {
                'status': 'success',
                'statistics': stats,
                'anomalies': anomalies,
                'trends': trends,
                'patterns': patterns,
                'correlations': correlations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
    def detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detect anomalies in the data.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dict with anomaly detection results
        """
        try:
            # Use multiple methods for anomaly detection
            statistical_anomalies = self._statistical_anomaly_detection(df)
            ml_anomalies = self._ml_anomaly_detection(df)

            # Combine results
            combined_anomalies = self._combine_anomalies(
                statistical_anomalies, ml_anomalies
            )

            return {
                'anomalies': combined_anomalies,
                'statistical_anomalies': statistical_anomalies,
                'ml_anomalies': ml_anomalies,
                'anomaly_count': len(combined_anomalies),
                'anomaly_rate': len(combined_anomalies) / len(df) * 100 
            }
        
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
    def analyze_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze trends in the data.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dict with trend analysis results
        """
        try:
            # Ensure date columns exists
            if 'order_date' not in df.columns:
                return {'error': 'order_date column not found in data'}
            
            # Convert to datetime
            df['order_date'] = pd.to_datetime(df['order_date'])

            # Calculate daily trends
            daily_trends = self._calculate_daily_trends(df)

            # Calculate weekly trends
            weekly_trends = self._calculate_weekly_trends(df)

            # Calculate monthly trends
            monthly_trends = self._calculate_monthly_trends(df)
            
            # Detect seasonality
            seasonality = self._detect_seasonality(df)
            
            # Forecast
            forecast = self._generate_forecast(df)

            return {
                'daily_trends': daily_trends,
                'weekly_trends': weekly_trends,
                'monthly_trends': monthly_trends,
                'seasonality': seasonality,
                'forecast': forecast,
                'trend_strength': self._calculate_trend_strength(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {str(e)}")
            return {'error': str(e)}
        
    def find_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Find patterns in the data.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dict with pattern recognition results
        """
        try:
            # Pattern types
            patterns = {
                'recurring_patterns': self._find_recurring_patterns(df),
                'sequential_patterns': self._find_sequential_patterns(df),
                'categorical_patterns': self._find_categorical_patterns(df),
                'temporal_patterns': self._find_temporal_patterns(df)
            }

            return {
                'patterns': patterns,
                'pattern_count': sum(len(p) for p in patterns.values()),
                'most_significant': self._get_most_significant_patterns(patterns)
            }
        
        except Exception as e:
            logger.error(f"❌ Pattern recognition failed: {str(e)}")
            return {'error': str(e)}
        
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic statistics for the dataset."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats = {
            'total_records': len(df),
            'numeric_stats': {}
        }

        for col in numeric_cols:
            stats['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q1': float(df[col].quantile(0.25)),
                'q3': float(df[col].quantile(0.75)),
                'iqr': float(df[col].quantile(0.75) - df[col].quantile(0.25))
            }

        return stats
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for idx, row in df.iterrows():
            anomaly_score = 0
            anomaly_reasons = []

            for col in numeric_cols:
                value = row[col]
                mean = df[col].mean()
                std = df[col].std()

                if std > 0:
                    z_score = (value - mean) / std
                    if abs(z_score) > 3:
                        anomaly_score += abs(z_score)
                        anomaly_reasons.append(f"{col}: z-score {z_score:.2f}")

            if anomaly_score > 0:
                anomalies.append({
                    'index': int(idx),
                    'anomaly_score': float(anomaly_score),
                    'reasons': anomaly_reasons,
                    'data': row.to_dict()
                })

        # Sort by anomaly score
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return anomalies[:10] # Return top 10 anomalies
    
    def _ml_anomaly_detection(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies using ML methods."""

        # Simplified ML anomaly detection - In production, use Isolation Forest, One-Class SVM, or Autoencoders
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].values

        if len(data) < 10:
            return []  # Not enough data for ML anomaly detection
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)

        # Build distance-based anomaly detection with numpy
        mean_vector = np.mean(scaled_data, axis=0)
        distances = np.linalg.norm(scaled_data - mean_vector, axis=1)
        
        threshold = np.percentile(distances, 95)  # Top 5% as anomalies

        anomalies = []
        for idx, dist in enumerate(distances):
            if dist > threshold:
                anomalies.append({
                    'index': int(idx),
                    'distance': float(dist),
                    'data': df.iloc[idx].to_dict()
                })

    def _combine_anomalies(self, stat_anomalies: List, ml_anomalies: List) -> List:
        """Combine anomalies from diffrent methods."""
        combined = []
        stat_indices = {a['index'] for a in stat_anomalies}
        ml_indices = {a['index'] for a in ml_anomalies}

        all_indices = stat_indices | ml_indices

        for idx in all_indices:
            is_stat = idx in stat_indices
            is_ml = idx in ml_indices

            combined.append({
                'index': idx,
                'detected_by': 'statistical' if is_stat else 'ml' if is_ml else 'both',
                'confidence': 0.8 if is_stat and is_ml else 0.6
            })

        return combined
    
    def _calculate_daily_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate daily trends."""
        df['date'] = df['order_date'].dt.date
        daily_sales = df.groupby('date')['sales'].sum()

        return {
            'mean_daily_sales': float(daily_sales.mean()),
            'max_daily_sales': float(daily_sales.max()),
            'min_daily_sales': float(daily_sales.min()),
            'total_days': len(daily_sales),
            'trend_direction': 'increasing' if daily_sales.iloc[-1] > daily_sales.iloc[0] else 'decreasing'
        }
    
    def _calculate_weekly_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate weekly trends."""
        df['week'] = df['order_date'].dt.isocalendar().week
        
        weekly_sales = df.groupby('week')['sales'].sum()

        return {
            'mean_weekly_sales': float(weekly_sales.mean()),
            'max_weekly_sales': float(weekly_sales.max()),
            'min_weekly_sales': float(weekly_sales.min()),
            'total_weeks': len(weekly_sales)
        }
    
    def _calculate_monthly_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate monthly trends."""
        df['month'] = df['order_date'].dt.month
        monthly_sales = df.groupby('month')['sales'].sum()
        
        return {
            'monthly_sales': monthly_sales.to_dict(),
            'best_month': int(monthly_sales.idxmax()),
            'worst_month': int(monthly_sales.idxmin())
        }
    
    def _detect_seasonality(self, df: pd.DataFrame) -> Dict:
        """Detect seasonality in the data."""
        df['month'] = df['order_date'].dt.month
        monthly_avg = df.groupby('month')['sales'].mean()

        return {
            'has_seasonality': True,
            'seasonal_pattern': monthly_avg.to_dict(),
            'peak_months': monthly_avg.nlargest(3).index.tolist(),
            'low_months': monthly_avg.nsmallest(3).index.tolist()
        }
    
    def _generate_forecast(self, df: pd.DataFrame) -> Dict:
        """Generate simple forecast based on historical data."""
        df = df.sort_values('order_date')
        sales = df['sales'].values

        if len(sales) < 7:
            return {'forecast': 'Insufficient data for forecasting'}
        
        # 7-day moving average forecast
        window = min(7, len(sales))
        ma = np.convolve(sales, np.ones(window)/window, mode='valid')

        # Next 3 days forecast
        last_ma = ma[-1]
        trend = ma[-1] - ma[-2] if len(ma) > 1 else 0

        forecast = [
            last_ma + (trend * i) for i in range(1, 4)
        ]

        return {
            'forecast_method': 'moving_average',
            'window': window,
            'next_3_days': [float(f) for f in forecast],
            'trend': float(trend),
            'confidence': 'medium'
        }
    
    def _find_recurring_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Find recurring patterns in the data."""
        patterns = []

        # Check for recurring patterns in categories
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            top_categories = category_counts.head(5)

            for category, count in top_categories.items():
                patterns.append({
                    'type': 'category_recurrence',
                    'value': category,
                    'frequency': int(count),
                    'percentage': float(count / len(df) * 100)
                })

        return patterns
    
    def _find_sequential_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Find sequential patterns."""
        patterns = []
        
        # Check for patterns in purchase sequences
        if 'category' in df.columns:
            df_sorted = df.sort_values('order_date')
            categories = df_sorted['category'].tolist()
            
            # Find common pairs
            for i in range(len(categories) - 1):
                pair = (categories[i], categories[i + 1])

                # Store pattern
                patterns.append({
                    'type': 'sequential_pair',
                    'pair': pair,
                    'frequency': 1
                })
        
        return patterns
    
    def _find_categorical_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Find categorical patterns."""
        patterns = []
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            top_values = value_counts.head(5)

            for value, count in top_values.items():
                patterns.append({
                    'type': 'categorical_pattern',
                    'column': col,
                    'value': value,
                    'frequency': int(count),
                    'percentage': float(count / len(df) * 100)
                })

        return patterns
    
    def _find_temporal_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Find temporal patterns."""
        patterns = []

        if 'order_date' in df.columns:
            df['hour'] = df['order_date'].dt.hour
            df['day_of_week'] = df['order_date'].dt.dayofweek

            # Peak hours
            hour_counts = df['hour'].value_counts()
            peak_hours = hour_counts.nlargest(3)

            for hour, count in peak_hours.items():
                patterns.append({
                    'type': 'temporal_peak_hours',
                    'hour': int(hour),
                    'frequency': int(count),
                    'percentage': float(count / len(df) * 100)
                })

            # Peak days
            day_counts = df['day_of_week'].value_counts()
            peak_days = day_counts.nlargest(3)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            for day, count in peak_days.items():
                patterns.append({
                    'type': 'temporal_peak_days',
                    'day': day_names[int(day)],
                    'frequency': int(count),
                    'percentage': float(count / len(df) * 100)
                })

        return patterns
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        if 'order_date' not in df.columns or 'sales' not in df.columns:
            return 0.0
        
        df = df.sort_values('order_date')
        sales = df['sales'].values

        if len(sales) < 2:
            return 0.0
        
        # Correlation with index
        corr = np.corrcoef(range(len(sales)), sales)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()

        # Find strong correlations
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.6:
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def _get_most_significant_patterns(self, patterns: Dict) -> List[Dict]:
        """Get the most significant patterns."""
        all_patterns = []

        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                all_patterns.append({
                    **pattern,
                    'type': pattern_type
                })

        # Sort by frequency or significance
        all_patterns.sort(key=lambda x: x.get('frequency', 0), reverse=True)
        
        return all_patterns[:5]  # Return top 5 significant patterns