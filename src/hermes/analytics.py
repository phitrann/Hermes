"""
Analytics, ML and recommendations for Hermes
"""
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class HermesAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.prediction_model = None
        self.encoders = {}

    def get_summary_stats(self):
        total = len(self.df)
        delayed = int((self.df['delay_minutes'] > 0).sum()) if 'delay_minutes' in self.df.columns else 0
        on_time = int(total - delayed)
        stats = {
            'total_shipments': int(total),
            'delayed_shipments': delayed,
            'on_time_shipments': on_time,
            'on_time_rate': (on_time / total) if total > 0 else 0.0,
            'delay_rate': (delayed / total) if total > 0 else 0.0,
            'avg_delay_minutes': self.df[self.df['delay_minutes'] > 0]['delay_minutes'].mean() if delayed > 0 else 0.0,
            'avg_delivery_time': self.df['delivery_time'].mean() if 'delivery_time' in self.df.columns else None,
            'date_range': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"
        }
        if 'delay_minutes' in self.df.columns:
            delay_values = self.df[self.df['delay_minutes'] > 0]['delay_minutes']
            stats['avg_delay_minutes'] = float(delay_values.mean()) if len(delay_values) > 0.0 else 0.0
            stats['median_delay_minutes'] = float(delay_values.median()) if len(delay_values) > 0.0 else 0.0
        else:
            stats['avg_delay_minutes'] = 0.0
            stats['median_delay_minutes'] = 0.0
            
        if 'cost' in self.df.columns:
            stats['total_cost'] = f"${self.df['cost'].sum():,.2f}"
        return stats

    def train_prediction_model(self):
        try:
            # Validate required columns
            required_cols = ['date', 'delay_minutes', 'route', 'warehouse']
            missing = [col for col in required_cols if col not in self.df.columns]
            if missing:
                logger.error(f"Missing required columns for prediction: {missing}")
                return None
            
            # Remove rows with NaN in critical columns
            df_model = self.df[required_cols].dropna().copy()
            
            if len(df_model) < 10:
                logger.error(f"Insufficient data for training: only {len(df_model)} valid rows")
                return None
            
            df_model['day_of_week'] = df_model['date'].dt.dayofweek
            df_model['week_of_year'] = df_model['date'].dt.isocalendar().week
            df_model['month'] = df_model['date'].dt.month
            
            le_route = LabelEncoder()
            le_warehouse = LabelEncoder()
            df_model['route_encoded'] = le_route.fit_transform(df_model['route'])
            df_model['warehouse_encoded'] = le_warehouse.fit_transform(df_model['warehouse'])
            self.encoders = {'route': le_route, 'warehouse': le_warehouse}
            
            features = ['route_encoded', 'warehouse_encoded', 'day_of_week', 'week_of_year', 'month']
            X = df_model[features]
            y = df_model['delay_minutes']
            
            self.prediction_model = LinearRegression()
            self.prediction_model.fit(X, y)
            predictions = self.prediction_model.predict(X)
            r2 = r2_score(y, predictions)
            rmse = float(np.sqrt(mean_squared_error(y, predictions)))
            logger.info(f"Model trained: R²={r2:.4f}, RMSE={rmse:.2f} (trained on {len(df_model)} samples)")
            return {'r2_score': r2, 'rmse': rmse, 'feature_importance': dict(zip(features, self.prediction_model.coef_))}
        except Exception as e:
            logger.exception(f"Model training failed: {e}")
            return None

    def predict_next_week(self):
        if self.prediction_model is None:
            self.train_prediction_model()
        if self.prediction_model is None:
            return None
        next_week_start = self.df['date'].max() + timedelta(days=1)
        predictions = []
        for day_offset in range(7):
            date = next_week_start + timedelta(days=day_offset)
            for route in self.df['route'].unique():
                for warehouse in self.df['warehouse'].unique():
                    # Create DataFrame with proper feature names to avoid sklearn warning
                    features_df = pd.DataFrame([[
                        self.encoders['route'].transform([route])[0],
                        self.encoders['warehouse'].transform([warehouse])[0],
                        date.dayofweek,
                        date.isocalendar()[1],
                        date.month
                    ]], columns=['route_encoded', 'warehouse_encoded', 'day_of_week', 'week_of_year', 'month'])
                    pred = self.prediction_model.predict(features_df)[0]
                    predictions.append(max(0, pred))
        return {
            'predicted_avg_delay': float(np.mean(predictions)),
            'predicted_median': float(np.median(predictions)),
            'confidence_interval': (float(np.percentile(predictions, 25)), float(np.percentile(predictions, 75))),
            'forecast_period': f"{next_week_start.date()} to {(next_week_start + timedelta(days=6)).date()}"
        }

    def generate_recommendations(self):
        recs = []
        if 'warehouse' in self.df.columns:
            wh_perf = self.df.groupby('warehouse').agg({
                'delivery_time': 'mean',
                'delay_minutes': 'mean',
                'on_time': 'mean'
            }).reset_index()
            if not wh_perf.empty:
                best_idx = wh_perf['delivery_time'].idxmin()
                worst_idx = wh_perf['delivery_time'].idxmax()
                recs.append({
                    'category': 'Best Practice',
                    'priority': 'High',
                    'finding': f"{wh_perf.loc[best_idx,'warehouse']} best delivery ({wh_perf.loc[best_idx,'delivery_time']:.2f} days)",
                    'action': f"Document and replicate processes"
                })
                recs.append({
                    'category': 'Needs Improvement',
                    'priority': 'High',
                    'finding': f"{wh_perf.loc[worst_idx,'warehouse']} slowest delivery ({wh_perf.loc[worst_idx,'delivery_time']:.2f} days)",
                    'action': f"Conduct process audit"
                })
        if 'delay_reason' in self.df.columns and (self.df['delay_minutes'] > 0).any():
            delay_impact = self.df[self.df['delay_minutes'] > 0].groupby('delay_reason').agg({'id':'count','delay_minutes':'mean'}).sort_values('id', ascending=False)
            if not delay_impact.empty:
                top = delay_impact.index[0]
                recs.append({'category':'Operational','priority':'Critical','finding':f"{top} most frequent delay","action":"Investigate and mitigate"})
        if 'route' in self.df.columns:
            route_perf = self.df.groupby('route')['on_time'].mean().sort_values()
            if not route_perf.empty:
                worst_route = route_perf.index[0]
                recs.append({'category':'Route Optimization','priority':'Medium','finding':f"{worst_route} low on-time","action":"Review schedule/capacity"})
        return recs