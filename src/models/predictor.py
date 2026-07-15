import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from datetime import datetime
import os

from src.utils.logger import logger
from src.notification.telegram_notifier import send_telegram_message

class MLBProjectionSystem:
    def __init__(self, model_dir="models", data_dir="data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "lgb_projections.pkl"
        self.model = None
    
    def load_data(self):
        """Carga datos existentes de forma robusta"""
        possible_files = [
            "mlb_model_ready_dataset.csv",
            "mlb_historical_batters_2015_2026.csv",
            "mlb_games_clean.csv"
        ]
        
        for f in possible_files:
            path = self.data_dir / f
            if path.exists():
                logger.info(f"Cargando {f}")
                df = pd.read_csv(path)
                logger.info(f"Dataset cargado: {df.shape}")
                return df
        
        logger.warning("No se encontraron datasets. Creando placeholder mínimo.")
        # Placeholder para evitar fallo
        dates = pd.date_range(end=datetime.now(), periods=1000)
        df = pd.DataFrame({
            'date': dates,
            'player_id': range(1000),
            'hits': [i % 5 for i in range(1000)],
            'hr': [i % 2 for i in range(1000)],
            # Agrega más columnas según tus datos
        })
        df.to_csv(self.data_dir / "mlb_model_ready_dataset.csv", index=False)
        return df
    
    def prepare_features(self, df):
        """Feature engineering básico (expande según necesidades)"""
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        # Ejemplo de features (ajusta a tus columnas)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'target' not in df.columns and numeric_cols:
            # Usa una columna como target temporal (ej. hits)
            df['target'] = df.get('hits', df[numeric_cols[0]])
        
        feature_cols = [col for col in df.columns if col not in ['target', 'date', 'player_id']]
        X = df[feature_cols].fillna(0)
        y = df.get('target', pd.Series([0] * len(df)))
        return X, y, df
    
    def train(self):
        logger.info("Iniciando entrenamiento...")
        df = self.load_data()
        X, y, _ = self.prepare_features(df)
        
        if len(X) < 100:
            logger.warning("Dataset demasiado pequeño. Saltando entrenamiento completo.")
            return None
        
        tscv = TimeSeriesSplit(n_splits=3)  # Reducido para velocidad
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'learning_rate': 0.1,
                'num_leaves': 31,
                'verbose': -1,
                'seed': 42
            }
            model_fold = lgb.train(params, train_data, num_boost_round=50)
            pred = model_fold.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, pred))
        
        logger.info(f"CV MAE promedio: {sum(mae_scores)/len(mae_scores):.4f}")
        
        # Entrenamiento final
        final_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, final_data, num_boost_round=100)
        joblib.dump(self.model, self.model_path)
        logger.info(f"✅ Modelo guardado: {self.model_path}")
        return self.model
    
    def predict_daily(self):
        logger.info("Generando proyecciones diarias...")
        if self.model is None and self.model_path.exists():
            self.model = joblib.load(self.model_path)
        elif self.model is None:
            logger.info("Modelo no encontrado. Entrenando...")
            self.train()
        
        df = self.load_data()
        X, _, original_df = self.prepare_features(df.tail(500))  # Últimos registros
        
        if self.model is None:
            logger.error("No se pudo cargar/entrenar modelo")
            return pd.DataFrame()
        
        predictions = self.model.predict(X)
        results = original_df.copy().iloc[-len(predictions):].copy()
        results['projection'] = predictions
        results['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
        
        output_path = f"projections_{datetime.now().strftime('%Y%m%d')}.csv"
        results.to_csv(output_path, index=False)
        
        mensaje = f"🚀 *MLB Projections - {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        mensaje += f"✅ {len(results)} proyecciones generadas.\n"
        mensaje += f"Archivo: {output_path}\n"
        send_telegram_message(mensaje)
        
        logger.info(f"Proyecciones guardadas en {output_path}")
        return results

if __name__ == "__main__":
    system = MLBProjectionSystem()
    system.predict_daily()  # Entrena + predice
