import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from datetime import datetime
from src.utils.logger import logger
from src.notification.telegram_notifier import send_telegram_message

class MLBProjectionSystem:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "lgb_projections.pkl"
        self.model = None
    
    def prepare_features(self, df):
        """Feature engineering básico -> expandir"""
        # Agrega aquí park factors, matchup, recent form, etc.
        feature_cols = [col for col in df.columns if col not in ['target', 'date', 'player_id']]
        X = df[feature_cols]
        y = df.get('target', pd.Series([0]*len(df)))  # Ajusta según tu target (FPPG, HR, etc.)
        return X, y
    
    def train(self, data_path="data/mlb_model_ready_dataset.csv"):
        logger.info("Iniciando entrenamiento...")
        df = pd.read_csv(data_path)
        X, y = self.prepare_features(df)
        
        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1
            }
            model = lgb.train(params, train_data, num_boost_round=100)
            pred = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, pred))
        
        logger.info(f"CV MAE: {sum(mae_scores)/len(mae_scores):.4f}")
        
        # Entrenamiento final
        final_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, final_data, num_boost_round=200)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Modelo guardado en {self.model_path}")
        return self.model
    
    def predict_daily(self):
        """Proyecciones diarias"""
        if not self.model:
            self.model = joblib.load(self.model_path)
        
        # Cargar datos frescos + inferencia
        daily_data = pd.DataFrame()  # Reemplaza con get_daily_games()
        if daily_data.empty:
            logger.warning("No hay datos nuevos")
            return pd.DataFrame()
        
        X_pred, _ = self.prepare_features(daily_data)
        predictions = self.model.predict(X_pred)
        
        results = daily_data.copy()
        results['projection'] = predictions
        results['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Guardar y notificar
        results.to_csv(f"projections_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        
        mensaje = f"🚀 *MLB Projections Actualizadas - {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        mensaje += f"Proyecciones generadas para {len(results)} jugadores/juegos.\n"
        mensaje += "Top performers listos para DFS/Betting."
        send_telegram_message(mensaje)
        
        logger.info("Proyecciones diarias completadas")
        return results

if __name__ == "__main__":
    system = MLBProjectionSystem()
    system.train()  # O cargar si existe
    system.predict_daily()
