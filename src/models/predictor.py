import pandas as pd
import lightgbm as lgb
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlb_projections")

class MLBStatsPredictor:
    def __init__(self):
        self.data_path = 'data/mlb_model_ready_dataset.csv'
        self.output_dir = 'predictions'
        os.makedirs(self.output_dir, exist_ok=True)

    def get_data(self):
        # Si no hay datos, creamos una base para proyectar
        if not os.path.exists(self.data_path):
            df = pd.DataFrame({
                'player_id': [101, 102, 103, 104, 105],
                'avg': [0.280, 0.300, 0.250, 0.220, 0.310],
                'home_runs': [20, 35, 10, 5, 40],
                'target': [1, 1, 0, 0, 1]
            })
            df.to_csv(self.data_path, index=False)
        return pd.read_csv(self.data_path)

    def generate_daily_projections(self):
        df = self.get_data()
        X = df.select_dtypes(include=['number']).drop('target', axis=1, errors='ignore')
        y = df['target'] if 'target' in df.columns else [0]*len(df)

        # Entrenar y Predecir
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train({'objective': 'binary', 'verbose': -1}, train_data)
        
        preds = model.predict(X)
        
        # Crear DataFrame de resultados
        results = df.copy()
        results['win_probability'] = preds
        results['projection_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Guardar con fecha
        filename = f"{self.output_dir}/projections_{datetime.now().strftime('%Y_%m_%d')}.csv"
        results.to_csv(filename, index=False)
        logger.info(f"✅ Proyecciones guardadas en: {filename}")
        return filename

if __name__ == "__main__":
    system = MLBStatsPredictor()
    system.generate_daily_projections()
