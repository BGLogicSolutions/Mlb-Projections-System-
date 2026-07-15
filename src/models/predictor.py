import pandas as pd
import lightgbm as lgb
import os
import logging

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlb_projections")

class MLBStatsPredictor:
    def __init__(self, data_path='data/mlb_model_ready_dataset.csv'):
        self.data_path = data_path
        self.model = None

    def load_and_clean_data(self):
        if not os.path.exists(self.data_path):
            logger.warning("Dataset no encontrado. Generando datos sintéticos para inicializar...")
            df_dummy = pd.DataFrame({
                'player_id': range(1, 101),
                'avg': [0.250] * 100,
                'home_runs': [15] * 100,
                'target': [0, 1] * 50
            })
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            df_dummy.to_csv(self.data_path, index=False)
            return df_dummy
        
        df = pd.read_csv(self.data_path)
        # MEJORA CRÍTICA: Filtrar solo columnas numéricas para LightGBM
        df_numeric = df.select_dtypes(include=['number', 'bool'])
        logger.info(f"Datos cargados: {df_numeric.shape}")
        return df_numeric

    def train(self):
        df = self.load_and_clean_data()
        if 'target' not in df.columns:
            logger.error("Error: La columna 'target' es necesaria para el entrenamiento.")
            return

        X = df.drop('target', axis=1)
        y = df['target']
        
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05
        }
        
        logger.info("Entrenando modelo de proyecciones MLB...")
        self.model = lgb.train(params, train_data, num_boost_round=50)
        logger.info("¡Modelo entrenado con éxito!")

    def run(self):
        logger.info("=== MLB PROJECTIONS SYSTEM READY ===")
        self.train()

if __name__ == "__main__":
    system = MLBStatsPredictor()
    system.run()
