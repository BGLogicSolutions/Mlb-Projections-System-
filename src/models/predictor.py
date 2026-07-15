import logging
import os

import lightgbm as lgb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlb_projections")


class MLBStatsPredictor:
    def __init__(self, data_path="data/mlb_model_ready_dataset.csv"):
        self.data_path = data_path
        self.model = None

    def load_and_clean_data(self):
        if not os.path.exists(self.data_path):
            logger.warning("Dataset no encontrado. Creando uno de prueba...")
            df_dummy = pd.DataFrame(
                {
                    "player_id": [1, 2, 3],
                    "avg": [0.280, 0.300, 0.250],
                    "home_runs": [20, 30, 10],
                    "target": [1, 1, 0],
                }
            )
            df_dummy.to_csv(self.data_path, index=False)
            return df_dummy

        df = pd.read_csv(self.data_path)
        # FIX: Seleccionar solo columnas numéricas para evitar errores en LightGBM
        df_numeric = df.select_dtypes(include=["number", "bool"])
        logger.info(f"Datos cargados y limpiados. Columnas: {list(df_numeric.columns)}")
        return df_numeric

    def train(self):
        df = self.load_and_clean_data()
        if "target" not in df.columns:
            logger.error("Falta la columna 'target' para entrenar.")
            return

        X = df.drop("target", axis=1)
        y = df["target"]

        train_data = lgb.Dataset(X, label=y)
        params = {"objective": "binary", "metric": "auc", "verbose": -1}

        logger.info("Entrenando modelo LightGBM...")
        self.model = lgb.train(params, train_data, num_boost_round=100)
        logger.info("Entrenamiento completado.")

    def run(self):
        logger.info("Iniciando Sistema de Proyecciones MLB...")
        self.train()


if __name__ == "__main__":
    system = MLBStatsPredictor()
    system.run()
