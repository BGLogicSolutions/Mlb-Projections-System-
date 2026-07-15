import pandas as pd
import lightgbm as lgb
import os

def train():
    if not os.path.exists('data/mlb_advanced_intelligence.csv'): return
    df = pd.read_csv('data/mlb_advanced_intelligence.csv')
    X = df.select_dtypes(include=['number']).fillna(0)
    # Target: Jugadores con rendimiento por encima del promedio (Top 30%)
    y = (X.iloc[:, 0] > X.iloc[:, 0].quantile(0.7)).astype(int)
    
    model = lgb.train({'objective': 'binary', 'metric': 'auc', 'verbose': -1}, 
                     lgb.Dataset(X, label=y), num_boost_round=200)
    os.makedirs('models', exist_ok=True)
    model.save_model('models/mlb_sabermetric_model.txt')
    print("🧠 Modelo Sabermétrico Entrenado.")

if __name__ == "__main__":
    train()
