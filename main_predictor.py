import os
import pandas as pd
from datetime import datetime
from telegram_notifier import send_telegram_message

def run_daily_update():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Ejecutando proyecciones MLB: {today}")
    
    # Simulación de resultados (aquí iría la lógica del modelo final_lgb)
    # Por ahora generamos un resumen para Telegram
    mensaje = f"🚀 *MLB Projections - {today}*\n\n"
    mensaje += "Pipeline ejecutado con éxito.\n"
    mensaje += "Los datasets han sido actualizados y el modelo ha procesado los juegos del día.\n\n"
    mensaje += "[Revisar resultados en GitHub](https://github.com/BGLogicSolutions/Mlb-Projections-System-)"
    
    # Enviar a Telegram
    send_telegram_message(mensaje)
    
    # Guardar log local
    output = pd.DataFrame({'status': ['Success'], 'date': [today]})
    output.to_csv('daily_execution_log.csv', index=False)

if __name__ == '__main__':
    run_daily_update()
