import pandas as pd
import requests
from datetime import datetime
import os

def send_telegram(message):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("❌ ERROR: Secrets de Telegram no configurados en GitHub.")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    res = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    if res.status_code == 200:
        print("✅ Mensaje enviado a Telegram con éxito.")
    else:
        print(f"❌ Error de Telegram: {res.text}")

def run_analysis():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Iniciando análisis para {today}...")
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,team"
    data = requests.get(url).json()
    
    # CASO: No hay juegos (All-Star Break)
    if "dates" not in data or not data["dates"] or len(data["dates"][0]["games"]) == 0:
        send_telegram(f"⚾ *MLB Info - {today}*\n\nNo hay juegos oficiales programados para hoy (posible pausa de All-Star o fin de temporada).")
        return

    try:
        intel = pd.read_csv('data/mlb_advanced_intelligence.csv')
        pitchers_db = intel[intel['type'] == 'Pitcher']
    except:
        send_telegram("❌ Error: Base de datos de inteligencia no encontrada.")
        return

    games = data["dates"][0]["games"]
    predictions = []

    for game in games:
        home_team = game["teams"]["home"]["team"]["name"]
        away_team = game["teams"]["away"]["team"]["name"]
        h_p_name = game["teams"]["home"].get("probablePitcher", {}).get("fullName", "TBD")
        a_p_name = game["teams"]["away"].get("probablePitcher", {}).get("fullName", "TBD")
        
        h_siera = pitchers_db[pitchers_db['name'] == h_p_name]['SIERA'].mean() if h_p_name in pitchers_db['name'].values else 4.40
        a_siera = pitchers_db[pitchers_db['name'] == a_p_name]['SIERA'].mean() if a_p_name in pitchers_db['name'].values else 4.40

        win_prob = 50 + ((a_siera - h_siera) * 12) + 5
        win_prob = max(min(win_prob, 92), 8)

        predictions.append({
            "matchup": f"{away_team} @ {home_team}",
            "prob": win_prob,
            "h_p": h_p_name, "a_p": a_p_name
        })

    predictions.sort(key=lambda x: abs(x['prob'] - 50), reverse=True)

    msg = f"⚾ *PROYECCIONES MLB - {today}*\n\n"
    for p in predictions:
        star = "⭐" if p['prob'] > 65 or p['prob'] < 35 else "🔹"
        pick = "LOCAL" if p['prob'] > 50 else "VISITA"
        conf = p['prob'] if p['prob'] > 50 else (100 - p['prob'])
        msg += f"{star} *{p['matchup']}*\n   Pick: `{pick}` ({conf:.1f}%)\n\n"

    best = predictions[0]
    msg += f"🏆 *TOP PICK:* {best['matchup']}"
    send_telegram(msg)

if __name__ == "__main__":
    run_analysis()
