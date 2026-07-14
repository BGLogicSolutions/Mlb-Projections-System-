import os
import requests

def send_telegram_message(message):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("❌ Error: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no están configurados.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("✅ Notificación enviada a Telegram.")
    except Exception as e:
        print(f"❌ Error al enviar a Telegram: {e}")
