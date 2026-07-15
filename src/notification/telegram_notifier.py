import os

import requests
from dotenv import load_dotenv

from src.utils.logger import logger

load_dotenv()


def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.error("Faltan credenciales Telegram")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Notificación Telegram enviada")
        return True
    except Exception as e:
        logger.error(f"Error Telegram: {e}")
        return False
