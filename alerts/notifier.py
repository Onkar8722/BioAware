import requests

# Replace with your actual bot token and chat ID
TELEGRAM_BOT_TOKEN = ""
CHAT_ID = ""


def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, data=data)
        print("Telegram alert sent")
    except Exception as e:
        print(f"Failed to send alert: {e}")
