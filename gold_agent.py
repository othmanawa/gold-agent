from flask import Flask, request
from openai import OpenAI
from dotenv import load_dotenv
import os, requests, json

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})

@app.route("/", methods=["GET"])
def home():
    return "Gold Agent actif ✅"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or request.form.to_dict() or {"raw": request.data.decode("utf-8")}

    prompt = f"""
Tu es mon analyste professionnel spécialisé XAUUSD Gold en M5.

Analyse uniquement ces données TradingView :
{json.dumps(data, indent=2)}

Si le signal est insuffisant, réponds : Aucun trade à prendre actuellement.

Réponds exactement dans ce format :

XAUUSD M5

Action : ACHAT / VENTE / ATTENDRE / AUCUN TRADE
Entrée :
Stop Loss :
TP1 :
TP2 :
TP3 :
Confiance /10 :
Risque :
Raison :
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    message = response.output_text
    send_telegram(message)

    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
