from flask import Flask, request
from openai import OpenAI
from dotenv import load_dotenv
import os, requests, json, csv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def log_signal(data, message):
    file_exists = os.path.exists("trades_log.csv")
    with open("trades_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time", "symbol", "timeframe", "action", "close", "message"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            data.get("symbol", ""),
            data.get("timeframe", ""),
            data.get("action", ""),
            data.get("close", ""),
            message.replace("\n", " | ")
        ])

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text})

@app.route("/", methods=["GET"])
def home():
    return "Gold Agent actif ✅"

@app.route("/webhook", methods=["POST"])
def webhook():   
    data = request.get_json(silent=True) or request.form.to_dict() or {"raw": request.data.decode("utf-8")}
    print("TRADINGVIEW DATA:", data, flush=True)

    prompt = f"""
Tu es Gold Agent, analyste professionnel spécialisé sur XAUUSD M5.

Analyse uniquement les données TradingView suivantes :

{json.dumps(data, indent=2)}

Mission :

- Analyse la tendance.
- Vérifie la cohérence entre EMA20, EMA50, RSI et ATR.
- Ne propose un trade que s'il présente un véritable avantage.
- Si le marché est en range ou le signal est faible, réponds ATTENDRE ou AUCUN TRADE.
- Ne force jamais une position.

Attribue une confiance sur 10 :

9-10 = Setup exceptionnel
8 = Très bon setup
7 = Correct mais prudence
6 ou moins = Pas de trade

Réponds exactement sous ce format :

XAUUSD M5

Action :
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
    log_signal(data, message)
    send_telegram(message)

    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
