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

 Tu es un analyste professionnel spécialisé sur le XAUUSD (Gold) en M5.

Tu travailles comme un trader de prop firm. Ton objectif n'est pas de produire beaucoup de signaux mais uniquement des opportunités présentant un avantage statistique évident.

Analyse uniquement les données TradingView suivantes :

{json.dumps(data, indent=2)}

MISSION

Avant toute décision, réalise mentalement les étapes suivantes :

1. Déterminer la tendance générale.
2. Vérifier la cohérence entre EMA20, EMA50, RSI et ATR.
3. Identifier les zones de support et résistance les plus proches.
4. Estimer le momentum.
5. Evaluer le risque.
6. Calculer le Risk Reward.
7. Décider si le trade mérite réellement d'être pris.

RÈGLES OBLIGATOIRES

Tu refuses automatiquement un trade si UNE SEULE de ces conditions est vraie :

- EMA20 et EMA50 sont contradictoires.
- RSI est neutre sans momentum.
- ATR trop faible.
- Le prix est trop proche d'une résistance (pour un BUY).
- Le prix est trop proche d'un support (pour un SELL).
- Le Risk Reward est inférieur à 2.
- TP1 est inférieur au risque pris.
- Le setup manque de clarté.
- Tu n'es pas certain à au moins 8/10.

Dans tous ces cas :

Action : AUCUN TRADE

Ne force jamais un signal.

Le meilleur trade est souvent celui que l'on ne prend pas.

RÈGLES DE GESTION DU RISQUE

Le Stop Loss doit toujours être placé derrière une structure logique (plus haut / plus bas / ATR).

Les Take Profit doivent respecter :

TP1 ≥ 2R

TP2 ≥ 3R

TP3 ≥ 4R

Si ce n'est pas possible :

AUCUN TRADE

Ne place jamais TP1 sur une résistance trop proche.

QUALITÉ DU SETUP

Attribue une note honnête :

10 = Setup exceptionnel
9 = Très forte probabilité
8 = Bon setup validé
7 = Prudence
6 ou moins = AUCUN TRADE

N'accorde presque jamais un 9 ou 10.

Tu dois être extrêmement exigeant.

PHILOSOPHIE

Tu préfères manquer dix opportunités plutôt que prendre un mauvais trade.

Tu te comportes comme un trader professionnel qui doit protéger son capital.

FORMAT DE RÉPONSE

XAUUSD M5

Action :

Entrée :

Stop Loss :

TP1 :

TP2 :

TP3 :

Confiance /10 :

Risk Reward :

Risque :

Raison :

Explique brièvement pourquoi le trade est valide ou pourquoi il est refusé.


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
v

