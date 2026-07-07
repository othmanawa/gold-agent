from flask import Flask, request
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timezone
import csv
import json
import os
import re
import requests
from typing import Any, Dict, Optional, Tuple

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

MIN_CONFIDENCE = 8
MIN_RR_TP1 = 2.0
MIN_RR_TP2 = 3.0
MIN_RR_TP3 = 4.0
ATR_RISK_MULTIPLIER = 1.0
MIN_ATR = 0.8
MAX_ATR = 12.0


def to_float(value: Any) -> Optional[float]:
    """Convert TradingView values to float, including French comma decimals."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "")
    text = text.replace("\u202f", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def format_price(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM_TOKEN or TELEGRAM_CHAT_ID missing", flush=True)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        timeout=15,
    )
    print("TELEGRAM STATUS:", response.status_code, response.text[:300], flush=True)


def log_signal(data: Dict[str, Any], message: str, decision: Dict[str, Any]) -> None:
    file_exists = os.path.exists("trades_log.csv")
    with open("trades_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "time_utc",
                "symbol",
                "timeframe",
                "action_received",
                "decision",
                "entry",
                "sl",
                "tp1",
                "tp2",
                "tp3",
                "risk_reward_tp1",
                "confidence",
                "close",
                "reason",
                "message",
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            data.get("symbol", ""),
            data.get("timeframe", ""),
            data.get("action", ""),
            decision.get("action", ""),
            decision.get("entry", ""),
            decision.get("sl", ""),
            decision.get("tp1", ""),
            decision.get("tp2", ""),
            decision.get("tp3", ""),
            decision.get("rr_tp1", ""),
            decision.get("confidence", ""),
            data.get("close", ""),
            decision.get("reason", ""),
            message.replace("\n", " | "),
        ])


def build_rejection(reason: str, data: Dict[str, Any], confidence: int = 5) -> Tuple[str, Dict[str, Any]]:
    message = f"""XAUUSD M5

Action : AUCUN TRADE
Entrée : N/A
Stop Loss : N/A
TP1 : N/A
TP2 : N/A
TP3 : N/A
Confiance /10 : {confidence}
Risk Reward : N/A
Risque : Évité
Raison : {reason}
""".strip()
    decision = {
        "action": "AUCUN TRADE",
        "confidence": confidence,
        "reason": reason,
        "entry": "N/A",
        "sl": "N/A",
        "tp1": "N/A",
        "tp2": "N/A",
        "tp3": "N/A",
        "rr_tp1": "N/A",
    }
    return message, decision


def analyse_setup(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    action = str(data.get("action", "")).upper().strip()
    entry = to_float(data.get("close"))
    ema20 = to_float(data.get("ema20"))
    ema50 = to_float(data.get("ema50"))
    ema200 = to_float(data.get("ema200"))
    rsi = to_float(data.get("rsi"))
    atr = to_float(data.get("atr"))
    volume = to_float(data.get("volume"))
    volume_ma = to_float(data.get("volume_ma"))
    support = to_float(data.get("support"))
    resistance = to_float(data.get("resistance"))

    required = {
        "action": action,
        "close": entry,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "atr": atr,
    }
    missing = [key for key, value in required.items() if value in (None, "")]
    if missing:
        return None, "Données TradingView insuffisantes ou ancien format d'alerte. Champs manquants : " + ", ".join(missing)

    if action not in {"BUY", "SELL"}:
        return None, f"Action reçue invalide : {action}."

    if atr < MIN_ATR:
        return None, f"ATR trop faible ({format_price(atr)}). Mouvement potentiel insuffisant."
    if atr > MAX_ATR:
        return None, f"ATR trop élevé ({format_price(atr)}). Volatilité excessive, risque de mèches violent."

    risk = atr * ATR_RISK_MULTIPLIER

    if action == "BUY":
        if not (ema20 > ema50):
            return None, "BUY refusé : EMA20 n'est pas au-dessus de EMA50."
        if rsi < 52 or rsi > 70:
            return None, f"BUY refusé : RSI non optimal ({rsi:.1f}). Il doit montrer du momentum sans surachat."
        if ema200 is not None and entry < ema200 and (ema200 - entry) < (2 * risk):
            return None, f"BUY refusé : prix sous EMA200, résistance dynamique trop proche ({format_price(ema200)})."

        sl = entry - risk
        tp1 = entry + (MIN_RR_TP1 * risk)
        tp2 = entry + (MIN_RR_TP2 * risk)
        tp3 = entry + (MIN_RR_TP3 * risk)

        if resistance is not None and resistance > entry and resistance < tp1:
            return None, (
                f"BUY refusé : résistance trop proche à {format_price(resistance)}. "
                f"Impossible d'atteindre TP1 minimum 2R ({format_price(tp1)})."
            )

        confidence = 8
        if ema200 is not None and entry > ema200:
            confidence += 1
        if volume is not None and volume_ma is not None and volume > volume_ma:
            confidence += 1
        confidence = min(confidence, 10)

    else:  # SELL
        if not (ema20 < ema50):
            return None, "SELL refusé : EMA20 n'est pas sous EMA50."
        if rsi > 48 or rsi < 30:
            return None, f"SELL refusé : RSI non optimal ({rsi:.1f}). Il doit montrer du momentum vendeur sans excès."
        if ema200 is not None and entry > ema200 and (entry - ema200) < (2 * risk):
            return None, f"SELL refusé : prix au-dessus de EMA200, support dynamique trop proche ({format_price(ema200)})."

        sl = entry + risk
        tp1 = entry - (MIN_RR_TP1 * risk)
        tp2 = entry - (MIN_RR_TP2 * risk)
        tp3 = entry - (MIN_RR_TP3 * risk)

        if support is not None and support < entry and support > tp1:
            return None, (
                f"SELL refusé : support trop proche à {format_price(support)}. "
                f"Impossible d'atteindre TP1 minimum 2R ({format_price(tp1)})."
            )

        confidence = 8
        if ema200 is not None and entry < ema200:
            confidence += 1
        if volume is not None and volume_ma is not None and volume > volume_ma:
            confidence += 1
        confidence = min(confidence, 10)

    setup = {
        "action": action,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "rr_tp1": MIN_RR_TP1,
        "rr_tp2": MIN_RR_TP2,
        "rr_tp3": MIN_RR_TP3,
        "confidence": confidence,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "atr": atr,
        "volume": volume,
        "volume_ma": volume_ma,
        "support": support,
        "resistance": resistance,
    }
    return setup, None


def build_prompt(data: Dict[str, Any], setup: Dict[str, Any]) -> str:
    return f"""
Tu es Gold Agent, analyste professionnel spécialisé sur XAUUSD Gold en M5.

Tu dois produire une analyse courte, claire et exploitable pour Telegram.

IMPORTANT : les niveaux ci-dessous ont déjà été calculés avec une règle stricte de Risk/Reward.
Tu n'as PAS le droit de modifier l'entrée, le Stop Loss, TP1, TP2 ou TP3.

Données TradingView :
{json.dumps(data, indent=2, ensure_ascii=False)}

Plan validé par le filtre de risque :
- Action : {setup['action']}
- Entrée : {format_price(setup['entry'])}
- Stop Loss : {format_price(setup['sl'])}
- TP1 : {format_price(setup['tp1'])} (2R minimum)
- TP2 : {format_price(setup['tp2'])} (3R minimum)
- TP3 : {format_price(setup['tp3'])} (4R minimum)
- Risque en prix : {format_price(setup['risk'])}
- Confiance : {setup['confidence']}/10

Règles :
- Ne promets jamais un gain.
- Reste professionnel.
- Mentionne que le setup reste risqué.
- Sois bref.
- Ne donne pas de TP inférieur à 2R.

Réponds exactement dans ce format :

XAUUSD M5

Action : {setup['action']}
Entrée : {format_price(setup['entry'])}
Stop Loss : {format_price(setup['sl'])}
TP1 : {format_price(setup['tp1'])}
TP2 : {format_price(setup['tp2'])}
TP3 : {format_price(setup['tp3'])}
Confiance /10 : {setup['confidence']}
Risk Reward : 1:2 minimum, objectif final 1:4
Risque :
Raison :
""".strip()


@app.route("/", methods=["GET"])
def home():
    return "Gold Agent actif ✅"


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or request.form.to_dict() or {"raw": request.data.decode("utf-8")}
    print("TRADINGVIEW DATA:", data, flush=True)

    setup, rejection_reason = analyse_setup(data)

    if rejection_reason:
        message, decision = build_rejection(rejection_reason, data, confidence=5)
        log_signal(data, message, decision)
        send_telegram(message)
        return {"status": "ok", "decision": "AUCUN TRADE", "reason": rejection_reason}

    prompt = build_prompt(data, setup)

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        message = response.output_text.strip()
    except Exception as exc:
        print("OPENAI ERROR:", repr(exc), flush=True)
        message = f"""XAUUSD M5

Action : {setup['action']}
Entrée : {format_price(setup['entry'])}
Stop Loss : {format_price(setup['sl'])}
TP1 : {format_price(setup['tp1'])}
TP2 : {format_price(setup['tp2'])}
TP3 : {format_price(setup['tp3'])}
Confiance /10 : {setup['confidence']}
Risk Reward : 1:2 minimum, objectif final 1:4
Risque : Modéré à élevé
Raison : Setup validé par les filtres EMA/RSI/ATR/RR. Erreur OpenAI temporaire, niveaux calculés automatiquement.
""".strip()

    decision = {
        "action": setup["action"],
        "confidence": setup["confidence"],
        "reason": "Setup validé par filtres internes + analyse IA",
        "entry": format_price(setup["entry"]),
        "sl": format_price(setup["sl"]),
        "tp1": format_price(setup["tp1"]),
        "tp2": format_price(setup["tp2"]),
        "tp3": format_price(setup["tp3"]),
        "rr_tp1": setup["rr_tp1"],
    }

    log_signal(data, message, decision)
    send_telegram(message)

    return {"status": "ok", "decision": setup["action"]}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
