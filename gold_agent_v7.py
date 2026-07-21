"""
GOLD AGENT V7 PRO
=================
Moteur de signaux XAUUSD basé sur un score probabiliste.

Architecture conservée :
TradingView -> Webhook Flask -> Moteur Python -> Telegram

OpenAI est facultatif et sert uniquement à reformuler l'analyse technique.
Aucune donnée de prix, décision ou valeur de risque n'est calculée par OpenAI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import logging
import math
import os
import threading

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from openai import OpenAI


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

load_dotenv()

APP_NAME = "GOLD AGENT V7 PRO"
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = "5"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

PORT = int(os.getenv("PORT", "5000"))
LOG_FILE = os.getenv("TRADE_LOG_FILE", "trades_log.csv")

# Seuils de décision. Le score brut maximal est de 15 points.
MAX_MARKET_SCORE = 15.0
PREMIUM_SCORE = float(os.getenv("PREMIUM_SCORE", "12.0"))
VALID_SCORE = float(os.getenv("VALID_SCORE", "9.0"))
WATCH_SCORE = float(os.getenv("WATCH_SCORE", "6.5"))

# Gestion du risque.
MIN_RR = float(os.getenv("MIN_RR", "2.0"))
TP2_RR = float(os.getenv("TP2_RR", "3.0"))
TP3_RR = float(os.getenv("TP3_RR", "4.0"))
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "1.20"))
STRUCTURE_ATR_BUFFER = float(os.getenv("STRUCTURE_ATR_BUFFER", "0.15"))
MIN_TARGET_ATR = float(os.getenv("MIN_TARGET_ATR", "1.50"))
MAX_STOP_ATR = float(os.getenv("MAX_STOP_ATR", "3.00"))

# Anti-spam.
TRADE_COOLDOWN_MINUTES = int(os.getenv("TRADE_COOLDOWN_MINUTES", "45"))
WATCH_COOLDOWN_MINUTES = int(os.getenv("WATCH_COOLDOWN_MINUTES", "20"))
DUPLICATE_ENTRY_ATR_FRACTION = float(os.getenv("DUPLICATE_ENTRY_ATR_FRACTION", "0.20"))

# Notifications.
SEND_WATCH_ALERTS = os.getenv("SEND_WATCH_ALERTS", "true").lower() == "true"
SEND_NO_TRADE_ALERTS = os.getenv("SEND_NO_TRADE_ALERTS", "false").lower() == "true"
USE_OPENAI_REWRITE = os.getenv("USE_OPENAI_REWRITE", "true").lower() == "true"

# Un verrou protège les écritures CSV et la mémoire anti-doublon.
STATE_LOCK = threading.Lock()
LAST_SIGNALS: Dict[str, Dict[str, Any]] = {}

app = Flask(__name__)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(APP_NAME)


# =============================================================================
# MODÈLES DE DONNÉES
# =============================================================================

@dataclass
class MarketData:
    symbol: str = DEFAULT_SYMBOL
    timeframe: str = DEFAULT_TIMEFRAME
    timestamp: str = ""

    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None

    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    volume: Optional[float] = None
    volume_ma: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None

    trend_m5: str = ""
    trend_h1: str = ""
    price_vs_ema200: str = ""
    volume_state: str = ""


@dataclass
class ScoreItem:
    name: str
    buy: float
    sell: float
    max_points: float
    buy_note: str
    sell_note: str


@dataclass
class ScoreResult:
    buy_score: float
    sell_score: float
    direction: str
    score: float
    opposing_score: float
    confidence: int
    details: List[ScoreItem] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list)


# =============================================================================
# OUTILS GÉNÉRAUX
# =============================================================================

def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        if isinstance(value, str):
            value = value.strip().replace(" ", "").replace(",", ".")
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalise_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def fmt_price(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def fmt_score(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def fmt_rr(value: Optional[float]) -> str:
    return "N/A" if value is None else f"1:{value:.2f}"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def is_bullish_text(value: str) -> bool:
    value = normalise_text(value)
    return any(token in value for token in ("bull", "hauss", "up", "long", "above"))


def is_bearish_text(value: str) -> bool:
    value = normalise_text(value)
    return any(token in value for token in ("bear", "baiss", "down", "short", "below"))


def safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


# =============================================================================
# 2. FLASK / 3. PARSING JSON TRADINGVIEW
# =============================================================================

def extract_payload() -> Dict[str, Any]:
    """Accepte le JSON actuel, un formulaire ou un corps JSON encodé en texte."""
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload

    form_payload = request.form.to_dict()
    if form_payload:
        return form_payload

    raw = request.data.decode("utf-8", errors="replace").strip()
    if raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            return {"raw": raw}

    return {}


def parse_market_data(data: Dict[str, Any]) -> MarketData:
    """Conserve les noms de champs du webhook existant."""
    return MarketData(
        symbol=str(data.get("symbol") or DEFAULT_SYMBOL).upper(),
        timeframe=str(data.get("timeframe") or DEFAULT_TIMEFRAME),
        timestamp=str(data.get("timestamp") or data.get("time") or ""),
        open=to_float(data.get("open")),
        high=to_float(data.get("high")),
        low=to_float(data.get("low")),
        close=to_float(data.get("close")),
        ema20=to_float(data.get("ema20")),
        ema50=to_float(data.get("ema50")),
        ema200=to_float(data.get("ema200")),
        rsi=to_float(data.get("rsi")),
        atr=to_float(data.get("atr")),
        volume=to_float(data.get("volume")),
        volume_ma=to_float(data.get("volume_ma")),
        support=to_float(data.get("support")),
        resistance=to_float(data.get("resistance")),
        trend_m5=normalise_text(data.get("trend_m5")),
        trend_h1=normalise_text(data.get("trend_h1")),
        price_vs_ema200=normalise_text(data.get("price_vs_ema200")),
        volume_state=normalise_text(data.get("volume_state")),
    )


def validate_market(market: MarketData) -> List[str]:
    errors: List[str] = []
    if market.close is None or market.close <= 0:
        errors.append("Prix de clôture absent ou invalide")
    if market.atr is None or market.atr <= 0:
        errors.append("ATR absent ou invalide")
    if market.high is not None and market.low is not None and market.high < market.low:
        errors.append("High inférieur au Low")
    return errors


# =============================================================================
# 4. ANALYSE TECHNIQUE
# =============================================================================

def candle_metrics(market: MarketData) -> Dict[str, Optional[float]]:
    body = None
    candle_range = None
    body_ratio = None
    close_location = None

    if market.open is not None and market.close is not None:
        body = market.close - market.open

    if market.high is not None and market.low is not None:
        candle_range = market.high - market.low

    if body is not None and candle_range is not None and candle_range > 0:
        body_ratio = abs(body) / candle_range
        close_location = (market.close - market.low) / candle_range if market.close is not None else None

    return {
        "body": body,
        "range": candle_range,
        "body_ratio": body_ratio,
        "close_location": close_location,
        "range_atr": safe_ratio(candle_range, market.atr),
        "body_atr": safe_ratio(abs(body) if body is not None else None, market.atr),
    }


def structure_room(market: MarketData, direction: str) -> Optional[float]:
    if market.close is None:
        return None
    if direction == "BUY" and market.resistance is not None:
        return market.resistance - market.close
    if direction == "SELL" and market.support is not None:
        return market.close - market.support
    return None


def add_score_item(
    details: List[ScoreItem],
    name: str,
    buy: float,
    sell: float,
    max_points: float,
    buy_note: str,
    sell_note: str,
) -> None:
    details.append(
        ScoreItem(
            name=name,
            buy=round(buy, 2),
            sell=round(sell, 2),
            max_points=max_points,
            buy_note=buy_note,
            sell_note=sell_note,
        )
    )


# =============================================================================
# 5. SCORE MOTEUR
# =============================================================================

def calculate_market_score(market: MarketData) -> ScoreResult:
    """
    Calcule simultanément un score BUY et un score SELL.

    Barème maximal : 15 points.
      - Tendance H1              3.0
      - Alignement EMA20/EMA50  2.0
      - Position vs EMA200      2.0
      - Tendance M5             1.5
      - RSI                     1.5
      - Momentum bougie         1.5
      - Volume                  1.0
      - Volatilité ATR          1.0
      - Structure S/R           1.5
    """
    details: List[ScoreItem] = []
    missing: List[str] = []
    metrics = candle_metrics(market)

    # 1) Tendance H1 — 3 points
    if is_bullish_text(market.trend_h1):
        add_score_item(details, "Tendance H1", 3.0, 0.0, 3.0,
                       "Tendance H1 haussière", "Tendance H1 opposée")
    elif is_bearish_text(market.trend_h1):
        add_score_item(details, "Tendance H1", 0.0, 3.0, 3.0,
                       "Tendance H1 opposée", "Tendance H1 baissière")
    else:
        add_score_item(details, "Tendance H1", 1.0, 1.0, 3.0,
                       "Tendance H1 neutre", "Tendance H1 neutre")
        missing.append("Tendance H1 non explicite")

    # 2) EMA20 / EMA50 — 2 points
    if market.ema20 is not None and market.ema50 is not None:
        spread = abs(market.ema20 - market.ema50)
        spread_atr = safe_ratio(spread, market.atr) or 0.0
        strength = 2.0 if spread_atr >= 0.15 else 1.5
        if market.ema20 > market.ema50:
            add_score_item(details, "EMA20 / EMA50", strength, 0.0, 2.0,
                           "EMA20 au-dessus de l'EMA50", "EMA opposées à la vente")
        elif market.ema20 < market.ema50:
            add_score_item(details, "EMA20 / EMA50", 0.0, strength, 2.0,
                           "EMA opposées à l'achat", "EMA20 sous l'EMA50")
        else:
            add_score_item(details, "EMA20 / EMA50", 0.5, 0.5, 2.0,
                           "EMA sans direction", "EMA sans direction")
    else:
        add_score_item(details, "EMA20 / EMA50", 0.0, 0.0, 2.0,
                       "Données EMA manquantes", "Données EMA manquantes")
        missing.append("EMA20 ou EMA50 manquante")

    # 3) Position par rapport à l'EMA200 — 2 points
    if market.close is not None and market.ema200 is not None:
        distance_atr = safe_ratio(abs(market.close - market.ema200), market.atr) or 0.0
        strength = 2.0 if distance_atr >= 0.20 else 1.5
        if market.close > market.ema200:
            add_score_item(details, "Prix / EMA200", strength, 0.0, 2.0,
                           "Prix au-dessus de l'EMA200", "Prix au-dessus de l'EMA200")
        elif market.close < market.ema200:
            add_score_item(details, "Prix / EMA200", 0.0, strength, 2.0,
                           "Prix sous l'EMA200", "Prix sous l'EMA200")
        else:
            add_score_item(details, "Prix / EMA200", 0.5, 0.5, 2.0,
                           "Prix sur l'EMA200", "Prix sur l'EMA200")
    elif is_bullish_text(market.price_vs_ema200):
        add_score_item(details, "Prix / EMA200", 1.5, 0.0, 2.0,
                       "Prix indiqué au-dessus de l'EMA200", "Contexte EMA200 opposé")
    elif is_bearish_text(market.price_vs_ema200):
        add_score_item(details, "Prix / EMA200", 0.0, 1.5, 2.0,
                       "Contexte EMA200 opposé", "Prix indiqué sous l'EMA200")
    else:
        add_score_item(details, "Prix / EMA200", 0.0, 0.0, 2.0,
                       "EMA200 manquante", "EMA200 manquante")
        missing.append("EMA200 manquante")

    # 4) Tendance M5 — 1.5 point
    if is_bullish_text(market.trend_m5):
        add_score_item(details, "Tendance M5", 1.5, 0.0, 1.5,
                       "Tendance M5 haussière", "Tendance M5 opposée")
    elif is_bearish_text(market.trend_m5):
        add_score_item(details, "Tendance M5", 0.0, 1.5, 1.5,
                       "Tendance M5 opposée", "Tendance M5 baissière")
    else:
        add_score_item(details, "Tendance M5", 0.5, 0.5, 1.5,
                       "Tendance M5 neutre", "Tendance M5 neutre")

    # 5) RSI — 1.5 point, sans bloquer les zones extrêmes.
    rsi = market.rsi
    if rsi is None:
        add_score_item(details, "RSI", 0.0, 0.0, 1.5, "RSI manquant", "RSI manquant")
        missing.append("RSI manquant")
    elif 52 <= rsi <= 68:
        add_score_item(details, "RSI", 1.5, 0.0, 1.5,
                       f"RSI acheteur ({rsi:.1f})", f"RSI défavorable ({rsi:.1f})")
    elif 32 <= rsi <= 48:
        add_score_item(details, "RSI", 0.0, 1.5, 1.5,
                       f"RSI défavorable ({rsi:.1f})", f"RSI vendeur ({rsi:.1f})")
    elif 48 < rsi < 52:
        add_score_item(details, "RSI", 0.5, 0.5, 1.5,
                       f"RSI neutre ({rsi:.1f})", f"RSI neutre ({rsi:.1f})")
    elif 68 < rsi <= 75:
        add_score_item(details, "RSI", 0.75, 0.0, 1.5,
                       f"Momentum fort mais RSI élevé ({rsi:.1f})", f"RSI élevé ({rsi:.1f})")
    elif 25 <= rsi < 32:
        add_score_item(details, "RSI", 0.0, 0.75, 1.5,
                       f"RSI faible ({rsi:.1f})", f"Momentum vendeur mais RSI bas ({rsi:.1f})")
    elif rsi > 75:
        add_score_item(details, "RSI", 0.25, 0.25, 1.5,
                       f"Surachat extrême ({rsi:.1f})", f"Possible retournement après surachat ({rsi:.1f})")
    else:
        add_score_item(details, "RSI", 0.25, 0.25, 1.5,
                       f"Survente extrême ({rsi:.1f})", f"Survente extrême ({rsi:.1f})")

    # 6) Momentum / bougie — 1.5 point
    body = metrics["body"]
    body_ratio = metrics["body_ratio"]
    close_location = metrics["close_location"]
    range_atr = metrics["range_atr"]
    if body is None:
        add_score_item(details, "Momentum", 0.0, 0.0, 1.5,
                       "OHLC insuffisant", "OHLC insuffisant")
        missing.append("Open absent pour mesurer le momentum")
    else:
        buy_momentum = 0.0
        sell_momentum = 0.0
        if body > 0:
            buy_momentum += 0.75
            if body_ratio is not None and body_ratio >= 0.55:
                buy_momentum += 0.40
            if close_location is not None and close_location >= 0.70:
                buy_momentum += 0.35
        elif body < 0:
            sell_momentum += 0.75
            if body_ratio is not None and body_ratio >= 0.55:
                sell_momentum += 0.40
            if close_location is not None and close_location <= 0.30:
                sell_momentum += 0.35

        # Une bougie très petite n'est pas une vraie impulsion.
        if range_atr is not None and range_atr < 0.35:
            buy_momentum *= 0.65
            sell_momentum *= 0.65

        add_score_item(
            details,
            "Momentum",
            min(buy_momentum, 1.5),
            min(sell_momentum, 1.5),
            1.5,
            "Bougie acheteuse avec clôture ferme" if buy_momentum >= 1 else "Momentum acheteur limité",
            "Bougie vendeuse avec clôture ferme" if sell_momentum >= 1 else "Momentum vendeur limité",
        )

    # 7) Volume — 1 point
    volume_ratio = safe_ratio(market.volume, market.volume_ma)
    if volume_ratio is not None:
        points = 1.0 if volume_ratio >= 1.15 else 0.6 if volume_ratio >= 1.0 else 0.0
        add_score_item(details, "Volume", points, points, 1.0,
                       f"Volume x{volume_ratio:.2f} de la moyenne",
                       f"Volume x{volume_ratio:.2f} de la moyenne")
    elif any(token in market.volume_state for token in ("high", "fort", "above", "sup")):
        add_score_item(details, "Volume", 0.8, 0.8, 1.0,
                       "Volume signalé supérieur", "Volume signalé supérieur")
    else:
        add_score_item(details, "Volume", 0.0, 0.0, 1.0,
                       "Volume non confirmé", "Volume non confirmé")
        missing.append("Volume ou moyenne de volume manquante")

    # 8) ATR / volatilité — 1 point
    if market.atr is not None and market.atr > 0 and market.close is not None:
        atr_pct = market.atr / market.close * 100
        if 0.04 <= atr_pct <= 0.45:
            atr_points = 1.0
            atr_note = f"Volatilité exploitable (ATR {atr_pct:.3f}%)"
        elif atr_pct < 0.04:
            atr_points = 0.25
            atr_note = f"Volatilité faible (ATR {atr_pct:.3f}%)"
        else:
            atr_points = 0.50
            atr_note = f"Volatilité élevée (ATR {atr_pct:.3f}%)"
        add_score_item(details, "ATR", atr_points, atr_points, 1.0, atr_note, atr_note)
    else:
        add_score_item(details, "ATR", 0.0, 0.0, 1.0,
                       "ATR invalide", "ATR invalide")

    # 9) Structure / espace avant S-R — 1.5 point
    buy_room = structure_room(market, "BUY")
    sell_room = structure_room(market, "SELL")
    buy_room_atr = safe_ratio(buy_room, market.atr)
    sell_room_atr = safe_ratio(sell_room, market.atr)

    def room_points(room: Optional[float], room_atr: Optional[float]) -> Tuple[float, str]:
        if room is None or room_atr is None:
            return 0.75, "Niveau opposé non fourni"
        if room <= 0:
            return 0.25, "Niveau opposé déjà franchi ou incohérent"
        if room_atr >= 2.5:
            return 1.5, f"Espace confortable ({room_atr:.1f} ATR)"
        if room_atr >= 1.5:
            return 1.0, f"Espace correct ({room_atr:.1f} ATR)"
        if room_atr >= 0.8:
            return 0.5, f"Niveau opposé assez proche ({room_atr:.1f} ATR)"
        return 0.0, f"Niveau opposé trop proche ({room_atr:.1f} ATR)"

    buy_structure, buy_note = room_points(buy_room, buy_room_atr)
    sell_structure, sell_note = room_points(sell_room, sell_room_atr)
    add_score_item(details, "Structure S/R", buy_structure, sell_structure, 1.5,
                   buy_note, sell_note)

    buy_score = round(sum(item.buy for item in details), 2)
    sell_score = round(sum(item.sell for item in details), 2)

    if buy_score >= sell_score:
        direction = "BUY"
        score = buy_score
        opposing_score = sell_score
    else:
        direction = "SELL"
        score = sell_score
        opposing_score = buy_score

    # La confiance tient compte du score absolu et de l'écart entre les deux scénarios.
    score_quality = score / MAX_MARKET_SCORE
    directional_edge = clamp((score - opposing_score) / 6.0, 0.0, 1.0)
    confidence = int(round(clamp((score_quality * 75) + (directional_edge * 25), 0, 99)))

    strengths: List[str] = []
    warnings: List[str] = []
    for item in details:
        earned = item.buy if direction == "BUY" else item.sell
        note = item.buy_note if direction == "BUY" else item.sell_note
        if earned >= item.max_points * 0.65:
            strengths.append(note)
        elif earned <= item.max_points * 0.25:
            warnings.append(note)

    if score - opposing_score < 1.5:
        warnings.append("Avantage directionnel faible entre BUY et SELL")

    return ScoreResult(
        buy_score=buy_score,
        sell_score=sell_score,
        direction=direction,
        score=score,
        opposing_score=opposing_score,
        confidence=confidence,
        details=details,
        strengths=strengths,
        warnings=warnings,
        missing_data=missing,
    )


# =============================================================================
# 6. CALCUL ENTRÉE / SL / TP / RISK REWARD
# =============================================================================

def calculate_trade_levels(market: MarketData, direction: str) -> Dict[str, Any]:
    entry = market.close
    atr = market.atr
    if entry is None or atr is None or atr <= 0:
        return {"valid": False, "reason": "Prix ou ATR invalide"}

    if direction == "BUY":
        atr_stop = entry - ATR_STOP_MULTIPLIER * atr
        structure_candidates = [value for value in (market.support, market.low) if value is not None and value < entry]
        structure_stop = min(structure_candidates) - STRUCTURE_ATR_BUFFER * atr if structure_candidates else atr_stop
        sl = min(atr_stop, structure_stop)
        risk = entry - sl
        tp1 = entry + MIN_RR * risk
        tp2 = entry + TP2_RR * risk
        tp3 = entry + TP3_RR * risk
        available_room = market.resistance - entry if market.resistance is not None else None
    else:
        atr_stop = entry + ATR_STOP_MULTIPLIER * atr
        structure_candidates = [value for value in (market.resistance, market.high) if value is not None and value > entry]
        structure_stop = max(structure_candidates) + STRUCTURE_ATR_BUFFER * atr if structure_candidates else atr_stop
        sl = max(atr_stop, structure_stop)
        risk = sl - entry
        tp1 = entry - MIN_RR * risk
        tp2 = entry - TP2_RR * risk
        tp3 = entry - TP3_RR * risk
        available_room = entry - market.support if market.support is not None else None

    if risk <= 0 or not math.isfinite(risk):
        return {"valid": False, "reason": "Stop Loss incohérent"}

    stop_atr = risk / atr
    rr = abs(tp1 - entry) / risk
    tp1_distance_atr = abs(tp1 - entry) / atr

    warnings: List[str] = []
    blockers: List[str] = []

    if rr < MIN_RR:
        blockers.append("Risk/Reward inférieur à 1:2")
    if stop_atr > MAX_STOP_ATR:
        blockers.append(f"Stop trop large ({stop_atr:.1f} ATR)")
    if tp1_distance_atr < MIN_TARGET_ATR:
        blockers.append(f"TP1 trop proche ({tp1_distance_atr:.1f} ATR)")

    room_rr = None
    if available_room is not None:
        room_rr = available_room / risk
        if available_room <= 0:
            blockers.append("Niveau opposé déjà atteint ou dépassé")
        elif room_rr < 1.0:
            blockers.append("Obstacle majeur avant 1R")
        elif room_rr < MIN_RR:
            warnings.append(f"Obstacle structurel avant TP1 ({room_rr:.2f}R)")

    return {
        "valid": not blockers,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "rr": rr,
        "stop_atr": stop_atr,
        "tp1_distance_atr": tp1_distance_atr,
        "room_rr": room_rr,
        "warnings": warnings,
        "blockers": blockers,
        "reason": " | ".join(blockers) if blockers else "Plan de risque valide",
    }


def classify_decision(score_result: ScoreResult, levels: Dict[str, Any]) -> Tuple[str, str]:
    direction = score_result.direction
    score = score_result.score

    if not levels.get("valid"):
        # Un plan invalide ne peut jamais devenir un signal réel.
        if score >= WATCH_SCORE:
            return f"WATCH {direction}", "Setup intéressant mais plan de risque non validé"
        return "NO TRADE", "Plan de risque non validé"

    if score >= PREMIUM_SCORE:
        return direction, "Signal Premium"
    if score >= VALID_SCORE:
        return direction, "Signal valide"
    if score >= WATCH_SCORE:
        return f"WATCH {direction}", "Setup à surveiller"
    return "NO TRADE", "Score insuffisant"


def build_trade_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    market = parse_market_data(data)
    validation_errors = validate_market(market)
    if validation_errors:
        return {
            "status": "NO_TRADE",
            "action": "NO TRADE",
            "label": "Données invalides",
            "score": 0.0,
            "confidence": 0,
            "reason": " | ".join(validation_errors),
            "market": asdict(market),
            "score_details": [],
            "strengths": [],
            "warnings": validation_errors,
        }

    score_result = calculate_market_score(market)
    levels = calculate_trade_levels(market, score_result.direction)
    action, label = classify_decision(score_result, levels)

    warnings = list(dict.fromkeys(
        score_result.warnings
        + levels.get("warnings", [])
        + levels.get("blockers", [])
        + score_result.missing_data
    ))

    if action in ("BUY", "SELL"):
        status = "TRADE"
    elif action.startswith("WATCH"):
        status = "WATCH"
    else:
        status = "NO_TRADE"

    reason_parts = score_result.strengths[:3]
    if warnings:
        reason_parts.append("Attention : " + "; ".join(warnings[:2]))

    result: Dict[str, Any] = {
        "status": status,
        "action": action,
        "direction": score_result.direction,
        "label": label,
        "score": score_result.score,
        "score_max": MAX_MARKET_SCORE,
        "buy_score": score_result.buy_score,
        "sell_score": score_result.sell_score,
        "confidence": score_result.confidence,
        "reason": " | ".join(reason_parts) or label,
        "market": asdict(market),
        "score_details": [asdict(item) for item in score_result.details],
        "strengths": score_result.strengths,
        "warnings": warnings,
        "missing_data": score_result.missing_data,
    }
    result.update({key: value for key, value in levels.items() if key != "valid"})
    return result


# =============================================================================
# 7. GÉNÉRATION DES MESSAGES TELEGRAM
# =============================================================================

def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram non configuré : message non envoyé")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )
    response.raise_for_status()


def generate_short_reason(data: Dict[str, Any], decision: Dict[str, Any]) -> str:
    fallback = decision.get("reason", "Analyse technique calculée par le moteur V7.")
    if not USE_OPENAI_REWRITE or openai_client is None:
        return fallback

    immutable_plan = {
        "action": decision.get("action"),
        "score": decision.get("score"),
        "confidence": decision.get("confidence"),
        "entry": decision.get("entry"),
        "sl": decision.get("sl"),
        "tp1": decision.get("tp1"),
        "tp2": decision.get("tp2"),
        "tp3": decision.get("tp3"),
        "rr": decision.get("rr"),
        "strengths": decision.get("strengths", [])[:5],
        "warnings": decision.get("warnings", [])[:4],
    }

    prompt = f"""
Tu reformules une analyse technique XAUUSD déjà calculée par un moteur Python.
Tu n'effectues aucun calcul et tu ne modifies aucune valeur.

Plan immuable :
{json.dumps(immutable_plan, ensure_ascii=False, default=str)}

Écris en français 2 phrases maximum, professionnelles et factuelles.
Mentionne les confirmations principales puis le risque principal.
Aucune promesse de gain, aucun nouvel objectif, aucun nouveau prix.
""".strip()

    try:
        response = openai_client.responses.create(model=OPENAI_MODEL, input=prompt)
        rewritten = response.output_text.strip()
        return rewritten or fallback
    except Exception as error:  # Le signal ne doit jamais dépendre d'OpenAI.
        logger.exception("Erreur OpenAI : %s", error)
        return fallback


def score_lines(decision: Dict[str, Any], limit: int = 6) -> List[str]:
    direction = decision.get("direction", "BUY")
    key = "buy" if direction == "BUY" else "sell"
    note_key = "buy_note" if direction == "BUY" else "sell_note"
    rows = sorted(
        decision.get("score_details", []),
        key=lambda item: float(item.get(key, 0)),
        reverse=True,
    )

    lines: List[str] = []
    for item in rows[:limit]:
        points = float(item.get(key, 0))
        maximum = float(item.get("max_points", 0))
        icon = "✅" if points >= maximum * 0.65 else "⚠️" if points > 0 else "❌"
        lines.append(f"{icon} {item.get('name')} : +{fmt_score(points)} — {item.get(note_key, '')}")
    return lines


def format_trade_message(decision: Dict[str, Any], reason: str) -> str:
    market = decision["market"]
    action = decision["action"]
    premium = decision.get("score", 0) >= PREMIUM_SCORE
    side_icon = "🟢" if action == "BUY" else "🔴"
    quality_icon = "🔥" if premium else "✅"

    analysis = "\n".join(score_lines(decision, limit=5))
    warnings = decision.get("warnings", [])
    warning_line = f"\n⚠️ {warnings[0]}" if warnings else ""

    return f"""━━━━━━━━━━━━━━━━━━
🟦 GOLD AGENT V7 PRO
{side_icon} {action} {market.get('symbol', DEFAULT_SYMBOL)}
{quality_icon} {decision.get('label')}
📊 Score : {fmt_score(decision.get('score', 0))}/{fmt_score(decision.get('score_max', MAX_MARKET_SCORE))}
❤️ Fiabilité : {decision.get('confidence', 0)}%

💰 Entrée : {fmt_price(decision.get('entry'))}
🛡 Stop Loss : {fmt_price(decision.get('sl'))}
🎯 TP1 : {fmt_price(decision.get('tp1'))}
🎯 TP2 : {fmt_price(decision.get('tp2'))}
🚀 TP3 : {fmt_price(decision.get('tp3'))}
⚖️ Risk Reward : {fmt_rr(decision.get('rr'))}

📊 ANALYSE
{analysis}

🧠 SYNTHÈSE
{reason}{warning_line}
━━━━━━━━━━━━━━━━━━
⚠️ Aucune performance n'est garantie. Gérez votre risque."""


def format_watch_message(decision: Dict[str, Any], reason: str) -> str:
    market = decision["market"]
    direction = decision.get("direction", "BUY")
    icon = "🟢" if direction == "BUY" else "🔴"
    analysis = "\n".join(score_lines(decision, limit=4))
    blockers = decision.get("blockers") or decision.get("warnings", [])
    blocker_text = "\n".join(f"• {item}" for item in blockers[:3]) or "• Confirmation supplémentaire requise"

    return f"""━━━━━━━━━━━━━━━━━━
🟦 GOLD AGENT V7 PRO
👀 WATCH {icon} {direction} {market.get('symbol', DEFAULT_SYMBOL)}
📊 Score : {fmt_score(decision.get('score', 0))}/{fmt_score(decision.get('score_max', MAX_MARKET_SCORE))}
❤️ Fiabilité : {decision.get('confidence', 0)}%

📍 Prix actuel : {fmt_price(decision.get('entry'))}
⚖️ RR théorique : {fmt_rr(decision.get('rr'))}

📊 POINTS POSITIFS
{analysis}

⏳ À CONFIRMER
{blocker_text}

🧠 {reason}
━━━━━━━━━━━━━━━━━━
Ce message est une surveillance, pas une entrée en position."""


def format_no_trade_message(decision: Dict[str, Any]) -> str:
    market = decision.get("market", {})
    warnings = decision.get("warnings", [])
    reasons = "\n".join(f"• {item}" for item in warnings[:4]) or f"• {decision.get('reason', 'Score insuffisant')}"
    return f"""━━━━━━━━━━━━━━━━━━
🟦 GOLD AGENT V7 PRO
⏳ NO TRADE {market.get('symbol', DEFAULT_SYMBOL)}
📊 Score : {fmt_score(decision.get('score', 0))}/{fmt_score(decision.get('score_max', MAX_MARKET_SCORE))}

Raisons :
{reasons}
━━━━━━━━━━━━━━━━━━"""


# =============================================================================
# 8. LOGS ET ANTI-DOUBLON
# =============================================================================

def log_decision_console(decision: Dict[str, Any]) -> None:
    logger.info("=" * 62)
    logger.info("DECISION %s", decision.get("action"))
    logger.info(
        "Score : %s/%s | BUY=%s | SELL=%s | Fiabilité=%s%%",
        fmt_score(decision.get("score", 0)),
        fmt_score(decision.get("score_max", MAX_MARKET_SCORE)),
        fmt_score(decision.get("buy_score", 0)),
        fmt_score(decision.get("sell_score", 0)),
        decision.get("confidence", 0),
    )

    direction = decision.get("direction", "BUY")
    score_key = "buy" if direction == "BUY" else "sell"
    for item in decision.get("score_details", []):
        logger.info(
            "%-18s %+4.1f/%s | %s",
            item.get("name", ""),
            float(item.get(score_key, 0)),
            fmt_score(float(item.get("max_points", 0))),
            item.get("buy_note" if direction == "BUY" else "sell_note", ""),
        )

    if decision.get("status") in ("TRADE", "WATCH"):
        logger.info(
            "Entrée=%s | SL=%s | TP1=%s | TP2=%s | TP3=%s | RR=%s",
            fmt_price(decision.get("entry")),
            fmt_price(decision.get("sl")),
            fmt_price(decision.get("tp1")),
            fmt_price(decision.get("tp2")),
            fmt_price(decision.get("tp3")),
            fmt_rr(decision.get("rr")),
        )

    if decision.get("strengths"):
        logger.info("Conditions validées : %s", " | ".join(decision["strengths"][:6]))
    if decision.get("warnings"):
        logger.info("Raisons / alertes : %s", " | ".join(decision["warnings"][:6]))
    logger.info("=" * 62)


def log_decision_csv(raw_data: Dict[str, Any], decision: Dict[str, Any], message: str = "") -> None:
    columns = [
        "time_utc", "symbol", "timeframe", "status", "decision", "direction",
        "score", "score_max", "buy_score", "sell_score", "confidence",
        "entry", "stop_loss", "tp1", "tp2", "tp3", "rr", "close",
        "strengths", "warnings", "message",
    ]
    market = decision.get("market", {})
    row = [
        datetime.now(timezone.utc).isoformat(),
        market.get("symbol", raw_data.get("symbol", "")),
        market.get("timeframe", raw_data.get("timeframe", "")),
        decision.get("status", ""),
        decision.get("action", ""),
        decision.get("direction", ""),
        decision.get("score", ""),
        decision.get("score_max", ""),
        decision.get("buy_score", ""),
        decision.get("sell_score", ""),
        decision.get("confidence", ""),
        decision.get("entry", ""),
        decision.get("sl", ""),
        decision.get("tp1", ""),
        decision.get("tp2", ""),
        decision.get("tp3", ""),
        decision.get("rr", ""),
        market.get("close", raw_data.get("close", "")),
        " | ".join(decision.get("strengths", [])),
        " | ".join(decision.get("warnings", [])),
        message.replace("\n", " | "),
    ]

    with STATE_LOCK:
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(columns)
            writer.writerow(row)


def signal_key(decision: Dict[str, Any]) -> str:
    market = decision.get("market", {})
    return f"{market.get('symbol', DEFAULT_SYMBOL)}:{decision.get('action', '')}"


def is_duplicate_signal(decision: Dict[str, Any]) -> bool:
    if decision.get("status") not in ("TRADE", "WATCH"):
        return False

    key = signal_key(decision)
    now = datetime.now(timezone.utc)
    cooldown = WATCH_COOLDOWN_MINUTES if decision.get("status") == "WATCH" else TRADE_COOLDOWN_MINUTES
    entry = to_float(decision.get("entry"))
    atr = to_float(decision.get("market", {}).get("atr"))

    with STATE_LOCK:
        previous = LAST_SIGNALS.get(key)
        if not previous:
            return False

        elapsed = now - previous["time"]
        previous_entry = previous.get("entry")
        tolerance = max((atr or 1.0) * DUPLICATE_ENTRY_ATR_FRACTION, 0.10)
        same_zone = (
            entry is not None
            and previous_entry is not None
            and abs(entry - previous_entry) <= tolerance
        )
        return elapsed < timedelta(minutes=cooldown) and same_zone


def remember_signal(decision: Dict[str, Any]) -> None:
    with STATE_LOCK:
        LAST_SIGNALS[signal_key(decision)] = {
            "time": datetime.now(timezone.utc),
            "entry": to_float(decision.get("entry")),
        }


# =============================================================================
# 9. ROUTES ET LANCEMENT SERVEUR
# =============================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "agent": APP_NAME,
        "symbol": DEFAULT_SYMBOL,
        "version": "7.0",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "time_utc": datetime.now(timezone.utc).isoformat()})


@app.route("/webhook", methods=["POST"])
def webhook():
    data = extract_payload()
    logger.info("TRADINGVIEW DATA: %s", json.dumps(data, ensure_ascii=False, default=str))

    if not data:
        return jsonify({"status": "error", "message": "Payload vide"}), 400

    try:
        decision = build_trade_plan(data)
        log_decision_console(decision)

        if is_duplicate_signal(decision):
            logger.info("SIGNAL DUPLICATE IGNORÉ : %s", decision.get("action"))
            log_decision_csv(data, decision, "DUPLICATE_IGNORE")
            return jsonify({
                "status": "ok",
                "decision": "DUPLICATE_IGNORE",
                "original_decision": decision.get("action"),
                "score": decision.get("score"),
            })

        message = ""
        if decision.get("status") == "TRADE":
            reason = generate_short_reason(data, decision)
            message = format_trade_message(decision, reason)
            send_telegram(message)
            remember_signal(decision)

        elif decision.get("status") == "WATCH" and SEND_WATCH_ALERTS:
            reason = generate_short_reason(data, decision)
            message = format_watch_message(decision, reason)
            send_telegram(message)
            remember_signal(decision)

        elif decision.get("status") == "NO_TRADE" and SEND_NO_TRADE_ALERTS:
            message = format_no_trade_message(decision)
            send_telegram(message)

        # Toutes les décisions sont enregistrées, même les NO TRADE.
        log_decision_csv(data, decision, message)

        return jsonify({
            "status": "ok",
            "decision": decision.get("action"),
            "label": decision.get("label"),
            "score": decision.get("score"),
            "score_max": decision.get("score_max"),
            "buy_score": decision.get("buy_score"),
            "sell_score": decision.get("sell_score"),
            "confidence": decision.get("confidence"),
            "rr": decision.get("rr"),
        })

    except requests.RequestException as error:
        logger.exception("Erreur Telegram/réseau : %s", error)
        return jsonify({"status": "error", "message": "Erreur d'envoi Telegram"}), 502
    except Exception as error:
        logger.exception("Erreur inattendue dans le webhook : %s", error)
        return jsonify({"status": "error", "message": "Erreur interne"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
