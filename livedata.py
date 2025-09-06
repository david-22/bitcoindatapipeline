import os
import time
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestClassifier

# --- Telegram Config ---
TELEGRAM_TOKEN = ''
TELEGRAM_CHAT_ID = ''

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, data=payload)

# --- Configuración de activos y sus equivalentes en Binance ---
symbols_config = {
    'DOGE_USD': 'DOGEUSDT',
    'XLM_USD': 'XLMUSDT',
    'VET_USD': 'VETUSDT',
    'MKR_USD': 'MKRUSDT',
    'MANA_USD': 'MANAUSDT',
    'IMX_USD': 'IMXUSDT',
    'GRT_USD': 'GRTUSDT',
    'AAVE_USD': 'AAVEUSDT',
    'ICP_USD': 'ICPUSDT',
    'ALGO_USD': 'ALGOUSDT',
    'LINK_USD': 'LINKUSDT',
    'SAND_USD': 'SANDUSDT',
    'BNB_USD': 'BNBUSDT',
    'ETC_USD': 'ETCUSDT',
    'BCH_USD': 'BCHUSDT',
    'SOL_USD': 'SOLUSDT',
    'NEO_USD': 'NEOUSDT',
    'DOT_USD': 'DOTUSDT',
    'DOGE_USD': 'DOGEUSDT',
    'ADA_USD': 'ADAUSDT',
    'XMR_USD': 'XMRUSDT',
    'DASH_USD': 'DASHUSDT',
    'XRP_USD': 'XRPUSDT',
    'LTC_USD': 'LTCUSDT',
    'ETH_USD': 'ETHUSDT',
    'BTC_USD': 'BTCUSDT'
}

def get_latest_ohlcv(binance_symbol, interval='5m', limit=500):
    url = f'https://api.binance.com/api/v3/klines?symbol={binance_symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trade_count',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    return df

def add_indicators(df):
    df['return'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * 100

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=100).mean()
    avg_loss = loss.rolling(window=100).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_crossover'] = 0
    df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'macd_crossover'] = 1
    df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'macd_crossover'] = -1

    df['bb_middle'] = df['close'].rolling(window=100).mean()
    df['bb_std'] = df['close'].rolling(window=100).std()
    df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])

    df['momentum'] = df['close'] - df['close'].shift(10)

    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    df.drop(columns=['tr1', 'tr2', 'tr3', 'true_range'], inplace=True)

    return df.dropna()

def esperar_cierre_vela_5m():
    zona = pytz.timezone("America/Mexico_City")
    ahora = datetime.now(zona)
    segundos_faltantes = (5 - (ahora.minute % 5)) * 60 - ahora.second
    if segundos_faltantes <= 0:
        segundos_faltantes += 5 * 60
    print(f"⏳ Esperando {segundos_faltantes} segundos...")
    time.sleep(segundos_faltantes)

def cargar_modelo(symbol_folder):
    base_path = f"C:/Users/Administrator/Downloads/Modelos/{symbol_folder}/"
    model = joblib.load(base_path + 'modelo.pkl')
    scaler = joblib.load(base_path + 'scaler.pkl')
    features = joblib.load(base_path + 'features.pkl')
    return model, scaler, features

# === LOOP PRINCIPAL ===
while True:
    esperar_cierre_vela_5m()

    for symbol_key, binance_symbol in symbols_config.items():
        try:
            model, scaler, features_columns = cargar_modelo(symbol_key)

            df = get_latest_ohlcv(binance_symbol)
            df = add_indicators(df)
            latest = df.iloc[-1:][features_columns]
            latest_scaled = scaler.transform(latest)

            prob = model.predict_proba(latest_scaled)[0][1]
            pred = model.predict(latest_scaled)[0]
            now = datetime.now(pytz.timezone("America/Mexico_City")).strftime("%Y-%m-%d %H:%M:%S")

            print(f"[{symbol_key}] {now} → Predicción: {'⬆️' if pred == 1 else '⬇️'} (Prob: {prob:.2%})")

            if prob >= 0.80:
                mensaje = f"🚨 <b>{symbol_key}</b>: Probabilidad ALTA de subida ⬆️ ({prob:.2%}) [{now}]"
                send_telegram_message(mensaje)
            elif prob <=0.15:
                mensaje = f"🚨 <b>{symbol_key}</b>: Probabilidad ALTA de bajada ⬇️ ({prob:.2%}) [{now}]"
                send_telegram_message(mensaje)

        except Exception as e:
            print(f"[{symbol_key}] ❌ ERROR: {e}")
