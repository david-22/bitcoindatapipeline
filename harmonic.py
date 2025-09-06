import time
import pandas as pd
import numpy as np
import requests
import pytz
import matplotlib.pyplot as plt
import io
import time
from datetime import datetime

# --- Configuración Telegram ---
TELEGRAM_TOKEN = '7732608702:AAH2xD9H-LY50j2TGajf2mmFPqPTatAPwrc'
TELEGRAM_CHAT_ID = '777734102'

def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        resp = requests.post(url, data=payload)
        if resp.status_code != 200:
            print(f"Error enviando mensaje Telegram: {resp.text}")
    except Exception as e:
        print(f"Excepción enviando Telegram: {e}")

def send_telegram_photo(photo_bytes, caption=""):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto'
    files = {'photo': ('pattern.png', photo_bytes)}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'HTML'}
    try:
        resp = requests.post(url, files=files, data=data)
        if resp.status_code != 200:
            print(f"Error enviando foto Telegram: {resp.text}")
    except Exception as e:
        print(f"Excepción enviando foto Telegram: {e}")

# --- Funciones para datos Binance y detección ---

def get_klines(symbol, interval='5m', limit=500):
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    resp = requests.get(url, params=params)
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df = df[['open_time', 'open', 'high', 'low', 'close']].copy()
    df['datetime_utc'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)

def detect_pivots(df, n=3):
    df['min'] = df['low'].rolling(window=n*2+1, center=True).apply(
        lambda x: x.values[n] if x.values[n] == x.min() else np.nan, raw=False)
    df['max'] = df['high'].rolling(window=n*2+1, center=True).apply(
        lambda x: x.values[n] if x.values[n] == x.max() else np.nan, raw=False)
    pivots = []
    for i, row in df.iterrows():
        if not np.isnan(row['min']):
            pivots.append((i, row['min']))
        elif not np.isnan(row['max']):
            pivots.append((i, row['max']))
    return pivots

def is_gartley(x, a, b, c, d, tol=0.2):
    def pct(p1, p2): return abs((p2 - p1) / (a - x)) if a != x else 0
    ab = pct(a, b)
    bc = abs((c - b) / (b - a)) if b != a else 0
    cd = abs((d - c) / (c - b)) if c != b else 0
    ad = pct(x, d)
    return (
        0.618 - tol <= ab <= 0.618 + tol and
        0.382 - tol <= bc <= 0.886 + tol and
        1.272 - tol <= cd <= 1.618 + tol and
        0.786 - tol <= ad <= 0.786 + tol
    )

def find_gartley_patterns(pivots):
    found = []
    for i in range(len(pivots) - 4):
        x_i, x = pivots[i]
        a_i, a = pivots[i+1]
        b_i, b = pivots[i+2]
        c_i, c = pivots[i+3]
        d_i, d = pivots[i+4]
        if is_gartley(x, a, b, c, d):
            found.append((x_i, a_i, b_i, c_i, d_i))
    return found

def esperar_cierre_vela_1m():
    zona = pytz.timezone("America/Mexico_City")  # Ajusta si cambias de zona
    ahora = datetime.now(zona)
    
    minutos_pasados = ahora.minute % 90
    segundos_faltantes = (90 - minutos_pasados) * 60 - ahora.second

    if segundos_faltantes <= 0:
        segundos_faltantes += 90 * 60  # para el siguiente intervalo

    print(f"⏳ Esperando {segundos_faltantes} segundos hasta la siguiente vela de 1 minutos...")
    time.sleep(segundos_faltantes)

# --- Lista de símbolos a analizar ---
symbols = ['PAXGUSDT','BTCUSDT', 'DASHUSDT', 'ETHUSDT', 'LTCUSDT', 'XRPUSDT', 'XMRUSDT', 'NEOUSDT', 'ADAUSDT', 'DOTUSDT', 'DOGEUSDT', 'SOLUSDT', 'AVAUSDT', 'BCHUSDT', 'ETCUSDT', 'BNBUSDT', 'SANDUSDT', 'LINKUSDT', 'NEARUSDT', 'ALGOUSDT', 'ICPUSDT', 'AAVEUSDT', 'BARUSDT', 'GALUSDT', 'GRTUSDT', 'IMXUSDT', 'MANAUSDT', 'MKRUSDT', 'VETUSDT', 'XLMUSDT', 'UNIUSDT', 'FETUSDT', 'XTZUSDT']
# --- Para evitar enviar alertas repetidas ---
alerted_patterns = set()
import pytz
local_tz = pytz.timezone("America/Mexico_City")

while True:
    esperar_cierre_vela_1m()
    for sym in symbols:
        print(f"\nAnalizando {sym} ...")
        df = get_klines(sym, interval='5m', limit=500)
        pivots = detect_pivots(df, n=3)
        patterns = find_gartley_patterns(pivots)

        print(f"Pivotes detectados: {len(pivots)}")
        print(f"Patrones Gartley encontrados: {len(patterns)}")

        for pat in patterns:
            pattern_id = (sym,) + pat
            if pattern_id not in alerted_patterns:
                alerted_patterns.add(pattern_id)
                x_i, a_i, b_i, c_i, d_i = pat

                # Convertir timestamps UTC a hora local
                start_dt_local = df.loc[x_i, 'datetime_utc'].tz_convert(local_tz)
                end_dt_local = df.loc[d_i, 'datetime_utc'].tz_convert(local_tz)

                caption = (f"🔥 Patrón Gartley detectado en <b>{sym}</b>\n"
                           f"📅 Desde: {start_dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                           f"📅 Hasta: {end_dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                           f"Indices: {pat}")

                print(caption)

                # Generar gráfica en memoria
                xs = [x_i, a_i, b_i, c_i, d_i]
                ys = [df.loc[i, 'close'] for i in xs]

                plt.figure(figsize=(10, 5))
                plt.plot(df['close'], label='Precio')
                plt.plot(xs, ys, marker='o', color='red', linewidth=2, label='Gartley')

                for j, point in zip(xs, ys):
                    plt.text(j, point, f"{point:.2f}", fontsize=9, ha='center', va='bottom')

                plt.title(f"Patrón Gartley en {sym}")
                plt.xlabel("Índice (minuto)")
                plt.ylabel("Precio")
                plt.legend()
                plt.grid()
                plt.tight_layout()

                # Guardar imagen en buffer bytes
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()

                # Enviar imagen y mensaje por Telegram
                send_telegram_photo(buf, caption=caption)

