import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# === Configuración API ===
api_key = 'd0b66f404eba416fb06551c378278454'

def descargar_datos(symbol, interval='5min', outputsize=5000):
    url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if 'values' in data:
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
        df = df.sort_values('datetime', ascending=False)
        return df
    else:
        raise ValueError(f"Error en los datos para {symbol}: {data.get('message', 'Datos no disponibles')}")

def calcular_indicadores(df):
    df['return'] = ((df['close'] - df['close'].shift(1))/df['close'].shift(1))*100

    # RSI
    window_length = 100
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_crossover'] = 0
    df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'macd_crossover'] = 1
    df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'macd_crossover'] = -1

    # Bollinger Bands
    period = 100
    k = 2
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (k * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (k * df['bb_std'])

    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(10)

    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()

    df.drop(columns=['tr1', 'tr2', 'tr3', 'true_range'], inplace=True)
    df.dropna(inplace=True)
    df['direction'] = np.where(df['return'] > 0, 1, 0)
    df.drop(columns=['datetime'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def entrenar_modelo(df, symbol, threshold=0.7):
    df_train, df_valid = train_test_split(df, test_size=0.25, random_state=42)
    X_train = df_train.drop(['direction', 'return'], axis=1)
    y_train = df_train['direction']
    X_valid = df_valid.drop(['direction', 'return'], axis=1)
    y_valid = df_valid['direction']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=9)
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_valid_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)
    cm = confusion_matrix(y_valid, y_pred)

    print(f"=== {symbol} ===")
    print(f"Precisión: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusión:")
    print(cm)

    # Guardar
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    path = f'C:\\Users\\Administrator\\Downloads\\Modelos\\{safe_symbol}'
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, os.path.join(path, 'modelo.pkl'))
    joblib.dump(scaler, os.path.join(path, 'scaler.pkl'))
    joblib.dump(list(X_train.columns), os.path.join(path, 'features.pkl'))


    return model, scaler

# === Uso con cualquier símbolo ===
for symbol in ['UNI/USD', 'FET/USD', 'XTZ/USD']:
    try:
        df_raw = descargar_datos(symbol, interval='5min')
        df = calcular_indicadores(df_raw)
        entrenar_modelo(df, symbol)
    except Exception as e:
        print(f"[{symbol}] ERROR: {e}")
