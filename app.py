import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2
from PIL import Image

st.set_page_config(page_title="Gold Pattern Extension", layout="centered")

st.title("🟡 استخراج النمط الأصلي وتكملته بالأزرق")
st.write("ارفع صورة الشارت وسيقوم التطبيق بمطابقتها وتكملة المسار القادم باللون الأزرق المضيء.")

@st.cache_data(ttl=3600)
def fetch_gold_deep_data(timeframe="15m", limit=6000):
    try:
        ticker = yf.Ticker("GC=F")
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "4h": "730d", "1d": "max"}
        period = period_map.get(timeframe, "60d")
        df = ticker.history(period=period, interval=timeframe)
        if df.empty or len(df) < 100:
            return None
        df = df.reset_index()
        time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df['FormattedTime'] = df[time_col].dt.strftime('%Y-%m-%d %H:%M')
        return df.tail(limit).reset_index(drop=True)
    except Exception as e:
        return None

def crop_exact_chart_area(pil_img):
    open_cv_image = np.array(pil_img.convert('RGB')) 
    img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    h_orig, w_orig, _ = img_bgr.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    candle_mask = mask_green | mask_red
    contours, _ = cv2.findContours(candle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_pts = np.vstack([c for c in contours if cv2.contourArea(c) > 5])
        x, y, w, h = cv2.boundingRect(all_pts)
        pad = 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_orig, x + w + pad), min(h_orig, y + h + pad)
        chart_crop = pil_img.crop((x0, y0, x1, y1))
        cropped_mask = candle_mask[y0:y1, x0:x1]
    else:
        chart_crop = pil_img
        cropped_mask = candle_mask

    cw, ch = chart_crop.size
    num_candles = 35
    step = cw / num_candles
    series_peaks = []

    for i in range(num_candles):
        x_start = int(i * step)
        x_end = max(x_start + 1, int((i + 1) * step))
        slice_m = cropped_mask[:, x_start:x_end]
        y_indices = np.where(slice_m > 0)[0]
        if len(y_indices) > 0:
            min_y = np.min(y_indices)
            val = (ch - min_y) / ch
        else:
            val = 0.5 if len(series_peaks) == 0 else series_peaks[-1]
        series_peaks.append(val)

    series_peaks = np.array(series_peaks, dtype=np.float32)
    s_min, s_max = series_peaks.min(), series_peaks.max()
    norm_series = (series_peaks - s_min) / (s_max - s_min) if s_max - s_min > 1e-5 else np.zeros_like(series_peaks)

    return chart_crop, norm_series

def search_gold_pattern_fast(norm_target, prices, future_candles=30):
    pattern_len = len(norm_target)
    n = len(prices)
    if n < pattern_len + future_candles:
        return None
    best_similarity, best_idx = -1.0, -1

    for i in range(0, n - pattern_len - future_candles, 2):
        window = prices[i : i + pattern_len]
        w_min, w_max = window.min(), window.max()
        if w_max - w_min < 1e-5:
            continue
        norm_window = (window - w_min) / (w_max - w_min)
        corr = np.corrcoef(norm_target, norm_window)[0, 1]
        if np.isnan(corr): corr = 0
        similarity = corr * 100.0
        if similarity > best_similarity:
            best_similarity, best_idx = similarity, i

    return best_idx, best_similarity

def plot_original_with_future_extension(df_history, best_idx, pattern_len, future_len, cropped_img, similarity_score):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1.2, 1.8]})
    fig.patch.set_facecolor('#131722')

    ax_top.imshow(cropped_img)
    ax_top.set_title("1️⃣ الرسم البياني الأصلي المستخرج من صورتك", color="white", fontsize=11)
    ax_top.axis('off')

    ax_bot.set_facecolor('#131722')
    sub_df = df_history.iloc[best_idx : best_idx + pattern_len + future_len].copy().reset_index(drop=True)
    match_end_time = df_history.iloc[best_idx + pattern_len - 1]['FormattedTime']

    for i in range(pattern_len):
        row = sub_df.iloc[i]
        color = '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        ax_bot.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1.2)
        ax_bot.bar(i, max(abs(row['Close'] - row['Open']), 0.05), bottom=min(row['Open'], row['Close']), color=color, width=0.6)

    for i in range(pattern_len, len(sub_df)):
        row = sub_df.iloc[i]
        color = '#00e5ff'
        ax_bot.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1.5)
        ax_bot.bar(i, max(abs(row['Close'] - row['Open']), 0.05), bottom=min(row['Open'], row['Close']), color=color, width=0.6)

    ax_bot.axvline(x=pattern_len - 0.5, color='#ff9800', linestyle='--', linewidth=1.5)
    ax_bot.set_title(f"2️⃣ التكملة القادمة باللون الأزرق (طابق الذهب بتاريخ: {match_end_time})", color="white", fontsize=10)
    ax_bot.tick_params(colors="white")
    ax_bot.grid(True, color='#2a2e39', alpha=0.3)

    plt.tight_layout()
    return fig

uploaded_file = st.file_uploader("ارفع صورة الشارت الأصلي", type=["png", "jpg", "jpeg"])
timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "4h", "1d"], index=0)

if st.button("استخراج وتكملة النمط بالأزرق 🎯") and uploaded_file is not None:
    with st.spinner("جاري المطابقة الحسابية..."):
        df = fetch_gold_deep_data(timeframe=timeframe, limit=6000)
        if df is not None:
            pil_img = Image.open(uploaded_file)
            num_candles = 35
            cropped_img, norm_target = crop_exact_chart_area(pil_img)
            result = search_gold_pattern_fast(norm_target, df['Close'].values, future_candles=30)
            
            if result:
                best_idx, similarity_score = result
                fig = plot_original_with_future_extension(df, best_idx, num_candles, 30, cropped_img, similarity_score)
                st.success(f"تمت المطابقة بنسبة: {similarity_score:.1f}%")
                st.pyplot(fig)
            else:
                st.error("لم يتم العثور على مطابقة.")