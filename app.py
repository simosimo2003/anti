import streamlit as st
import yfinance as yf
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Line Matcher AI", layout="wide")

st.title("محرك مطابقة خط الشارت المباشر 🪙⚡")

uploaded_file = st.file_uploader("ارفع صورة الشارت المقصوصة", type=["png", "jpg", "jpeg"])
timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])

if uploaded_file and st.button("بدء المسح الحقيقي 🎯"):
    with st.spinner("جاري استخراج خط السعر البنفسجي وتصفية الخلفية..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # تحويل الألوان لنطاق HSV لعزل اللون البنفسجي/الأزرق بدقة
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # درجة اللون البنفسجي الخاص بشارت TradingView الظاهر في صورتك
        lower_purple = np.array([100, 30, 40])
        upper_purple = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # إذا كانت الصورة مقصوصة بشكل جيد، نستخرج مسار السعر
        h, w = mask.shape
        y_points = []
        
        for col in range(w):
            pos = np.where(mask[:, col] > 0)[0]
            if len(pos) > 0:
                y_points.append(-float(np.mean(pos)))
            elif len(y_points) > 0:
                y_points.append(y_points[-1]) # استكمال النقاط المفقودة

        if len(y_points) < 20:
            st.error("لم يتمكن السكربت من العثور على الخط البنفسجي. يرجى قص حواف الصورة والتركيز على الخط فقط.")
            st.stop()

        user_pattern = np.array(y_points, dtype=np.float64)
        norm_user = (user_pattern - np.mean(user_pattern)) / (np.std(user_pattern) + 1e-8)
        user_drop = np.min(np.diff(norm_user)) # قياس زاوية السقوط الحاد

    with st.spinner("جاري البحث في تاريخ الذهب عن النمط المطابق..."):
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر جلب البيانات المالية.")
            st.stop()

        prices = data['Close'].values.flatten().astype(np.float64)
        timestamps = data.index
        
        window_len = len(norm_user)
        future_len = int(window_len * 0.5)
        
        best_score = float("inf")
        best_idx = -1

        for i in range(0, len(prices) - window_len - future_len, 1):
            hist_window = prices[i : i + window_len]
            if np.std(hist_window) == 0:
                continue
                
            norm_hist = (hist_window - np.mean(hist_window)) / np.std(hist_window)
            hist_drop = np.min(np.diff(norm_hist))
            
            shape_diff = np.mean(np.abs(norm_user - norm_hist))
            drop_penalty = abs(user_drop - hist_drop) * 8.0 # عقوبة حازمة للهبوط العمودي
            
            score = shape_diff + drop_penalty
            
            if score < best_score:
                best_score = score
                best_idx = i

        match_start_time = timestamps[best_idx].strftime('%Y-%m-%d %H:%M')
        match_end_time = timestamps[best_idx + window_len].strftime('%Y-%m-%d %H:%M')
        
        match_percentage = max(50.0, min(99.0, 100 - (best_score * 10)))
        
        st.success(f"🔥 تم العثور على النمط المطابق بنسبة: **{match_percentage:.1f}%**")
        st.info(f"📅 **تاريخ وقوع النمط في الماضي:** من **{match_start_time}** إلى **{match_end_time}**")

        matched_history = prices[best_idx : best_idx + window_len]
        matched_future = prices[best_idx + window_len : best_idx + window_len + future_len]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(matched_history)), matched_history, label="النمط المطابق تاريخياً", color="#00f2fe", linewidth=2.5)
        ax.plot(range(len(matched_history)-1, len(matched_history) + len(matched_future)), 
                np.insert(matched_future, 0, matched_history[-1]), 
                label="المسار القادم المتوقع", color="#1e90ff", linewidth=3.5, linestyle="-")
        
        ax.axvline(x=len(matched_history)-1, color="#ffd700", linestyle="--", alpha=0.9, label="نقطة الانطلاق الحالية")
        
        ax.set_facecolor("#131722")
        fig.patch.set_facecolor("#131722")
        ax.tick_params(colors="white")
        ax.grid(True, color="#2a2e39", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left")
        
        st.pyplot(fig)
