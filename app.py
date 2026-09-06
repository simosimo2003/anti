import streamlit as st
import yfinance as yf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Deep Pattern Matcher AI", layout="wide")

st.title("مطابقة النمط التاريخي بدقة فائقة 🪙⚡")
st.write("نظام مسح شامل (شمعة بشمعة) لاستخراج أدق نمط تاريخي مطابق واستخراج تاريخ حدوثه المباشر.")

# 1. إدخال الصورة والفريم
uploaded_file = st.file_uploader("ارفع صورة الشارت الأصلي", type=["png", "jpg", "jpeg"])
timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])

if uploaded_file and st.button("بدء المسح التاريخي الشامل والعميق 🎯"):
    with st.spinner("جاري إجراء مسح دقيق (شمعة بشمعة) عبر السلسلة الزمنية للذهب... قد يستغرق لحظات قليلة..."):
        # أ) استخراج النمط الحقيقي من الصورة بدقة العالية
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        h, w = img.shape
        crop_img = img[int(h*0.08):int(h*0.92), int(w*0.02):int(w*0.88)]
        
        edges = cv2.Canny(crop_img, 30, 120)
        points = np.where(edges > 0)
        
        if len(points[0]) > 0:
            x_unique = np.unique(points[1])
            y_profile = []
            for x in x_unique:
                y_vals = points[0][points[1] == x]
                y_profile.append(-np.mean(y_vals))
            
            # الاحتفاظ بالصورة الأصلية بدون تشويه القمم أو الهبوط العمودي
            raw_user_pattern = np.array(y_profile, dtype=np.float64)
            user_pattern = gaussian_filter1d(raw_user_pattern, sigma=0.4) # تنعيم متناهي الصغر
            user_pattern = (user_pattern - np.mean(user_pattern)) / (np.std(user_pattern) + 1e-8)
            
            user_diff = np.diff(user_pattern)
        else:
            st.error("لم يتم التعرف على حركة الشموع. يرجى رفع صورة واضحة للشارت.")
            st.stop()

        # ب) جلب البيانات التاريخية الكاملة مع التواريخ (Timestamps)
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر جلب البيانات المالية المباشرة للذهب.")
            st.stop()

        prices = data['Close'].values.flatten().astype(np.float64)
        timestamps = data.index
        
        window_len = len(user_pattern)
        future_len = int(window_len * 0.5)
        
        best_score = float("inf")
        best_idx = -1

        # ج) مسح شامل حقيقي شمعة بشمعة (step = 1)
        # البحث في كامل السلسلة بدون تخطي أي شمعة
        for i in range(0, len(prices) - window_len - future_len, 1):
            hist_window = prices[i : i + window_len]
            norm_hist = (hist_window - np.mean(hist_window)) / (np.std(hist_window) + 1e-8)
            
            # 1. المطابقة الهيكلية الحادة
            shape_error = np.mean(np.abs(user_pattern - norm_hist))
            
            # 2. مطابقة سرعة وانكسار الهبوط (Velocity Gradient)
            hist_diff = np.diff(norm_hist)
            velocity_error = np.mean(np.abs(user_diff - hist_diff))
            
            # 3. عقوبة الاختلاف في موقع أدنى قاع وأعلى قمة
            user_min, user_max = np.argmin(user_pattern), np.argmax(user_pattern)
            hist_min, hist_max = np.argmin(norm_hist), np.argmax(norm_hist)
            extrema_penalty = (abs(user_min - hist_min) + abs(user_max - hist_max)) / window_len
            
            score = shape_error + (velocity_error * 2.5) + (extrema_penalty * 1.5)
            
            if score < best_score:
                best_score = score
                best_idx = i

        # د) استخراج التاريخ والنتائج
        match_start_time = timestamps[best_idx].strftime('%Y-%m-%d %H:%M')
        match_end_time = timestamps[best_idx + window_len].strftime('%Y-%m-%d %H:%M')
        
        # حساب نسبة المطابقة الواقعية
        match_percentage = max(50.0, min(99.5, 100 - (best_score * 20)))
        
        st.success(f"🔥 تم العثور على النمط المطابق بنسبة: **{match_percentage:.1f}%**")
        st.info(f"📅 **تاريخ وقوع النمط في الماضي:** من **{match_start_time}** إلى **{match_end_time}**")

        # هـ) رسم النمط المرفوع + المطابق التاريخي + التوقع القادم
        matched_history = prices[best_idx : best_idx + window_len]
        matched_future = prices[best_idx + window_len : best_idx + window_len + future_len]
        
        # ضبط مقياس الصورة الأصلية لتبدأ من نفس سعر البداية التاريخي
        scaled_user_pattern = (user_pattern * np.std(matched_history)) + np.mean(matched_history)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. رسم النمط التاريخي المطابق
        ax.plot(range(len(matched_history)), matched_history, label="النمط المطابق تاريخياً", color="#00f2fe", linewidth=2.5)
        
        # 2. رسم المسار المتوقع القادم باللون الأزرق الممتد
        ax.plot(range(len(matched_history)-1, len(matched_history) + len(matched_future)), 
                np.insert(matched_future, 0, matched_history[-1]), 
                label="المسار القادم المتوقع", color="#1e90ff", linewidth=3, linestyle="-")
        
        # 3. خط بداية التوقع المستقبلي
        ax.axvline(x=len(matched_history)-1, color="#ffd700", linestyle="--", alpha=0.9, label="نقطة الانطلاق الحالية (تنسيخ التوقع)")
        
        ax.set_facecolor("#131722")
        fig.patch.set_facecolor("#131722")
        ax.tick_params(colors="white")
        ax.grid(True, color="#2a2e39", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left")
        
        st.pyplot(fig)
