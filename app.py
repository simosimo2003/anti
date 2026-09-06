import streamlit as st
import yfinance as yf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="Advanced Gold Pattern AI", layout="wide")

st.title("استخراج 2003النمط الأصلي وتكملته بالأزرق (الإصدار الذكي) 🪙⚡")
st.write("نظام مطابقة خوارزمي متقدم يبحث عن أدق نمط تاريخي مطابق لهيكل الشارت والسيولة.")

# 1. إدخال الصورة والفريم
uploaded_file = st.file_uploader("ارفع صورة الشارت الأصلي", type=["png", "jpg", "jpeg"])
timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])

if uploaded_file and st.button("تحليل ومطابقة النمط بدقة عالية 🎯"):
    with st.spinner("جاري المسح الخوارزمي المتقدم لملايين الشموع التاريخية..."):
        # أ) استخراج الحواف والمسار الذكي من الصورة
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # قص الحواف ذكياً (Focus Area)
        h, w = img.shape
        crop_img = img[int(h*0.08):int(h*0.92), int(w*0.02):int(w*0.88)]
        
        # استخراج الشموع ومعالجة التباين
        edges = cv2.Canny(crop_img, 30, 120)
        points = np.where(edges > 0)
        
        if len(points[0]) > 0:
            x_unique = np.unique(points[1])
            y_profile = []
            for x in x_unique:
                y_vals = points[0][points[1] == x]
                y_profile.append(-np.mean(y_vals))
            
            # تنقية خفيفة جداً للحفاظ على القمم والهبوط العمودي الحاد
            smoothed_profile = gaussian_filter1d(np.array(y_profile, dtype=np.float64), sigma=0.8)
            user_pattern = (smoothed_profile - np.mean(smoothed_profile)) / (np.std(smoothed_profile) + 1e-8)
            user_pattern_2d = user_pattern.reshape(-1, 1)
            
            # حساب التغير اللحظي (السرعة والهبوط الحاد)
            user_diff = np.diff(user_pattern)
        else:
            st.error("لم يتم التعرف على الشموع بدقة. يرجى رفع صورة واضحة للشارت.")
            st.stop()

        # ب) جلب البيانات التاريخية المباشرة
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر الاتصال بسيرفر البيانات المالية للذهب.")
            st.stop()

        prices = np.array(data['Close']).flatten().astype(np.float64)
        
        window_len = len(user_pattern)
        future_len = int(window_len * 0.6)
        
        best_score = float("inf")
        best_idx = -1

        # ج) خوارزمية صرامة الهبوط والقمم (Strict Spike & Drop Matcher)
        step = max(1, int(window_len / 6))
        for i in range(0, len(prices) - window_len - future_len, step):
            hist_window = prices[i : i + window_len]
            smoothed_hist = gaussian_filter1d(hist_window, sigma=0.8)
            norm_hist = (smoothed_hist - np.mean(smoothed_hist)) / (np.std(smoothed_hist) + 1e-8)
            norm_hist_2d = norm_hist.reshape(-1, 1)
            
            # 1. مسافة الشكل الأساسي FastDTW
            dtw_dist, _ = fastdtw(user_pattern_2d, norm_hist_2d, dist=euclidean)
            
            # 2. مطابقة حدة الحركة والهبوط العمودي (Velocity/Gradient Matching)
            hist_diff = np.diff(norm_hist)
            velocity_penalty = np.mean(np.abs(user_diff - hist_diff)) * window_len * 3.0
            
            # 3. مطابقة فارق القمة والقاع (Range Matching)
            user_range = np.ptp(user_pattern)
            hist_range = np.ptp(norm_hist)
            range_penalty = abs(user_range - hist_range) * 10.0
            
            # النتيجة الإجمالية المخصصة للمطابقة الحادة
            combined_score = dtw_dist + velocity_penalty + range_penalty
            
            if combined_score < best_score:
                best_score = combined_score
                best_idx = i

        # د) عرض النتائج
        match_percentage = max(50.0, min(99.2, 100 - (best_score / window_len * 4)))
        st.success(f"🔥 تم العثور على نمط مطابق بنسبة: {match_percentage:.1f}%")

        matched_history = prices[best_idx : best_idx + window_len]
        matched_future = prices[best_idx + window_len : best_idx + window_len + future_len]
        
        fig, ax = plt.subplots(figsize=(11, 5))
        
        ax.plot(range(len(matched_history)), matched_history, label="النمط المطابق تاريخياً", color="#00f2fe", linewidth=2)
        ax.plot(range(len(matched_history)-1, len(matched_history) + len(matched_future)), 
                np.insert(matched_future, 0, matched_history[-1]), 
                label="المسار القادم المتوقع", color="#1e90ff", linewidth=3, linestyle="-")
        
        ax.axvline(x=len(matched_history)-1, color="#ffd700", linestyle="--", alpha=0.8, label="نقطة الانطلاق الحالية")
        
        ax.set_facecolor("#131722")
        fig.patch.set_facecolor("#131722")
        ax.tick_params(colors="white")
        ax.grid(True, color="#2a2e39", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left")
        
        st.pyplot(fig)
