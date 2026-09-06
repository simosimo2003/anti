import streamlit as st
import yfinance as yf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Advanced Pattern Engine AI", layout="wide")

st.title("محرك المطابقة الذكي العميق (Deep Pattern Engine) 🪙⚡")
st.write("نظام تحليل بصري ورياضي متقدم لمطابقة هيكل الشارت، السرعة، والانهيارات الحادة شمعة بشمعة.")

# 1. مدخلات الصورة والفريم الزمني
uploaded_file = st.file_uploader("ارفع صورة الشارت الأصلي", type=["png", "jpg", "jpeg"])
timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])

if uploaded_file and st.button("بدء المسح الذكي الشامل والعميق 🎯"):
    with st.spinner("جاري معالجة البكسلات واستخراج النمط بدقة رياضية..."):
        # أ) قراءة الصورة وعزل خط الشارت عبر الألوان (HSV Color Isolation)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        
        # تركيز المسح على منطقة الرسم البياني مجردة
        crop_img = img[int(h*0.10):int(h*0.85), int(w*0.02):int(w*0.88)]
        h_c, w_c, _ = crop_img.shape
        
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # استخراج الخط الأرجواني/الأزرق الخاص بشارت TradingView
        # نطاق شامل للألوان المضيئة والخطوط في خلفية الداكنة
        lower_line = np.array([100, 40, 40])
        upper_line = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_line, upper_line)
        
        # إذا لم يتوفر لون محدد، نعتمد الاستخراج بالتدرج الرمادي عالي التباين
        if np.sum(mask) == 0:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # استخراج منحنى السعر لكل عمود بكسل (Pixel-by-Pixel Column Extraction)
        y_profile = []
        x_indices = []
        for col in range(w_c):
            rows = np.where(mask[:, col] > 0)[0]
            if len(rows) > 0:
                # أخذ متوسط موقع الخط في العمود
                y_profile.append(-np.mean(rows))
                x_indices.append(col)
                
        if len(y_profile) < 20:
            # طريقة احتياطية دقيقة تعتمد على أغمق نقاط السعر
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            y_profile = [-np.argmin(gray[:, col]) for col in range(w_c)]

        user_pattern = np.array(y_profile, dtype=np.float64)
        
        # معايرة المعالم للنمط المرفوع (Standardization)
        user_pattern = (user_pattern - np.mean(user_pattern)) / (np.std(user_pattern) + 1e-8)
        
        # حساب السرعة والتسارع (First and Second Derivatives)
        user_grad = np.gradient(user_pattern)
        user_acc = np.gradient(user_grad)

    with st.spinner("جاري مطابقة النمط عبر تاريخ الذهب والبحث عن أعمق انكسار عمودي..."):
        # ب) جلب بيانات الذهب التاريخية
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر جلب البيانات المالية المباشرة.")
            st.stop()

        prices = data['Close'].values.flatten().astype(np.float64)
        timestamps = data.index
        
        window_len = len(user_pattern)
        future_len = int(window_len * 0.5)
        
        best_score = float("inf")
        best_idx = -1
        
        # موقع أدنى قاع وأعلى قمة في صورة المستخدم
        user_min_idx = np.argmin(user_pattern)
        user_max_idx = np.argmax(user_pattern)

        # ج) خوارزمية البحث الشامل الحازمة (Strict Matcher)
        for i in range(0, len(prices) - window_len - future_len, 1):
            hist_window = prices[i : i + window_len]
            std_dev = np.std(hist_window)
            if std_dev == 0:
                continue
                
            norm_hist = (hist_window - np.mean(hist_window)) / std_dev
            
            # 1. مطابقة المسار الهيكلي الإجمالي (Shape Error)
            shape_error = np.mean(np.abs(user_pattern - norm_hist))
            
            # 2. مطابقة سرعة وسعة الهبوط العمودي (Velocity & Drop Match)
            hist_grad = np.gradient(norm_hist)
            velocity_error = np.mean(np.abs(user_grad - hist_grad))
            
            # 3. عقوبة موقع القاع الأعمق (Spike Drop Placement Penalty)
            hist_min_idx = np.argmin(norm_hist)
            hist_max_idx = np.argmax(norm_hist)
            peak_valley_penalty = (abs(user_min_idx - hist_min_idx) + abs(user_max_idx - hist_max_idx)) / window_len
            
            # النتيجة المركبة المحسوبة للأداء العالي
            total_score = shape_error + (velocity_error * 3.5) + (peak_valley_penalty * 2.5)
            
            if total_score < best_score:
                best_score = total_score
                best_idx = i

        # د) عرض التواريخ والنتائج
        match_start_time = timestamps[best_idx].strftime('%Y-%m-%d %H:%M')
        match_end_time = timestamps[best_idx + window_len].strftime('%Y-%m-%d %H:%M')
        
        # نسبة المطابقة الرياضية
        match_percentage = max(50.0, min(99.2, 100 - (best_score * 16)))
        
        st.success(f"🔥 تم العثور على أحدث نمط مطابق تاريخياً بنسبة: **{match_percentage:.1f}%**")
        st.info(f"📅 **تاريخ وقوع هذا النمط في الماضي:** من **{match_start_time}** إلى **{match_end_time}**")

        # هـ) الرسم البياني النهائي (النمط التاريخي + التوقع الممتد بالأزرق)
        matched_history = prices[best_idx : best_idx + window_len]
        matched_future = prices[best_idx + window_len : best_idx + window_len + future_len]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # رسم النمط المطابق
        ax.plot(range(len(matched_history)), matched_history, label="النمط المطابق تاريخياً", color="#00f2fe", linewidth=2.5)
        
        # رسم التوقع القادم الممتد باللون الأزرق
        ax.plot(range(len(matched_history)-1, len(matched_history) + len(matched_future)), 
                np.insert(matched_future, 0, matched_history[-1]), 
                label="المسار القادم المتوقع", color="#1e90ff", linewidth=3.5, linestyle="-")
        
        # الخط الأصفر الفاصل
        ax.axvline(x=len(matched_history)-1, color="#ffd700", linestyle="--", alpha=0.9, label="نقطة الانطلاق الحالية (نسخ التوقع)")
        
        ax.set_facecolor("#131722")
        fig.patch.set_facecolor("#131722")
        ax.tick_params(colors="white")
        ax.grid(True, color="#2a2e39", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left")
        
        st.pyplot(fig)
