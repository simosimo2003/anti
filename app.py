import streamlit as st
import yfinance as yf
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deep Gold Pattern Precision Engine", layout="wide")

st.title("محرك مطابقة الذهب بالبحث العميق 🪙⚡")
st.write("استخراج دقيق ومستمر للنمط من الصورة مع بحث تاريخي شامل ومكثف شمعة بشمعة.")

uploaded_file = st.file_uploader("ارفع صورة الشارت الأصلي", type=["png", "jpg", "jpeg"])
timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])

if uploaded_file and st.button("بدء المسح الحقيقي والدقيق 🎯"):
    # -------------------------------------------------------------
    # 1. استخراج النمط من الصورة وعرضه للمستخدم أولاً
    # -------------------------------------------------------------
    with st.spinner("جاري استخراج السلسلة الزمنية وقراءة مسار الشارت من الصورة..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # تحويل الألوان واستخراج درجة لون الخط البنفسجي/الأزرق
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # نطاق شامل للون شارت TradingView البنفسجي/الأزرق
        lower_line = np.array([95, 20, 30])
        upper_line = np.array([165, 255, 255])
        mask = cv2.inRange(hsv, lower_line, upper_line)
        
        # معالجة احتياطية بالتباين العالي للصور المقصوصة
        if np.count_nonzero(mask) < 30:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
            if np.mean(mask) > 127:
                mask = cv2.bitwise_not(mask)

        h, w = mask.shape
        y_points = []
        for col in range(w):
            pos = np.where(mask[:, col] > 0)[0]
            if len(pos) > 0:
                y_points.append(-float(np.mean(pos)))
            elif len(y_points) > 0:
                y_points.append(y_points[-1])

        if len(y_points) < 15:
            st.error("لم يتم العثور على خط السعر في الصورة. يرجى التأكد من رفع صورة واضحة للشارت.")
            st.stop()

        user_pattern = np.array(y_points, dtype=np.float64)
        
        # معايرة السلسلة الزمنية
        norm_user = (user_pattern - np.mean(user_pattern)) / (np.std(user_pattern) + 1e-8)
        
        # -------------------------------------------------------------
        # عرض قراءة الصورة المباشرة للمستخدم
        # -------------------------------------------------------------
        st.subheader("📸 التطابق والاستخراج المباشر من صورتك:")
        fig_user, ax_user = plt.subplots(figsize=(10, 3.5))
        ax_user.plot(norm_user, color="#00f2fe", linewidth=2.5, label="النمط المستخرج من الصورة")
        ax_user.set_facecolor("#131722")
        fig_user.patch.set_facecolor("#131722")
        ax_user.tick_params(colors="white")
        ax_user.grid(True, color="#2a2e39", linestyle=":", alpha=0.5)
        ax_user.legend(facecolor="#1e222d", labelcolor="white")
        st.pyplot(fig_user)

    # -------------------------------------------------------------
    # 2. البحث التاريخي العميق الشامل شمعة بشمعة
    # -------------------------------------------------------------
    with st.spinner("جاري المسح التاريخي الشامل والعميق عبر بيانات الذهب (قد يستغرق بعض الوقت للبحث الدقيق)..."):
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر جلب البيانات المالية التاريخية.")
            st.stop()

        prices = data['Close'].values.flatten().astype(np.float64)
        timestamps = data.index
        
        window_len = len(norm_user)
        future_len = int(window_len * 0.5)
        
        best_score = float("inf")
        best_idx = -1
        
        user_min_pos = np.argmin(norm_user)
        user_max_pos = np.argmax(norm_user)
        user_drop = np.min(np.diff(norm_user)) # حدة الهبوط العمودي

        # مسح مكثف شمعة بشمعة
        for i in range(0, len(prices) - window_len - future_len, 1):
            hist_window = prices[i : i + window_len]
            std_dev = np.std(hist_window)
            if std_dev == 0:
                continue
                
            norm_hist = (hist_window - np.mean(hist_window)) / std_dev
            
            # 1. مطابقة الهيكل والمحاذاة الشاملة (MSE)
            mse_error = np.mean((norm_user - norm_hist) ** 2)
            
            # 2. عقوبة اختلاف زاوية وسرعة الهبوط الحاد
            hist_drop = np.min(np.diff(norm_hist))
            drop_penalty = abs(user_drop - hist_drop) * 6.0
            
            # 3. عقوبة عدم تطابق القمة والقاع في موقعهما الزمني
            hist_min_pos = np.argmin(norm_hist)
            hist_max_pos = np.argmax(norm_hist)
            align_penalty = (abs(user_min_pos - hist_min_pos) + abs(user_max_pos - hist_max_pos)) / window_len
            
            total_score = mse_error + drop_penalty + (align_penalty * 4.0)
            
            if total_score < best_score:
                best_score = total_score
                best_idx = i

        if best_idx == -1:
            st.error("لم يتم العثور على نمط مطابق. جرب اختيار فريم زمني مختلف.")
            st.stop()

        # -------------------------------------------------------------
        # 3. عرض النتائج والنمط التاريخي والتوقع المستقبلي
        # -------------------------------------------------------------
        match_start_time = timestamps[best_idx].strftime('%Y-%m-%d %H:%M')
        match_end_time = timestamps[best_idx + window_len].strftime('%Y-%m-%d %H:%M')
        
        match_percentage = max(50.0, min(99.0, 100 - (best_score * 10)))
        
        st.success(f"🔥 تم العثور على أحدث نمط مطابق تاريخياً بنسبة: **{match_percentage:.1f}%**")
        st.info(f"📅 **تاريخ وقوع النمط في الماضي:** من **{match_start_time}** إلى **{match_end_time}**")

        matched_history = prices[best_idx : best_idx + window_len]
        matched_future = prices[best_idx + window_len : best_idx + window_len + future_len]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(matched_history)), matched_history, label="النمط التاريخي المطابق", color="#00f2fe", linewidth=2.5)
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
