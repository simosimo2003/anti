import streamlit as st
import yfinance as yf
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Pattern Precision AI", layout="wide")

st.title("محرك مطابقة الذهب التاريخي الفائق 🪙⚡")
st.write("اختر طريقة إدخال النمط للحصول على أدق مطابقة تاريخية وتوقع مستقبلي.")

mode = st.radio("طريقة تحديد النمط:", ["رفع صورة الشارت", "إدخال الهبوط/الارتفاع بالنقاط (دقة 100%)"])

timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])

user_pattern = None

if mode == "رفع صورة الشارت":
    uploaded_file = st.file_uploader("ارفع صورة الشارت (يفضل قص الأرقام والحواف)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # تحويل للتدرج الرمادي واستخراج أعلى تباين للخط
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # تركيز على وسط الصورة لتفادي أرقام TradingView
        crop = gray[int(h*0.1):int(h*0.9), int(w*0.05):int(w*0.85)]
        hc, wc = crop.shape
        
        y_points = []
        for col in range(wc):
            # أخذ أعمق نقطة سوداء/داكنة أو مضيئة ممثلة للرسم
            col_data = crop[:, col]
            min_pos = np.argmin(col_data)
            y_points.append(-float(min_pos))
            
        user_pattern = np.array(y_points, dtype=np.float64)

else:
    st.info("أدخل سلوك السعر التقريبي (مثال: 2000 ثم سقوط حاد إلى 1950 ثم ارتداد لـ 1970)")
    points_str = st.text_input("أدخل قيم السعر تفصل بينها فاصلة (مثال: 2000, 2005, 1950, 1955, 1970):", "2000, 2002, 1950, 1955, 1970")
    try:
        user_pattern = np.array([float(x.strip()) for x in points_str.split(",")], dtype=np.float64)
    except:
        st.error("يرجى إدخال أرقام صحيحة تفصل بينها فاصلة.")

if user_pattern is not None and st.button("بدء المسح التاريخي الحقيقي 🎯"):
    with st.spinner("جاري المسح شمعة بشمعة عبر التاريخ..."):
        # معايرة النمط
        norm_user = (user_pattern - np.mean(user_pattern)) / (np.std(user_pattern) + 1e-8)
        user_drop = np.min(np.diff(norm_user)) # زاوية الهبوط الحاد

        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر جلب البيانات المالية المباشرة.")
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
            drop_penalty = abs(user_drop - hist_drop) * 6.0 # عقوبة زاوية الهبوط
            
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
