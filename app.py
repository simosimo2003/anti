import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import cv2

st.set_page_config(page_title="Ultra Draw & Match AI", layout="wide")

st.title("محرك الرسم والتطابق التاريخي للذهب 🪙⚡")
st.write("ارسم النمط المباشر بيدك على اللوحة السوداء أدناه، وسيقوم النظام بالبحث العميق شمعة بشمعة عبر تاريخ الذهب ليعطيك النمط المطابق والتوقع المستقبلي.")

# 1. إعدادات اللوحة والفريم
col1, col2 = st.columns([1, 3])

with col1:
    timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])
    stroke_width = st.slider("سمك خط الرسم:", 2, 10, 4)
    st.info("💡 **طريقة الاستخدام:** ارسم النموذج من اليسار إلى اليمين داخل اللوحة السوداء بنفس شكل الحركة (قمة، قاع، أو هبوط عمودي).")

with col2:
    st.write("### 🎨 لوحة الرسم التفاعلية:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color="#00f2fe",
        background_color="#131722",
        height=320,
        width=700,
        drawing_mode="freedraw",
        key="canvas",
    )

# حفظ رسمة المستخدم تلقائياً في session_state لحمايتها من الاختفاء عند ضغط الزر
if canvas_result is not None and canvas_result.image_data is not None:
    # تحقق من وجود بكسلات مرسومة فعلياً
    raw_img = canvas_result.image_data.astype(np.uint8)
    if np.any(raw_img[:, :, :3] > 20):
        st.session_state["saved_drawing"] = raw_img

if st.button("بدء المسح التاريخي العميق والدقيق 🎯"):
    # استرجاع الرسمة إما من الكانفاس الحالي أو من الجلسة المحفوظة
    img = st.session_state.get("saved_drawing", None)

    if img is None:
        st.error("لم يتم العثور على رسمة. يرجى الرسم داخل اللوحة السوداء بوضوح أولاً.")
        st.stop()

    # دمج قنوات الألوان لتحديد خط الرسم
    drawn_mask = (img[:, :, 0] > 20) | (img[:, :, 1] > 20) | (img[:, :, 2] > 20)

    h, w = drawn_mask.shape
    y_points = []

    for col in range(w):
        pos = np.where(drawn_mask[:, col])[0]
        if len(pos) > 0:
            y_points.append(-float(np.mean(pos)))
        elif len(y_points) > 0:
            y_points.append(y_points[-1])

    if len(y_points) < 5:
        st.error("الرسمة قصيرة جداً. يرجى رسم مسار واضح ينتهي باتجاه اليمين ثم اضغط الزر.")
        st.stop()

    user_pattern = np.array(y_points, dtype=np.float64)
    norm_user = (user_pattern - np.mean(user_pattern)) / (np.std(user_pattern) + 1e-8)
    user_drop = np.min(np.diff(norm_user))

    with st.spinner("جاري المسح التاريخي العميق (شمعة بشمعة) عبر كامل تاريخ الذهب..."):
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر جلب البيانات المالية المباشرة للذهب.")
            st.stop()

        prices = data['Close'].values.flatten().astype(np.float64)
        timestamps = data.index
        
        window_len = len(norm_user)
        future_len = int(window_len * 0.5)
        
        best_score = float("inf")
        best_idx = -1
        
        user_min_pos = np.argmin(norm_user)
        user_max_pos = np.argmax(norm_user)

        for i in range(0, len(prices) - window_len - future_len, 1):
            hist_window = prices[i : i + window_len]
            std_dev = np.std(hist_window)
            if std_dev == 0:
                continue
                
            norm_hist = (hist_window - np.mean(hist_window)) / std_dev
            
            shape_error = np.mean((norm_user - norm_hist) ** 2)
            hist_drop = np.min(np.diff(norm_hist))
            drop_penalty = abs(user_drop - hist_drop) * 5.0
            
            hist_min_pos = np.argmin(norm_hist)
            hist_max_pos = np.argmax(norm_hist)
            align_penalty = (abs(user_min_pos - hist_min_pos) + abs(user_max_pos - hist_max_pos)) / window_len
            
            total_score = shape_error + drop_penalty + (align_penalty * 3.5)
            
            if total_score < best_score:
                best_score = total_score
                best_idx = i

        if best_idx == -1:
            st.error("لم يتم العثور على نمط مطابق للرسم. جرب تغيير الفريم الزمني.")
            st.stop()

        match_start_time = timestamps[best_idx].strftime('%Y-%m-%d %H:%M')
        match_end_time = timestamps[best_idx + window_len].strftime('%Y-%m-%d %H:%M')
        
        match_percentage = max(50.0, min(99.0, 100 - (best_score * 12)))
        
        st.success(f"🔥 تم العثور على أحدث نمط تاريخي مطابق لرسمتك بنسبة: **{match_percentage:.1f}%**")
        st.info(f"📅 **تاريخ وقوع هذا النمط في الماضي:** من **{match_start_time}** إلى **{match_end_time}**")

        matched_history = prices[best_idx : best_idx + window_len]
        matched_future = prices[best_idx + window_len : best_idx + window_len + future_len]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(matched_history)), matched_history, label="النمط المطابق تاريخياً لرسمتك", color="#00f2fe", linewidth=2.5)
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
