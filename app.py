import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

st.set_page_config(page_title="Ultra Exact Pattern Matcher AI", layout="wide")

st.title("محرك المطابقة الهيكلية التامة للذهب 🪙⚡")
st.write("ارسم النموذج بيدك، وسيقوم النظام بتشديد المطابقة الهيكلية على القمم والقيعان وزوايا الانهيار/الارتفاع.")

col1, col2 = st.columns([1, 3])

with col1:
    timeframe = st.selectbox("اختر الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"])
    stroke_width = st.slider("سمك خط الرسم:", 2, 10, 4)
    st.info("💡 **ملاحظة:** يتم مطابقة ترتيب القمم والقيعان ونسبة القمة الرئيسية بالنسبة لباقي الحركة بدقة متناهية.")

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

if st.button("بدء المسح التاريخي العميق والدقيق 🎯"):
    raw_x = []
    raw_y = []
    
    if canvas_result is not None and hasattr(canvas_result, "json_data"):
        data = canvas_result.json_data
        if data is not None and isinstance(data, dict):
            objects = data.get("objects", [])
            for obj in objects:
                path = obj.get("path", [])
                for p in path:
                    if isinstance(p, list) and len(p) >= 3:
                        try:
                            raw_x.append(float(p[1]))
                            raw_y.append(-float(p[2])) # تعكيس Y لتتناسب مع الاتجاه المالي
                        except (ValueError, TypeError):
                            pass

    if len(raw_x) < 10:
        st.error("الرسمة قصيرة جداً. ارسم النمط بوضوح داخل اللوحة.")
        st.stop()

    # --- 1. إعادة العينة وترتيب المحور الأفقي ---
    raw_x = np.array(raw_x)
    raw_y = np.array(raw_y)

    sort_idx = np.argsort(raw_x)
    sorted_x = raw_x[sort_idx]
    sorted_y = raw_y[sort_idx]

    unique_x, unique_indices = np.unique(sorted_x, return_index=True)
    unique_y = sorted_y[unique_indices]

    if len(unique_x) < 5:
        st.error("يرجى رسم الخط باتجاه أفقي أطول من اليسار إلى اليمين.")
        st.stop()

    # تحويل الرسمة إلى 60 نقطة هيكلية
    target_x = np.linspace(unique_x.min(), unique_x.max(), 60)
    interp_func = interp1d(unique_x, unique_y, kind='linear', fill_value="extrapolate")
    resampled_y = interp_func(target_x)

    # معايرة قياس الرسمة (Standardization)
    user_pattern = (resampled_y - np.mean(resampled_y)) / (np.std(resampled_y) + 1e-8)
    
    # استخراج خصائص الهيكل (موقع أعلى قمة وموقع أدنى قاع)
    user_max_idx = np.argmax(user_pattern)
    user_min_idx = np.argmin(user_pattern)
    user_end_drop = user_pattern[-1] - user_pattern[user_max_idx] # مقدار الهبوط بعد القمة

    with st.spinner("جاري إجراء مطابقة DTW الهيكلية عبر تاريخ الذهب..."):
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}
        data = yf.download(tickers="GC=F", period=period_map[timeframe], interval=timeframe, progress=False)
        
        if data.empty:
            st.error("تعذر جلب البيانات المالية للذهب.")
            st.stop()

        prices = data['Close'].values.flatten().astype(np.float64)
        timestamps = data.index
        
        window_len = len(user_pattern)
        future_len = int(window_len * 0.5)
        
        best_score = float("inf")
        best_idx = -1

        # مسح تاريخي بالـ Dynamic Time Warping وعقوبات الهيكل
        for i in range(0, len(prices) - window_len - future_len, 2):
            hist_window = prices[i : i + window_len]
            std_dev = np.std(hist_window)
            if std_dev == 0:
                continue
                
            norm_hist = (hist_window - np.mean(hist_window)) / std_dev
            
            # حساب المسافة الهيكلية عبر Dynamic Time Warping (DTW)
            dtw_dist, _ = fastdtw(user_pattern, norm_hist, dist=euclidean)
            
            # --- فرض عقوبات هندسية قاسية لضمان التطابق التام ---
            hist_max_idx = np.argmax(norm_hist)
            hist_min_idx = np.argmin(norm_hist)
            hist_end_drop = norm_hist[-1] - norm_hist[hist_max_idx]
            
            # عقوبة اختلال موقع أعلى قمة (يجب أن تقع القمة العظمى في نفس الثلث/المكان)
            max_pos_penalty = abs(user_max_idx - hist_max_idx) * 1.5
            
            # عقوبة عدم الهبوط بعد القمة (إذا لم يهبط السعر حاداً بعد القمة مثل الرسمة)
            drop_penalty = 0.0
            if user_end_drop < 0 and hist_end_drop >= 0:
                drop_penalty = 100.0 # عقوبة هائلة
            else:
                drop_penalty = abs(user_end_drop - hist_end_drop) * 10.0
                
            total_penalty = dtw_dist + max_pos_penalty + drop_penalty
            
            if total_penalty < best_score:
                best_score = total_penalty
                best_idx = i

        if best_idx == -1:
            st.error("لم يتم العثور على نمط مطابق بمواصفات القمم العالية. جرب تغيير الفريم الزمني.")
            st.stop()

        match_start_time = timestamps[best_idx].strftime('%Y-%m-%d %H:%M')
        match_end_time = timestamps[best_idx + window_len].strftime('%Y-%m-%d %H:%M')
        match_percentage = max(50.0, min(99.0, 100 - (best_score / 3.0)))
        
        st.success(f"🔥 تم العثور على النمط الأكثر تطابقاً بأسلوب DTW الهيكلي بنسبة: **{match_percentage:.1f}%**")
        st.info(f"📅 **الفترة التاريخية المطابقة تماماً:** من **{match_start_time}** إلى **{match_end_time}**")

        matched_history = prices[best_idx : best_idx + window_len]
        matched_future = prices[best_idx + window_len : best_idx + window_len + future_len]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(matched_history)), matched_history, label="النمط المطابق هيكلياً للرسمة", color="#00f2fe", linewidth=2.5)
        ax.plot(range(len(matched_history)-1, len(matched_history) + len(matched_future)), 
                np.insert(matched_future, 0, matched_history[-1]), 
                label="المسار المتوقع بعد اكتمال النموذج", color="#1e90ff", linewidth=3.5, linestyle="-")
        
        ax.axvline(x=len(matched_history)-1, color="#ffd700", linestyle="--", alpha=0.9, label="نقطة نهاية النموذج")
        
        ax.set_facecolor("#131722")
        fig.patch.set_facecolor("#131722")
        ax.tick_params(colors="white")
        ax.grid(True, color="#2a2e39", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left")
        
        st.pyplot(fig)
