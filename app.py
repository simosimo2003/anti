import time
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.lib.stride_tricks import sliding_window_view

st.set_page_config(page_title="محرك مطابقة الأنماط - الوضع الأقصى", layout="wide")

st.title("🪙 محرك مطابقة الأنماط التاريخية للذهب - الوضع الأقصى للبحث")
st.caption(
    "بحث متعدد المصادر × متعدد المقاييس الزمنية (Multi-Scale) × مطابقة هجينة "
    "(بيرسون + DTW) × توقع مرجّح بمسار واحد قوي + نطاق ثقة إحصائي"
)

st.warning(
    "⚠️ حتى مع أقوى خوارزمية بحث: تشابه شكل تاريخي لا يضمن تكرار نفس الحركة مستقبلاً. "
    "المسار الأزرق أدناه هو **أفضل تقدير مرجّح إحصائيًا بناءً على ما حدث فعليًا بعد أقرب الحالات "
    "المشابهة تاريخيًا** — وليس تنبؤًا مؤكدًا. كلما ضاق نطاق الثقة حوله كانت الحالات التاريخية أكثر اتساقًا."
)

# ---------------------------------------------------------------------------
# إعدادات المستخدم
# ---------------------------------------------------------------------------
col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    uploaded_file = st.file_uploader("ارفع صورة الشارت الأصلي", type=["png", "jpg", "jpeg"])
with col_b:
    timeframe = st.selectbox("الفريم الزمني", ["1m", "5m", "15m", "1h", "1d"], index=3)
with col_c:
    top_k = st.slider("عدد أفضل الحالات المستقلة المستخدمة في التوقع", 5, 50, 15)

TICKER_MAP = {
    "GC=F — عقود الذهب الأمريكية (COMEX)": "GC=F",
    "MGC=F — عقود الذهب المصغرة": "MGC=F",
    "XAUUSD=X — سعر السبوت (Spot)": "XAUUSD=X",
}
sources_selected = st.multiselect(
    "مصادر بيانات الذهب",
    list(TICKER_MAP.keys()),
    default=list(TICKER_MAP.keys()),
)
selected_tickers = [TICKER_MAP[s] for s in sources_selected] or ["GC=F"]

max_mode = st.checkbox(
    "🔥 تفعيل الوضع الأقصى للبحث (أبطأ بكثير — قد يستغرق عدة دقائق — لكن أشمل وأدق)",
    value=True,
)

if max_mode:
    SCALE_FACTORS = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    DTW_CANDIDATE_POOL = 250
    DTW_RADIUS = 20
else:
    SCALE_FACTORS = [0.95, 1.00, 1.05]
    DTW_CANDIDATE_POOL = 60
    DTW_RADIUS = 12

BASE_PATTERN_LEN = 120
COMMON_HORIZON_POINTS = 60   # طول موحّد (نسبي) لدمج مسارات المستقبل مهما اختلف مقياس النمط

run = st.button("🚀 ابدأ البحث الشامل العميق")

# ---------------------------------------------------------------------------
# 1) استخراج النمط من الصورة
# ---------------------------------------------------------------------------
def extract_pattern_from_image(file_bytes: np.ndarray):
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    crop = img[int(h * 0.06):int(h * 0.94), int(w * 0.02):int(w * 0.90)]
    crop = cv2.GaussianBlur(crop, (3, 3), 0)

    edges1 = cv2.Canny(crop, 25, 90)
    edges2 = cv2.Canny(crop, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return None

    x_unique = np.unique(xs)
    y_profile = np.array([-np.mean(ys[xs == x]) for x in x_unique], dtype=np.float64)
    y_profile = gaussian_filter1d(y_profile, sigma=0.6)

    x_old = np.linspace(0, 1, len(y_profile))
    x_new = np.linspace(0, 1, BASE_PATTERN_LEN)
    y_resampled = np.interp(x_new, x_old, y_profile)

    pattern = (y_resampled - np.mean(y_resampled)) / (np.std(y_resampled) + 1e-8)
    return pattern


def resample_pattern(pattern: np.ndarray, scale: float) -> np.ndarray:
    """يعيد بناء النمط بطول مختلف لمحاكاة أن نفس الحركة قد تحدث تاريخيًا أسرع أو أبطأ."""
    new_len = max(20, int(round(len(pattern) * scale)))
    x_old = np.linspace(0, 1, len(pattern))
    x_new = np.linspace(0, 1, new_len)
    resampled = np.interp(x_new, x_old, pattern)
    resampled = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-8)
    return resampled


# ---------------------------------------------------------------------------
# 2) جلب البيانات
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(ticker: str, period: str, interval: str):
    return yf.download(tickers=ticker, period=period, interval=interval, progress=False)


def get_close_series(df: pd.DataFrame) -> np.ndarray:
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        close = df["Close"]
    return close.values.astype(np.float64).flatten()


PERIOD_MAP = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}


# ---------------------------------------------------------------------------
# 3) مسح متجه شامل (بلا تخطي أي نافذة) لكل تركيبة (مصدر × مقياس)
# ---------------------------------------------------------------------------
def vectorized_prescreen(prices: np.ndarray, pattern: np.ndarray):
    w = len(pattern)
    if len(prices) <= w + 1:
        return None

    windows = sliding_window_view(prices, w)
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True) + 1e-8
    norm_windows = (windows - means) / stds

    corr = (norm_windows @ pattern) / w
    shape_error = np.mean(np.abs(norm_windows - pattern), axis=1)

    pattern_diff = np.diff(pattern)
    windows_diff = np.diff(norm_windows, axis=1)
    velocity_error = np.mean(np.abs(windows_diff - pattern_diff), axis=1)

    user_min_idx, user_max_idx = int(np.argmin(pattern)), int(np.argmax(pattern))
    hist_min_idx = np.argmin(norm_windows, axis=1)
    hist_max_idx = np.argmax(norm_windows, axis=1)
    extrema_penalty = (np.abs(hist_min_idx - user_min_idx) + np.abs(hist_max_idx - user_max_idx)) / w

    composite = shape_error + 2.0 * velocity_error + 1.0 * extrema_penalty - 1.5 * corr

    return {"composite": composite, "corr": corr, "norm_windows": norm_windows}


# ---------------------------------------------------------------------------
# 4) تنقيح دقيق بـ DTW على أفضل المرشحين فقط
# ---------------------------------------------------------------------------
def dtw_distance(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def intervals_overlap(a_start, a_end, b_start, b_end, max_ratio=0.3):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    shorter = max(1, min(a_end - a_start, b_end - b_start))
    return (inter / shorter) > max_ratio


# ---------------------------------------------------------------------------
# التشغيل الرئيسي
# ---------------------------------------------------------------------------
if uploaded_file and run:
    t_start = time.time()

    with st.spinner("جاري استخراج النمط من الصورة..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        base_pattern = extract_pattern_from_image(file_bytes)

    if base_pattern is None:
        st.error("لم يتم التعرف على حركة الشموع. يرجى رفع صورة أوضح للشارت.")
        st.stop()

    all_candidates = []
    period = PERIOD_MAP[timeframe]
    total_steps = len(selected_tickers) * len(SCALE_FACTORS)
    progress = st.progress(0.0, text="جاري تحميل البيانات وتحليلها...")
    step = 0

    price_cache = {}
    for ticker in selected_tickers:
        data = fetch_data(ticker, period, timeframe)
        if data is None or data.empty:
            st.warning(f"تعذر جلب بيانات المصدر {ticker} — سيتم تجاهله.")
            continue
        price_cache[ticker] = (get_close_series(data), data.index)

    for ticker, (prices, timestamps) in price_cache.items():
        for scale in SCALE_FACTORS:
            step += 1
            progress.progress(step / max(1, total_steps),
                               text=f"مسح {ticker} بمقياس زمني ×{scale:.2f} ...")

            scaled_pattern = resample_pattern(base_pattern, scale)
            result = vectorized_prescreen(prices, scaled_pattern)
            if result is None:
                continue

            composite = result["composite"]
            order = np.argsort(composite)
            top_idx_for_dtw = order[:DTW_CANDIDATE_POOL]

            for idx in top_idx_for_dtw:
                idx = int(idx)
                hist_window = result["norm_windows"][idx]
                dtw_dist = dtw_distance(scaled_pattern, hist_window, DTW_RADIUS)
                all_candidates.append({
                    "ticker": ticker,
                    "idx": idx,
                    "w": len(scaled_pattern),
                    "scale": scale,
                    "timestamps": timestamps,
                    "prices": prices,
                    "composite": float(composite[idx]),
                    "corr": float(result["corr"][idx]),
                    "dtw": dtw_dist,
                })

    progress.empty()
    elapsed = time.time() - t_start

    if not all_candidates:
        st.error("لم يتم إيجاد أي تطابق صالح في أي من المصادر/المقاييس المختارة.")
        st.stop()

    # ترتيب نهائي: دمج رتبة الشكل (composite) ورتبة DTW
    composites = np.array([c["composite"] for c in all_candidates])
    dtws = np.array([c["dtw"] / c["w"] for c in all_candidates])  # تطبيع DTW بطول النافذة
    rank_composite = np.argsort(np.argsort(composites))
    rank_dtw = np.argsort(np.argsort(dtws))
    final_rank_score = rank_composite + rank_dtw
    order_final = np.argsort(final_rank_score)

    seen_per_ticker = {}
    final_selection = []
    for oi in order_final:
        cand = all_candidates[int(oi)]
        key = cand["ticker"]
        seen_per_ticker.setdefault(key, [])
        c_start, c_end = cand["idx"], cand["idx"] + cand["w"]
        if not any(intervals_overlap(c_start, c_end, s0, s1) for (s0, s1) in seen_per_ticker[key]):
            seen_per_ticker[key].append((c_start, c_end))
            final_selection.append(cand)
        if len(final_selection) >= top_k:
            break

    if not final_selection:
        st.error("تعذر استخلاص عدد كافٍ من الحالات المستقلة.")
        st.stop()

    st.caption(f"⏱️ اكتمل البحث في {elapsed:.1f} ثانية — تم فحص {len(all_candidates):,} مرشح "
               f"عبر {len(price_cache)} مصدر و {len(SCALE_FACTORS)} مقاييس زمنية مختلفة.")

    best = final_selection[0]
    match_percentage = float(np.clip(50 + best["corr"] * 50, 50, 99.5))
    w_best = best["w"]
    start_t = best["timestamps"][best["idx"]].strftime("%Y-%m-%d %H:%M")
    end_t = best["timestamps"][best["idx"] + w_best].strftime("%Y-%m-%d %H:%M")

    st.success(
        f"🔥 أفضل تطابق: نسبة تشابه شكلي **{match_percentage:.1f}%** "
        f"(المصدر: {best['ticker']}, مقياس زمني ×{best['scale']:.2f})"
    )
    st.info(f"📅 الفترة التاريخية لأفضل تطابق: من **{start_t}** إلى **{end_t}**")

    # ---------------- بناء جدول النتائج + مسارات المستقبل الموحّدة ----------------
    rows = []
    normalized_future_paths = []
    weights = []

    for rank, cand in enumerate(final_selection, start=1):
        idx, w_c, prices, ts = cand["idx"], cand["w"], cand["prices"], cand["timestamps"]
        future_len_c = max(2, w_c // 2)
        hist_window = prices[idx: idx + w_c]
        future_window = prices[idx + w_c: idx + w_c + future_len_c]
        if len(future_window) < 2:
            continue

        last_hist_price = hist_window[-1]
        pct_path = (future_window / last_hist_price - 1.0) * 100.0

        # توحيد طول المسار زمنيًا (نسبة مئوية من طول النمط) حتى تتجمّع المقاييس المختلفة بشكل عادل
        x_old = np.linspace(0, 1, len(pct_path))
        x_new = np.linspace(0, 1, COMMON_HORIZON_POINTS)
        pct_path_norm = np.interp(x_new, x_old, pct_path)
        normalized_future_paths.append(pct_path_norm)

        # وزن الحالة: كلما كانت رتبتها النهائية أفضل (composite + DTW) زاد وزنها في التوقع
        weight = 1.0 / (1.0 + rank)
        weights.append(weight)

        rows.append({
            "الترتيب": rank,
            "المصدر": cand["ticker"],
            "المقياس الزمني": f"×{cand['scale']:.2f}",
            "من": ts[idx].strftime("%Y-%m-%d %H:%M"),
            "إلى": ts[idx + w_c].strftime("%Y-%m-%d %H:%M"),
            "ارتباط الشكل": round(cand["corr"], 3),
            "DTW (مطبّع)": round(cand["dtw"] / w_c, 3),
            "أقصى تغيّر لاحق %": round(float(pct_path[-1]), 2),
        })

    st.subheader("📊 أفضل الحالات المستقلة المستخدمة في بناء التوقع")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if len(normalized_future_paths) >= 2:
        matrix = np.array(normalized_future_paths)          # (k, COMMON_HORIZON_POINTS)
        w_arr = np.array(weights)
        w_arr = w_arr / w_arr.sum()

        weighted_mean_path = np.average(matrix, axis=0, weights=w_arr)
        weighted_var_path = np.average((matrix - weighted_mean_path) ** 2, axis=0, weights=w_arr)
        weighted_std_path = np.sqrt(weighted_var_path)

        # مقياس "قوة/اتساق الإشارة": كلما قل التشتت الموزون نسبة لحجم الحركة المتوقعة زادت الثقة النسبية
        signal_consistency = float(np.clip(
            100 * (1 - (weighted_std_path.mean() / (np.abs(weighted_mean_path).mean() + 1e-6 + weighted_std_path.mean()))),
            0, 100
        ))

        st.metric("🎯 مؤشر اتساق الحالات التاريخية (ليس احتمال ربح)", f"{signal_consistency:.1f}%")
        st.caption(
            "هذا المؤشر يقيس فقط مدى تقارب النتائج فيما بينها بعد الحالات المشابهة تاريخيًا — "
            "ارتفاعه يعني أن الحالات المشابهة تصرّفت بشكل متقارب، وانخفاضه يعني تشتتًا كبيرًا "
            "(أي أن الاعتماد على النمط وحده غير موثوق في هذه الحالة)."
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        hist_slice = best["prices"][best["idx"]: best["idx"] + w_best]
        ax.plot(range(w_best), hist_slice,
                label=f"أفضل نمط مطابق ({best['ticker']}, ×{best['scale']:.2f})",
                color="#00f2fe", linewidth=2.2)

        base_price = hist_slice[-1]
        x_future = np.linspace(w_best - 1, w_best - 1 + w_best // 2, COMMON_HORIZON_POINTS)

        mean_price_path = base_price * (1 + weighted_mean_path / 100.0)
        upper_price_path = base_price * (1 + (weighted_mean_path + weighted_std_path) / 100.0)
        lower_price_path = base_price * (1 + (weighted_mean_path - weighted_std_path) / 100.0)

        ax.plot(x_future, mean_price_path, color="#1e90ff", linewidth=3.2,
                label=f"المسار المرجّح الأقوى (توقّع مبني على {len(final_selection)} حالة مرجّحة)")
        ax.fill_between(x_future, lower_price_path, upper_price_path,
                         color="#1e90ff", alpha=0.18, label="نطاق التذبذب التاريخي حول التوقع")

        ax.axvline(x=w_best - 1, color="#ffd700", linestyle="--", alpha=0.9,
                   label="نقطة نهاية النمط الحالي")

        ax.set_facecolor("#131722")
        fig.patch.set_facecolor("#131722")
        ax.tick_params(colors="white")
        ax.grid(True, color="#2a2e39", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left", fontsize=9)
        ax.set_title("النمط المطابق + المسار المرجّح الأقوى (بحث متعدد المقاييس والمصادر)", color="white")

        st.pyplot(fig)
    else:
        st.warning("عدد الحالات الصالحة لبناء توقع مرجّح غير كافٍ.")
