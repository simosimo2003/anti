import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.lib.stride_tricks import sliding_window_view

st.set_page_config(page_title="محرك مطابقة الأنماط - إصدار احترافي", layout="wide")

st.title("🪙 محرك مطابقة الأنماط التاريخية للذهب - إصدار متقدم")
st.caption(
    "بحث متعدد المصادر • مطابقة هجينة (ارتباط بيرسون + DTW + تحليل السرعة) "
    "• استبعاد التداخل بين النتائج • توقع احتمالي بمجموعة (Ensemble) بدل خط واحد وهمي الدقة"
)

st.warning(
    "⚠️ تنويه هام: هذه الأداة أداة استكشافية لمطابقة الشكل البصري فقط. "
    "تشابه نمط سعري في الماضي لا يعني تكرار نفس الحركة مستقبلاً؛ الأسواق المالية "
    "تتأثر بعوامل كثيرة لا يلتقطها الشكل البصري وحده. النتائج هنا إحصائية/استكشافية "
    "وليست توصية استثمارية أو ضمانًا لأي حركة سعرية."
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
    top_k = st.slider("عدد أفضل النتائج المستقلة (Ensemble)", 3, 30, 10)

TICKER_MAP = {
    "GC=F — عقود الذهب الأمريكية (COMEX)": "GC=F",
    "MGC=F — عقود الذهب المصغرة": "MGC=F",
    "XAUUSD=X — سعر السبوت (Spot)": "XAUUSD=X",
}
sources_selected = st.multiselect(
    "مصادر بيانات الذهب (اختيار أكثر من مصدر يعمّق البحث ويقلل الاعتماد على مصدر واحد)",
    list(TICKER_MAP.keys()),
    default=["GC=F — عقود الذهب الأمريكية (COMEX)", "XAUUSD=X — سعر السبوت (Spot)"],
)
selected_tickers = [TICKER_MAP[s] for s in sources_selected] or ["GC=F"]

PATTERN_LEN = 120   # عدد النقاط الموحّد الذي يُعاد أخذ العينة إليه (يجعل المطابقة مستقلة عن دقة الصورة)
DTW_RADIUS = 12      # نافذة Sakoe-Chiba لتسريع DTW
DTW_CANDIDATE_POOL = 60  # عدد أفضل المرشحين (لكل مصدر) الذين يمرّون لمرحلة DTW الدقيقة

run = st.button("🚀 بدء البحث الشامل العميق")

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

    # دمج نتائج أكثر من عتبة Canny لالتقاط الخط حتى لو كان باهتًا أو سميكًا
    edges1 = cv2.Canny(crop, 25, 90)
    edges2 = cv2.Canny(crop, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return None

    x_unique = np.unique(xs)
    y_profile = np.array([-np.mean(ys[xs == x]) for x in x_unique], dtype=np.float64)
    y_profile = gaussian_filter1d(y_profile, sigma=0.6)

    # إعادة أخذ العينات لعدد نقاط ثابت -> النمط يصبح مستقلاً عن دقة/عرض الصورة المرفوعة
    x_old = np.linspace(0, 1, len(y_profile))
    x_new = np.linspace(0, 1, PATTERN_LEN)
    y_resampled = np.interp(x_new, x_old, y_profile)

    pattern = (y_resampled - np.mean(y_resampled)) / (np.std(y_resampled) + 1e-8)
    return pattern


# ---------------------------------------------------------------------------
# 2) جلب البيانات (مع تخزين مؤقت) من كل مصدر
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_data(ticker: str, period: str, interval: str):
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    return df


def get_close_series(df: pd.DataFrame) -> np.ndarray:
    """يتعامل مع أعمدة عادية أو MultiIndex التي قد تعيدها yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
        close = close.iloc[:, 0]
    else:
        close = df["Close"]
    return close.values.astype(np.float64).flatten()


PERIOD_MAP = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "max"}


# ---------------------------------------------------------------------------
# 3) المرحلة الأولى: مسح متجه (Vectorized) شامل شمعة بشمعة، بلا أي تخطي
# ---------------------------------------------------------------------------
def vectorized_prescreen(prices: np.ndarray, pattern: np.ndarray):
    w = len(pattern)
    if len(prices) <= w + 1:
        return None

    windows = sliding_window_view(prices, w)          # (n-w+1, w)
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True) + 1e-8
    norm_windows = (windows - means) / stds

    # ارتباط بيرسون لكل نافذة (كلا الطرفين موحّدان القياس، فالجداء النقطي/العرض = الارتباط)
    corr = (norm_windows @ pattern) / w

    shape_error = np.mean(np.abs(norm_windows - pattern), axis=1)

    pattern_diff = np.diff(pattern)
    windows_diff = np.diff(norm_windows, axis=1)
    velocity_error = np.mean(np.abs(windows_diff - pattern_diff), axis=1)

    user_min_idx, user_max_idx = int(np.argmin(pattern)), int(np.argmax(pattern))
    hist_min_idx = np.argmin(norm_windows, axis=1)
    hist_max_idx = np.argmax(norm_windows, axis=1)
    extrema_penalty = (np.abs(hist_min_idx - user_min_idx) + np.abs(hist_max_idx - user_max_idx)) / w

    # نتيجة مركّبة: كل مركّبات الخطأ تُخفَّض، والارتباط الجيد يُخفِّض النتيجة أيضًا
    composite = shape_error + 2.0 * velocity_error + 1.0 * extrema_penalty - 1.5 * corr

    return {
        "composite": composite,
        "corr": corr,
        "shape_error": shape_error,
        "norm_windows": norm_windows,
    }


# ---------------------------------------------------------------------------
# 4) المرحلة الثانية: تنقيح دقيق بخوارزمية DTW (Dynamic Time Warping)
#    تُطبَّق فقط على أفضل المرشحين من المرحلة الأولى -> سريعة وليست بطيئة
# ---------------------------------------------------------------------------
def dtw_distance(a: np.ndarray, b: np.ndarray, radius: int = DTW_RADIUS) -> float:
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


def select_diverse_top_k(order: np.ndarray, min_gap: int, k: int):
    """يستبعد النتائج شديدة التداخل زمنيًا حتى لا تهيمن نافذة واحدة قريبة من نفسها على القائمة."""
    selected = []
    for idx in order:
        if all(abs(int(idx) - s) >= min_gap for s in selected):
            selected.append(int(idx))
        if len(selected) >= k:
            break
    return selected


# ---------------------------------------------------------------------------
# التشغيل الرئيسي
# ---------------------------------------------------------------------------
if uploaded_file and run:
    with st.spinner("جاري استخراج النمط من الصورة..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        user_pattern = extract_pattern_from_image(file_bytes)

    if user_pattern is None:
        st.error("لم يتم التعرف على حركة الشموع. يرجى رفع صورة أوضح للشارت.")
        st.stop()

    all_candidates = []  # كل عنصر: dict(ticker, idx, prices, composite, corr)
    period = PERIOD_MAP[timeframe]

    progress = st.progress(0.0, text="جاري تحميل البيانات وتحليلها من كل مصدر...")
    for i, ticker in enumerate(selected_tickers):
        data = fetch_data(ticker, period, timeframe)
        if data is None or data.empty:
            st.warning(f"تعذر جلب بيانات المصدر {ticker} — سيتم تجاهله.")
            continue

        prices = get_close_series(data)
        timestamps = data.index

        result = vectorized_prescreen(prices, user_pattern)
        if result is None:
            st.warning(f"بيانات المصدر {ticker} قصيرة جدًا لطول النمط المستخرج — سيتم تجاهله.")
            continue

        composite = result["composite"]
        order = np.argsort(composite)  # الأفضل أولاً (أصغر قيمة)
        top_idx_for_dtw = order[:DTW_CANDIDATE_POOL]

        for idx in top_idx_for_dtw:
            idx = int(idx)
            hist_window = result["norm_windows"][idx]
            dtw_dist = dtw_distance(user_pattern, hist_window)
            all_candidates.append({
                "ticker": ticker,
                "idx": idx,
                "timestamps": timestamps,
                "prices": prices,
                "composite": float(composite[idx]),
                "corr": float(result["corr"][idx]),
                "dtw": dtw_dist,
            })
        progress.progress((i + 1) / len(selected_tickers))

    progress.empty()

    if not all_candidates:
        st.error("لم يتم إيجاد أي تطابق صالح في أي من المصادر المختارة.")
        st.stop()

    # ترتيب نهائي يجمع بين رتبة المطابقة الشكلية (composite) ورتبة DTW
    composites = np.array([c["composite"] for c in all_candidates])
    dtws = np.array([c["dtw"] for c in all_candidates])
    rank_composite = np.argsort(np.argsort(composites))
    rank_dtw = np.argsort(np.argsort(dtws))
    final_rank_score = rank_composite + rank_dtw
    order_final = np.argsort(final_rank_score)

    w = len(user_pattern)
    seen_per_ticker = {}
    final_selection = []
    for oi in order_final:
        cand = all_candidates[int(oi)]
        key = cand["ticker"]
        seen_per_ticker.setdefault(key, [])
        # استبعاد النوافذ شديدة التداخل زمنيًا لنفس المصدر
        if all(abs(cand["idx"] - s) >= w // 2 for s in seen_per_ticker[key]):
            seen_per_ticker[key].append(cand["idx"])
            final_selection.append(cand)
        if len(final_selection) >= top_k:
            break

    if not final_selection:
        st.error("تعذر استخلاص نتائج مستقلة كافية.")
        st.stop()

    # ---------------- عرض النتائج ----------------
    best = final_selection[0]
    match_percentage = float(np.clip(50 + best["corr"] * 50, 50, 99.5))

    start_t = best["timestamps"][best["idx"]].strftime("%Y-%m-%d %H:%M")
    end_t = best["timestamps"][best["idx"] + w].strftime("%Y-%m-%d %H:%M")

    st.success(f"🔥 أفضل تطابق مستقل: نسبة تشابه تقريبية **{match_percentage:.1f}%** (المصدر: {best['ticker']})")
    st.info(f"📅 الفترة التاريخية لأفضل تطابق: من **{start_t}** إلى **{end_t}**")

    # جدول بأفضل النتائج المستقلة
    rows = []
    future_len = w // 2
    ensemble_returns = []

    for rank, cand in enumerate(final_selection, start=1):
        idx, prices, ts = cand["idx"], cand["prices"], cand["timestamps"]
        hist_window = prices[idx: idx + w]
        future_window = prices[idx + w: idx + w + future_len]
        if len(future_window) < 2:
            continue
        last_hist_price = hist_window[-1]
        pct_path = (future_window / last_hist_price - 1.0) * 100.0
        ensemble_returns.append(pct_path[:future_len])

        rows.append({
            "الترتيب": rank,
            "المصدر": cand["ticker"],
            "من": ts[idx].strftime("%Y-%m-%d %H:%M"),
            "إلى": ts[idx + w].strftime("%Y-%m-%d %H:%M"),
            "ارتباط الشكل": round(cand["corr"], 3),
            "مسافة DTW": round(cand["dtw"], 2),
            "أقصى تغيّر لاحق %": round(float(pct_path[-1]), 2) if len(pct_path) else None,
        })

    st.subheader("📊 أفضل النتائج المستقلة (غير متداخلة زمنيًا)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---------------- توقع جماعي (Ensemble) بدل خط واحد وهمي الدقة ----------------
    min_len = min(len(r) for r in ensemble_returns) if ensemble_returns else 0
    if min_len >= 2:
        matrix = np.array([r[:min_len] for r in ensemble_returns])
        mean_path = matrix.mean(axis=0)
        std_path = matrix.std(axis=0)

        fig, ax = plt.subplots(figsize=(12, 6))

        scaled_user_pattern = (user_pattern * np.std(best["prices"][best["idx"]:best["idx"] + w])
                                + np.mean(best["prices"][best["idx"]:best["idx"] + w]))
        ax.plot(range(w), best["prices"][best["idx"]:best["idx"] + w],
                label=f"أفضل نمط مطابق ({best['ticker']})", color="#00f2fe", linewidth=2.2)

        x_future = np.arange(w - 1, w - 1 + min_len + 1)
        base_price = best["prices"][best["idx"] + w - 1]
        mean_price_path = base_price * (1 + np.insert(mean_path, 0, 0) / 100.0)
        upper_price_path = base_price * (1 + np.insert(mean_path + std_path, 0, 0) / 100.0)
        lower_price_path = base_price * (1 + np.insert(mean_path - std_path, 0, 0) / 100.0)

        ax.plot(x_future, mean_price_path, color="#1e90ff", linewidth=2.5,
                label=f"متوسط المسار اللاحق (Ensemble لعدد {len(ensemble_returns)} حالة)")
        ax.fill_between(x_future, lower_price_path, upper_price_path,
                         color="#1e90ff", alpha=0.2, label="نطاق التذبذب التاريخي (±1 انحراف معياري)")

        ax.axvline(x=w - 1, color="#ffd700", linestyle="--", alpha=0.9, label="نقطة نهاية النمط الحالي")

        ax.set_facecolor("#131722")
        fig.patch.set_facecolor("#131722")
        ax.tick_params(colors="white")
        ax.grid(True, color="#2a2e39", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="white", loc="upper left", fontsize=9)
        ax.set_title("النمط المطابق + التوقع الاحتمالي الجماعي (وليس خطًا حتميًا واحدًا)", color="white")

        st.pyplot(fig)

        st.caption(
            "المسار الأزرق هو متوسط ما حدث تاريخيًا بعد أفضل "
            f"{len(ensemble_returns)} حالة مشابهة، والمنطقة الشفافة تمثل مدى التذبذب "
            "بين تلك الحالات — وليس تنبؤًا مضمونًا. كلما اتسع النطاق كان تشتت النتائج التاريخية أكبر "
            "وبالتالي موثوقية الإشارة أقل."
        )
    else:
        st.warning("لا توجد بيانات مستقبلية كافية بعد النافذة المطابقة لبناء توقع جماعي.")
