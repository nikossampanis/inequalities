
import re
import random
import io
import datetime
import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# =========================================================
# Core math engine (SymPy)
# =========================================================
x = sp.Symbol('x', real=True)

def normalize_input(s: str) -> str:
    s = s.strip()
    s = s.replace("^", "**")
    s = re.sub(r"\babs\s*\(", "Abs(", s, flags=re.IGNORECASE)
    return s

def parse_inequality(line: str):
    line = normalize_input(line)
    ops = ["<=", ">=", "<", ">"]
    op_found = None
    for op in ops:
        if op in line:
            op_found = op
            break
    if not op_found:
        raise ValueError("Δεν βρέθηκε τελεστής ανισότητας (<, <=, >, >=).")

    left, right = line.split(op_found, 1)
    local_dict = {"x": x, "Abs": sp.Abs}
    L = sp.sympify(left.strip(), locals=local_dict)
    R = sp.sympify(right.strip(), locals=local_dict)

    return {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op_found](L, R)

def solve_ineq(rel):
    sol = sp.solve_univariate_inequality(rel, x, relational=False)
    sol = sp.Intersection(sol, sp.S.Reals)
    return sp.simplify(sol)

def endpoint_explanation(sol_set):
    intervals = []
    if isinstance(sol_set, sp.Interval):
        intervals = [sol_set]
    elif isinstance(sol_set, sp.Union):
        intervals = [arg for arg in sol_set.args if isinstance(arg, sp.Interval)]
    if not intervals:
        return None

    def fmt(v):
        if v is sp.S.NegativeInfinity: return "-∞"
        if v is sp.S.Infinity: return "∞"
        return sp.pretty(v)

    lines = []
    for I in intervals:
        a, b = I.start, I.end
        left_closed = (I.left_open is False)
        right_closed = (I.right_open is False)
        left_symbol = "[" if left_closed else "("
        right_symbol = "]" if right_closed else ")"
        lines.append(
            f"{left_symbol}{fmt(a)}, {fmt(b)}{right_symbol} "
            f"(αριστερό: {'κλειστό' if left_closed else 'ανοικτό'}, "
            f"δεξί: {'κλειστό' if right_closed else 'ανοικτό'})"
        )
    return lines

def plot_number_line(sol_set, xmin=-10, xmax=10, title="Αριθμητική ευθεία λύσεων"):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.hlines(0, xmin, xmax, linewidth=2)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(list(range(int(xmin), int(xmax) + 1, max(1, int((xmax-xmin)/10) or 1))))
    ax.set_title(title, pad=10)
    ax.spines[['left', 'right', 'top']].set_visible(False)

    intervals = []
    if sol_set is sp.S.EmptySet:
        intervals = []
    elif sol_set is sp.S.Reals:
        intervals = [sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)]
    elif isinstance(sol_set, sp.Interval):
        intervals = [sol_set]
    elif isinstance(sol_set, sp.Union):
        intervals = [arg for arg in sol_set.args if isinstance(arg, sp.Interval)]

    def clamp(v):
        if v is sp.S.NegativeInfinity: return xmin
        if v is sp.S.Infinity: return xmax
        v = float(v)
        return max(xmin, min(xmax, v))

    for I in intervals:
        a, b = I.start, I.end
        aa, bb = clamp(a), clamp(b)
        ax.hlines(0, aa, bb, linewidth=8, alpha=0.6)

        if a not in (sp.S.NegativeInfinity, sp.S.Infinity):
            fa = float(a)
            if xmin <= fa <= xmax:
                if I.left_open:
                    ax.plot(fa, 0, marker='o', markersize=9, fillstyle='none')
                else:
                    ax.plot(fa, 0, marker='o', markersize=9)

        if b not in (sp.S.NegativeInfinity, sp.S.Infinity):
            fb = float(b)
            if xmin <= fb <= xmax:
                if I.right_open:
                    ax.plot(fb, 0, marker='o', markersize=9, fillstyle='none')
                else:
                    ax.plot(fb, 0, marker='o', markersize=9)

    return fig

# =========================================================
# Exercise bank (A' Lykeio-friendly)
# =========================================================
EXERCISES = [
    {"id":"A1","topic":"Γραμμική","prompt":"Λύσε την ανίσωση:  2x - 3 ≤ 5","ineq":"2*x - 3 <= 5",
     "hint":"Μεταφέρεις σταθερούς όρους, διαιρείς με θετικό αριθμό."},
    {"id":"A2","topic":"Γραμμική","prompt":"Λύσε την ανίσωση:  -3x + 6 > 0","ineq":"-3*x + 6 > 0",
     "hint":"Όταν διαιρείς με αρνητικό, αλλάζει η φορά της ανισότητας."},
    {"id":"B1","topic":"Τετραγωνική","prompt":"Λύσε την ανίσωση:  x² - 5x + 6 ≥ 0","ineq":"x^2 - 5*x + 6 >= 0",
     "hint":"Βρες ρίζες, μετά πίνακα προσήμων."},
    {"id":"B2","topic":"Τετραγωνική","prompt":"Λύσε την ανίσωση:  x² - 9 < 0","ineq":"x^2 - 9 < 0",
     "hint":"x² - 9 = (x-3)(x+3)."},
    {"id":"C1","topic":"Ρητή","prompt":"Λύσε την ανίσωση:  (x - 1)/(x + 2) ≥ 0","ineq":"(x-1)/(x+2) >= 0",
     "hint":"Κρίσιμα σημεία: x=1, x=-2 (όμως x≠-2)."},
    {"id":"D1","topic":"Απόλυτη","prompt":"Λύσε την ανίσωση:  |x - 3| ≤ 5","ineq":"Abs(x-3) <= 5",
     "hint":"|A| ≤ k ⇔ -k ≤ A ≤ k (k≥0)."},
    {"id":"D2","topic":"Απόλυτη","prompt":"Λύσε την ανίσωση:  |2x + 1| > 3","ineq":"Abs(2*x+1) > 3",
     "hint":"|A| > k ⇔ A>k ή A<-k (k≥0)."},
]

def pick_random_exercise(topic_filter="Όλα"):
    pool = EXERCISES if topic_filter == "Όλα" else [e for e in EXERCISES if e["topic"] == topic_filter]
    return random.choice(pool)

# =========================================================
# PDF Export
# =========================================================
def make_pdf_report(exercise, user_text, sol_set_str, endpoint_lines, fig_png_bytes, score, streak):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    y = H - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Δραστηριότητα: Ανισώσεις Α’ Λυκείου")
    y -= 22
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Ημερομηνία: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y -= 16
    c.drawString(40, y, "Developed by Nikolaos Sampanis")
    y -= 18

    c.setLineWidth(1)
    c.line(40, y, W-40, y)
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Άσκηση ({exercise['topic']}): {exercise['id']}")
    y -= 18

    c.setFont("Helvetica", 11)
    text = c.beginText(40, y)
    text.setLeading(14)
    text.textLine(exercise["prompt"])
    text.textLine(f"Ανίσωση: {exercise['ineq']}")
    text.textLine("")
    text.textLine("Απάντηση μαθητή:")
    for line in (user_text.strip() or "(κενό)").splitlines():
        text.textLine(line)
    text.textLine("")
    text.textLine("Ορθή λύση (σύνολο):")
    text.textLine(sol_set_str)
    if endpoint_lines:
        text.textLine("")
        text.textLine("Ανοικτό/κλειστό:")
        for ln in endpoint_lines:
            text.textLine(f"- {ln}")
    c.drawText(text)

    if fig_png_bytes:
        img = ImageReader(io.BytesIO(fig_png_bytes))
        img_w = W - 80
        img_h = 140
        c.drawImage(img, 40, 120, width=img_w, height=img_h, preserveAspectRatio=True, mask='auto')

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, 90, f"Score: {score}   |   Streak: {streak}")

    c.setFont("Helvetica", 9)
    c.setFillGray(0.35)
    c.drawRightString(W-40, 30, "Streamlit classroom activity • Inequalities Quest")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# =========================================================
# Streamlit UI (gaming vibe)
# =========================================================
st.set_page_config(page_title="Inequalities Quest", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem;}
.hero {
  border-radius: 22px;
  padding: 18px 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.12);
}
.card {
  border-radius: 18px;
  padding: 14px 16px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
}
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18); margin-right: 8px;
  font-size: 0.9rem; opacity:0.95;
}
.small {opacity: 0.85; font-size: 0.95rem;}
hr {opacity:0.25;}
</style>
""", unsafe_allow_html=True)

if "exercise" not in st.session_state:
    st.session_state.exercise = pick_random_exercise()
if "score" not in st.session_state:
    st.session_state.score = 0
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "topic_filter" not in st.session_state:
    st.session_state.topic_filter = "Όλα"
if "last_solution" not in st.session_state:
    st.session_state.last_solution = None

st.markdown('<div class="hero">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.2, 0.8, 1.0])
with c1:
    st.markdown("## 🎮 Inequalities Quest")
    st.markdown('<span class="badge">A’ Λυκείου</span><span class="badge">Random</span><span class="badge">PDF</span>', unsafe_allow_html=True)
with c2:
    st.metric("⭐ Score", st.session_state.score)
with c3:
    st.metric("🔥 Streak", st.session_state.streak)
st.markdown('<div class="small">Developed by <b>Nikolaos Sampanis</b></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["🧩 Δραστηριότητα", "🔍 Εξερεύνηση", "📘 Θεωρία"])

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    t1, t2 = st.columns([0.7, 0.3])
    with t1:
        topics = ["Όλα"] + sorted({e["topic"] for e in EXERCISES})
        st.session_state.topic_filter = st.selectbox("Φίλτρο θεματικής", topics, index=topics.index(st.session_state.topic_filter))
    with t2:
        if st.button("🔁 Νέα random άσκηση", use_container_width=True):
            st.session_state.exercise = pick_random_exercise(st.session_state.topic_filter)
            st.session_state.last_solution = None

    ex = st.session_state.exercise
    st.markdown(f"### 🏁 Mission: **{ex['prompt']}**")
    st.markdown(f"**Κωδικός:** `{ex['id']}`   ·   **Θεματική:** `{ex['topic']}`")
    with st.expander("💡 Hint", expanded=False):
        st.write(ex["hint"])

    st.markdown("#### ✍️ Γράψε τη λύση σου (σε διαστήματα)")
    st.caption("Παράδειγμα: (-∞,2] U (5,∞)  ή  [ -2, 3 )  ή  ∅  ή  R")
    user_answer = st.text_area("Απάντηση μαθητή", height=90, placeholder="Γράψε τη λύση σου εδώ...")

    solve_col, reveal_col, pdf_col = st.columns([0.34, 0.34, 0.32])
    do_check = solve_col.button("✅ Έλεγχος", use_container_width=True)
    reveal = reveal_col.toggle("👁️ Εμφάνιση ορθής λύσης", value=False)
    export_pdf = pdf_col.button("📄 Εξαγωγή PDF", use_container_width=True)

    def compute_solution():
        rel = parse_inequality(ex["ineq"])
        sol = solve_ineq(rel)
        return rel, sol

    def parse_student_set(s: str):
        s = (s or "").strip()
        if not s:
            return None
        s = s.replace("∪", "U")
        if s in ["∅", "EmptySet"]:
            return sp.S.EmptySet
        if s in ["R", "Reals", "ℝ"]:
            return sp.S.Reals
        s = s.replace("∞", "oo").replace(" ", "")
        parts = s.split("U")
        sets = []
        for p in parts:
            m = re.match(r"^([\(\[])([^,]+),([^)\]]+)([\)\]])$", p)
            if not m:
                return None
            lbr, a, b, rbr = m.groups()
            a = sp.sympify(a, locals={"oo": sp.oo})
            b = sp.sympify(b, locals={"oo": sp.oo})
            left_open = (lbr == "(")
            right_open = (rbr == ")")
            sets.append(sp.Interval(a, b, left_open=left_open, right_open=right_open))
        out = sets[0]
        for ss in sets[1:]:
            out = sp.Union(out, ss)
        return sp.simplify(sp.Intersection(out, sp.S.Reals))

    if do_check or reveal or export_pdf:
        try:
            rel, sol = compute_solution()
            sol_str = str(sol)
            xmin, xmax = st.slider("Εύρος ευθείας για την άσκηση", -50, 50, (-10, 10), key="activity_range")
            fig = plot_number_line(sol, xmin=xmin, xmax=xmax, title="Λύση στην αριθμητική ευθεία")
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            png_bytes = png_buf.getvalue()

            endpoint_lines = endpoint_explanation(sol)

            student_set = parse_student_set(user_answer)
            correct = None if student_set is None else (sp.simplify(student_set) == sp.simplify(sol))

            if do_check:
                if correct is True:
                    st.success("✅ Σωστό! +10 πόντοι")
                    st.session_state.score += 10
                    st.session_state.streak += 1
                elif correct is False:
                    st.error("❌ Όχι ακριβώς. Ξαναδοκίμασε!")
                    st.session_state.streak = 0
                else:
                    st.warning("ℹ️ Δεν κατάλαβα τη μορφή. Δοκίμασε: (-∞,2] U (5,∞) ή [-2,3) ή ∅ ή R")

            st.markdown("#### 📊 Οπτικοποίηση")
            st.image(png_bytes, caption="Αριθμητική ευθεία λύσεων", use_container_width=True)

            st.session_state.last_solution = {
                "sol_str": sol_str,
                "endpoint_lines": endpoint_lines,
                "plot_png": png_bytes,
            }

            if reveal:
                st.markdown("#### ✅ Ορθή λύση")
                st.code(sol_str, language="text")
                if endpoint_lines:
                    st.markdown("**Ανοικτό/κλειστό:**")
                    for ln in endpoint_lines:
                        st.write(f"- {ln}")

            if export_pdf:
                payload = st.session_state.last_solution
                pdf_bytes = make_pdf_report(
                    exercise=ex,
                    user_text=user_answer,
                    sol_set_str=payload["sol_str"],
                    endpoint_lines=payload["endpoint_lines"],
                    fig_png_bytes=payload["plot_png"],
                    score=st.session_state.score,
                    streak=st.session_state.streak
                )
                st.download_button(
                    "⬇️ Κατέβασε το PDF",
                    data=pdf_bytes,
                    file_name=f"activity_{ex['id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Σφάλμα: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Εξερεύνηση (ελεύθερη είσοδος)")
    raw = st.text_area("Μία ανίσωση ανά γραμμή", value="Abs(x-2) < 3\nx^2 - 5*x + 6 > 0", height=120)
    xmin, xmax = st.slider("Εύρος αριθμητικής ευθείας", -50, 50, (-10, 10), key="explore_range")
    if st.button("✅ Λύσε", use_container_width=True, key="explore_solve"):
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            st.warning("Γράψε τουλάχιστον μία ανίσωση.")
        else:
            parsed = []
            sols = []
            for line in lines:
                rel = parse_inequality(line)
                sol = solve_ineq(rel)
                parsed.append((line, sol))
                sols.append(sol)

            common = sols[0]
            for sset in sols[1:]:
                common = sp.Intersection(common, sset)
            common = sp.simplify(common)

            left, right = st.columns(2)
            with left:
                st.markdown("#### Αποτελέσματα ανά ανίσωση")
                for i, (line, sol) in enumerate(parsed, start=1):
                    st.markdown(f"**{i}.** `{line}`")
                    st.code(str(sol), language="text")
                    fig = plot_number_line(sol, xmin=xmin, xmax=xmax, title=f"Λύση ανίσωσης {i}")
                    st.pyplot(fig, clear_figure=True)
                    st.divider()
            with right:
                st.markdown("#### 🤝 Κοινή λύση (Τομή)")
                st.code(str(common), language="text")
                fig = plot_number_line(common, xmin=xmin, xmax=xmax, title="Κοινή λύση")
                st.pyplot(fig, clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📘 Μικρή θεωρία")
    st.markdown("""
**Διαστήματα**
- `(a,b)` ανοικτό, `[a,b]` κλειστό  
- `(-∞,a]` όλα τα x ≤ a

**Απόλυτη τιμή**
- `|A| ≤ k` ⇔ `-k ≤ A ≤ k`  
- `|A| > k` ⇔ `A > k` ή `A < -k`

**Πίνακας προσήμων**
- Κρίσιμα σημεία (ρίζες, παρονομαστής=0)
- Πρόσημο σε διαστήματα
    """)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr>
<div style="text-align:center; opacity:0.7; font-size:0.95rem;">
Developed by <b>Nikolaos Sampanis</b> · Inequalities Quest · Streamlit
</div>
""", unsafe_allow_html=True)
