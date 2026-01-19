import re
import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt

# ==========================================================
# Symbol
# ==========================================================
x = sp.Symbol("x", real=True)

# ==========================================================
# Parsing / solving utilities
# ==========================================================

def normalize_input(s: str) -> str:
    """Normalize user input:
    - allow ^ for powers
    - allow abs(...) or Abs(...)
    """
    s = s.strip().replace("^", "**")
    s = re.sub(r"\babs\s*\(", "Abs(", s, flags=re.IGNORECASE)
    return s


def parse_inequality(line: str):
    """Parse a single inequality into a SymPy Relational."""
    line = normalize_input(line)
    ops = ["<=", ">=", "<", ">"]
    op = None
    for candidate in ops:
        if candidate in line:
            op = candidate
            break
    if op is None:
        raise ValueError("Δεν βρέθηκε τελεστής ανισότητας: χρησιμοποίησε <, <=, >, >=")

    left, right = line.split(op, 1)
    left, right = left.strip(), right.strip()

    local = {"x": x, "Abs": sp.Abs}
    L = sp.sympify(left, locals=local)
    R = sp.sympify(right, locals=local)

    if op == "<":
        return sp.Lt(L, R)
    if op == "<=":
        return sp.Le(L, R)
    if op == ">":
        return sp.Gt(L, R)
    if op == ">=":
        return sp.Ge(L, R)
    raise ValueError("Άγνωστος τελεστής.")


def solve_rel(rel):
    """Solve inequality over reals and return a SymPy Set."""
    sol = sp.solve_univariate_inequality(rel, x, relational=False)
    sol = sp.Intersection(sol, sp.S.Reals)
    return sp.simplify(sol)


def intervals_from_set(sol_set):
    if sol_set is sp.S.EmptySet:
        return []
    if sol_set is sp.S.Reals:
        return [sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)]
    if isinstance(sol_set, sp.Interval):
        return [sol_set]
    if isinstance(sol_set, sp.Union):
        return [arg for arg in sol_set.args if isinstance(arg, sp.Interval)]
    return []


def endpoint_explanation(sol_set):
    intervals = intervals_from_set(sol_set)
    if not intervals:
        return None

    def fmt(v):
        if v is sp.S.NegativeInfinity:
            return "-∞"
        if v is sp.S.Infinity:
            return "∞"
        return str(v)

    lines = []
    for I in intervals:
        left_closed = not I.left_open
        right_closed = not I.right_open
        left_symbol = "[" if left_closed else "("
        right_symbol = "]" if right_closed else ")"
        lines.append(
            f"• {left_symbol}{fmt(I.start)}, {fmt(I.end)}{right_symbol} → "
            f"αριστερό άκρο {'κλειστό' if left_closed else 'ανοικτό'}, "
            f"δεξί άκρο {'κλειστό' if right_closed else 'ανοικτό'}."
        )
    return "\n".join(lines)


def plot_number_line(sol_set, xmin=-10, xmax=10):
    """Simple number line plot for interval solutions."""
    fig, ax = plt.subplots(figsize=(9, 2.2))

    ax.hlines(0, xmin, xmax, linewidth=2)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)

    ax.spines[["left", "right", "top"]].set_visible(False)

    intervals = intervals_from_set(sol_set)

    def clamp(v):
        if v is sp.S.NegativeInfinity:
            return xmin
        if v is sp.S.Infinity:
            return xmax
        return max(xmin, min(xmax, float(v)))

    for I in intervals:
        a = clamp(I.start)
        b = clamp(I.end)
        ax.hlines(0, a, b, linewidth=8, alpha=0.6)

        # endpoints markers if visible in range
        if I.start not in (sp.S.NegativeInfinity, sp.S.Infinity):
            va = float(I.start)
            if xmin <= va <= xmax:
                if I.left_open:
                    ax.plot(va, 0, marker='o', markersize=9, fillstyle='none')
                else:
                    ax.plot(va, 0, marker='o', markersize=9)
        if I.end not in (sp.S.NegativeInfinity, sp.S.Infinity):
            vb = float(I.end)
            if xmin <= vb <= xmax:
                if I.right_open:
                    ax.plot(vb, 0, marker='o', markersize=9, fillstyle='none')
                else:
                    ax.plot(vb, 0, marker='o', markersize=9)

    ax.set_title("Αριθμητική ευθεία λύσεων", pad=10)
    return fig

# ==========================================================
# Activity bank (multiple choice)
# ==========================================================

ACTIVITIES = [
    {
        "title": "Γραμμική ανίσωση",
        "prompt": "Λύσε: 2x - 3 ≤ 5",
        "ineq": "2*x - 3 <= 5",
        "choices": [
            "x ≤ 4",
            "x < 4",
            "x ≥ 4",
            "x ∈ ( -∞, 4 )",
        ],
        "correct": 0,
        "hint": "Πρόσθεσε 3 και διαίρεσε με 2 (θετικός αριθμός, δεν αλλάζει το σύμβολο).",
    },
    {
        "title": "Απόλυτη τιμή",
        "prompt": "Λύσε: |x - 3| ≤ 5",
        "ineq": "Abs(x-3) <= 5",
        "choices": [
            "x ∈ [ -2, 8 ]",
            "x ∈ ( -2, 8 )",
            "x ∈ [ -8, 2 ]",
            "x ∈ ( -∞, -2 ] ∪ [ 8, ∞ )",
        ],
        "correct": 0,
        "hint": "|x-a| ≤ r  ⇔  a-r ≤ x ≤ a+r.",
    },
    {
        "title": "Πολυωνυμική (πίνακας προσήμων)",
        "prompt": "Λύσε: x² - 9 > 0",
        "ineq": "x**2 - 9 > 0",
        "choices": [
            "x ∈ ( -3, 3 )",
            "x ∈ ( -∞, -3 ) ∪ ( 3, ∞ )",
            "x ∈ [ -3, 3 ]",
            "x ∈ ( -∞, 3 )",
        ],
        "correct": 1,
        "hint": "x²-9=(x-3)(x+3). Θέλεις γινόμενο θετικό.",
    },
    {
        "title": "Ρητή ανίσωση",
        "prompt": "Λύσε: (x-1)/(x+2) ≥ 0",
        "ineq": "(x-1)/(x+2) >= 0",
        "choices": [
            "x ∈ ( -∞, -2 ) ∪ [ 1, ∞ )",
            "x ∈ ( -2, 1 )",
            "x ∈ [ -2, 1 ]",
            "x ∈ ( -∞, 1 ]",
        ],
        "correct": 0,
        "hint": "Κρίσιμα σημεία: x=-2 (απαγορεύεται) και x=1 (μηδενισμός αριθμητή).",
    },
]

# ==========================================================
# UI
# ==========================================================

st.set_page_config(page_title="Ανισώσεις Α’ Λυκείου — Visual Solver", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 16px 18px;
  background: rgba(255,255,255,0.03);
}
.small {opacity: 0.85; font-size: 0.95rem;}
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18); margin-right: 8px;
  font-size: 0.9rem; opacity:0.95;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 Visual Solver Ανισώσεων (Α’ Λυκείου)")
st.markdown(
    "Λύνει ανισώσεις στο ℝ, οπτικοποιεί σε αριθμητική ευθεία, και βρίσκει κοινές λύσεις (τομή)."
)

# Tabs
explore_tab, activity_tab, theory_tab = st.tabs(["🔍 Εξερεύνηση", "🎯 Δραστηριότητα", "🧠 Θεωρία"])

# -----------------------------
# Explore Tab
# -----------------------------
with explore_tab:
    colA, colB = st.columns([1.05, 0.95], gap="large")

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Είσοδος ανισώσεων")
        st.markdown(
            '<span class="badge">Σύνταξη</span> π.χ. `2*x-3 <= 5`, `Abs(x-2) < 3`, `x^2 - 5*x + 6 > 0`',
            unsafe_allow_html=True,
        )

        default_text = "2*x - 3 <= 5\nAbs(x-2) < 3\nx^2 - 5*x + 6 > 0"
        raw = st.text_area("Ανισώσεις (μία ανά γραμμή)", value=default_text, height=170)

        st.markdown("### 🔧 Ρυθμίσεις οπτικοποίησης")
        xmin, xmax = st.slider("Εύρος αριθμητικής ευθείας", -50, 50, (-10, 10))
        st.markdown(
            '<p class="small">Σημείωση: τα άκρα ±∞ «κόβονται» στο επιλεγμένο εύρος για το γράφημα.</p>',
            unsafe_allow_html=True,
        )

        solve_btn = st.button("✅ Λύσε", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Γρήγορα παραδείγματα")
        ex1 = "x - 4 < 2"
        ex2 = "Abs(x-3) <= 5"
        ex3 = "(x-1)/(x+2) >= 0"
        ex4 = "x^2 - 9 <= 0"
        b1, b2 = st.columns(2)
        if b1.button("Παράδειγμα 1", use_container_width=True):
            raw = ex1
        if b2.button("Παράδειγμα 2", use_container_width=True):
            raw = ex2
        if b1.button("Παράδειγμα 3", use_container_width=True):
            raw = ex3
        if b2.button("Παράδειγμα 4", use_container_width=True):
            raw = ex4

        st.markdown("### 🧩 Tips σύνταξης")
        st.markdown(
            """
- Μεταβλητή: `x`
- Δύναμη: `^` ή `**` (π.χ. `x^2`)
- Απόλυτο: `Abs(...)` ή `abs(...)`
- Σύμβολα: `<  <=  >  >=`
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if solve_btn:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            st.warning("Γράψε τουλάχιστον μία ανίσωση.")
            st.stop()

        st.divider()

        parsed = []
        solutions = []
        for i, line in enumerate(lines, start=1):
            try:
                rel = parse_inequality(line)
                sol = solve_rel(rel)
                parsed.append((line, rel, sol))
                solutions.append(sol)
            except Exception as e:
                st.error(f"Σφάλμα στη γραμμή {i}: `{line}`\n\nΛεπτομέρειες: {e}")
                st.stop()

        common = solutions[0]
        for s in solutions[1:]:
            common = sp.Intersection(common, s)
        common = sp.simplify(common)

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("## 📌 Αποτελέσματα ανά ανίσωση")
            for idx, (line, rel, sol) in enumerate(parsed, start=1):
                st.markdown(f"### Ανίσωση {idx}")
                st.code(line, language="text")
                st.markdown("**Λύση (διαστήματα / σύνολα):**")
                st.code(str(sol), language="text")

                expl = endpoint_explanation(sol)
                if expl:
                    st.markdown("**Ανοικτό / κλειστό:**")
                    st.markdown(expl)

                fig = plot_number_line(sol, xmin=xmin, xmax=xmax)
                st.pyplot(fig, clear_figure=True)
                st.divider()

        with right:
            st.markdown("## 🤝 Κοινή λύση (Τομή)")
            st.markdown("Τα x που ικανοποιούν **όλες** τις ανισώσεις ταυτόχρονα.")
            st.markdown("**Κοινή λύση:**")
            st.code(str(common), language="text")

            expl = endpoint_explanation(common)
            if expl:
                st.markdown("**Ανοικτό / κλειστό:**")
                st.markdown(expl)

            fig = plot_number_line(common, xmin=xmin, xmax=xmax)
            st.pyplot(fig, clear_figure=True)

# -----------------------------
# Activity Tab
# -----------------------------
with activity_tab:
    st.markdown("""
<div class="card">
<h3>🎯 Δραστηριότητα (Multiple Choice + Επιβεβαίωση)</h3>
<p class="small">
Διάλεξε μια άσκηση, απάντησε, και μετά πάτα <b>Έλεγχος</b> για να δεις τη λύση και την αριθμητική ευθεία.
</p>
</div>
""", unsafe_allow_html=True)

    titles = [a["title"] + " — " + a["prompt"] for a in ACTIVITIES]
    idx = st.selectbox("Επίλεξε άσκηση", range(len(ACTIVITIES)), format_func=lambda i: titles[i])
    act = ACTIVITIES[idx]

    st.markdown("### 📝 Εκφώνηση")
    st.info(act["prompt"])

    choice = st.radio("Επίλεξε σωστή απάντηση", act["choices"], index=None)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        check = st.button("✅ Έλεγχος", use_container_width=True)
    with c2:
        show_hint = st.button("💡 Υπόδειξη", use_container_width=True)

    if show_hint:
        st.warning(act["hint"])

    if check:
        if choice is None:
            st.warning("Διάλεξε πρώτα μια απάντηση.")
            st.stop()

        correct_text = act["choices"][act["correct"]]
        if choice == correct_text:
            st.success("Σωστό! ✅")
        else:
            st.error("Όχι ακόμη — δες τη σωστή λύση παρακάτω.")
            st.markdown(f"**Σωστή απάντηση:** {correct_text}")

        # Solve and display
        try:
            rel = parse_inequality(act["ineq"])
            sol = solve_rel(rel)
        except Exception as e:
            st.error(f"Σφάλμα στη λύση της άσκησης: {e}")
            st.stop()

        st.markdown("### ✅ Επίσημη λύση")
        st.code(str(sol), language="text")

        expl = endpoint_explanation(sol)
        if expl:
            st.markdown("**Ανοικτό / κλειστό:**")
            st.markdown(expl)

        xmin, xmax = st.slider("Εύρος ευθείας (δραστηριότητα)", -50, 50, (-10, 10), key=f"rng_{idx}")
        fig = plot_number_line(sol, xmin=xmin, xmax=xmax)
        st.pyplot(fig, clear_figure=True)

# -----------------------------
# Theory Tab
# -----------------------------
with theory_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🧠 Θεωρία (συνοπτικά)")
    st.markdown(
        """
### 1) Διαστήματα και σύμβολα
- **Ανοικτό διάστημα**: `(a, b)` → τα άκρα *δεν* ανήκουν.
- **Κλειστό διάστημα**: `[a, b]` → τα άκρα *ανήκουν*.
- Μικτά: `(a, b]` ή `[a, b)`.

### 2) Απόλυτη τιμή
- `|x-a| ≤ r`  ⇔  `a-r ≤ x ≤ a+r`
- `|x-a| < r`  ⇔  `a-r < x < a+r`
- `|x-a| ≥ r`  ⇔  `x ≤ a-r` ή `x ≥ a+r`

### 3) Κοινές λύσεις (τομή)
- Αν έχεις 2 ανισώσεις, η **κοινή λύση** είναι τα x που ισχύουν και στις δύο.
- Πρακτικά: παίρνεις τα διαστήματα και κρατάς την **τομή**.

### 4) Συμβουλή για πίνακα προσήμων
- Για πολυωνυμικές/ρητές ανισώσεις: βρίσκεις κρίσιμα σημεία (ρίζες, απαγορευμένες τιμές)
  και ελέγχεις πρόσημο σε κάθε διάστημα.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# Footer branding
# ==========================================================
st.markdown(
    """
<hr>
<div style="text-align:center; opacity:0.78; font-size:0.95rem;">
Developed by <b>Nikolaos Sampanis</b> · Mathematics Education
</div>
""",
    unsafe_allow_html=True,
)
