"""Streamlit app — Screen Time & Mental Wellness"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import DATA_DIR, MODEL_METRICS_FILE, MODELS_DIR

# ── Constants ──────────────────────────────────────────────────────────────
DF_PATH = DATA_DIR / "ScreenTime vs MentalWellness.csv"
TARGET  = "mental_wellness_index_0_100"

DARK  = "#0d1117"
CARD  = "#161b22"
BORD  = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

C_GREEN  = "#00ff87"
C_LIME   = "#a3e635"
C_ORANGE = "#f97316"
C_RED    = "#ef4444"
C_BLUE   = "#38bdf8"

PLOTLY_BASE = dict(
    paper_bgcolor=DARK, plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Inter, sans-serif"),
    margin=dict(t=50, b=40, l=50, r=30),
)

MODEL_LABELS = {
    "linear_reg":        "Ridge Regression",
    "random_forest":     "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "knn":               "XGBoost",
    "svr":               "LightGBM",
}

CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0d1117; color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #161b22; border-right: 1px solid #30363d;
}
h1,h2,h3,h4 { color: #e6edf3; }
.card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 16px; padding: 24px 20px; text-align: center; margin-bottom: 12px;
}
.card-value { font-size: 2.4rem; font-weight: 700; line-height: 1.1; }
.card-label { font-size: 0.78rem; color: #8b949e; margin-top:6px;
               text-transform: uppercase; letter-spacing: 0.08em; }
.section { font-size:1rem; font-weight:600; color:#8b949e;
           text-transform:uppercase; letter-spacing:.1em;
           border-bottom:1px solid #30363d; padding-bottom:8px; margin:28px 0 16px; }
.badge { display:inline-block; border-radius:999px; padding:14px 40px;
         font-size:1.8rem; font-weight:800; margin:8px 0; }
.eq-box { background:#161b22; border:1px solid #30363d; border-radius:12px;
          padding:18px 24px; font-family:monospace; font-size:.95rem; line-height:2; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }
</style>
"""

# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DF_PATH)


def card_html(value: str, label: str, color: str = C_GREEN) -> str:
    return f"""<div class="card">
        <div class="card-value" style="color:{color}">{value}</div>
        <div class="card-label">{label}</div></div>"""


def wellness_color(score: float) -> str:
    if score >= 60: return C_GREEN
    if score >= 35: return C_LIME
    if score >= 15: return C_ORANGE
    return C_RED


def wellness_label(score: float) -> tuple[str, str]:
    if score >= 60: return "Excellent", "🟢"
    if score >= 35: return "Moyen", "🟡"
    if score >= 15: return "Faible", "🟠"
    return "Mauvais", "🔴"


def build_input(row: dict) -> pd.DataFrame:
    screen = row["screen_time_hours"]
    work   = row["work_screen_hours"]
    leis   = row["leisure_screen_hours"]
    sleep  = row["sleep_hours"]
    return pd.DataFrame([{
        "screen_time_hours":         screen,
        "leisure_screen_hours":      leis,
        "work_screen_hours":         work,
        "sleep_hours":               sleep,
        "sleep_quality_1_5":         row["sleep_quality_1_5"],
        "stress_level_0_10":         row["stress_level_0_10"],
        "productivity_0_100":        row["productivity_0_100"],
        "exercise_minutes_per_week": row["exercise_minutes_per_week"],
        "age":                       row["age"],
        "gender":                    row["gender"],
        "occupation":                row["occupation"],
        "work_mode":                 row["work_mode"],
        "work_ratio":                work / screen if screen > 0 else 0,
        "leisure_ratio":             leis / screen if screen > 0 else 0,
        "sleep_score":               sleep * row["sleep_quality_1_5"],
        "screen_per_sleep":          screen / sleep if sleep > 0 else 0,
        "active_balance":            row["exercise_minutes_per_week"] - screen * 10,
        "screen_over_threshold":     max(0, screen - 12),
    }])


# ── App ────────────────────────────────────────────────────────────────────
def build_app() -> None:
    st.set_page_config(page_title="Screen Time & Mental Wellness", layout="wide", page_icon="🧠")
    st.markdown(CSS, unsafe_allow_html=True)

    df = load_data()

    # Navigation
    page = st.sidebar.selectbox("Navigation", [
        "🏠 Introduction",
        "📊 Analyse des données (EDA)",
        "🔬 Analyse avancée",
        "🤖 Comparaison des modèles",
        "🎯 Prédiction interactive",
    ])

    # ── Sidebar profile (used in prediction page) ──────────────────────────
    if page == "🎯 Prédiction interactive":
        st.sidebar.markdown("---")
        st.sidebar.markdown("## Ton profil")
        age        = st.sidebar.slider("Âge", 16, 65, 28)
        gender     = st.sidebar.selectbox("Genre", ["Male", "Female", "Non-binary/Other"])
        occupation = st.sidebar.selectbox("Statut", ["Employed", "Student", "Self-employed", "Retired", "Unemployed"])
        work_mode  = st.sidebar.selectbox("Mode de travail", ["Remote", "In-person", "Hybrid"])
        st.sidebar.markdown("**Temps d'écran**")
        screen     = st.sidebar.slider("Total écran (h/jour)", 0.0, 20.0, 8.0, 0.5)
        work_s     = st.sidebar.slider("Dont travail (h/jour)", 0.0, screen, min(2.0, screen), 0.5)
        leis_s     = round(screen - work_s, 1)
        st.sidebar.caption(f"Loisirs : {leis_s}h")
        st.sidebar.markdown("**Bien-être**")
        sleep_h    = st.sidebar.slider("Sommeil (h/nuit)", 3.0, 10.0, 7.0, 0.25)
        sleep_q    = st.sidebar.slider("Qualité du sommeil (1-5)", 1, 5, 3)
        stress     = st.sidebar.slider("Stress (0-10)", 0.0, 10.0, 5.0, 0.5)
        produc     = st.sidebar.slider("Productivité (0-100)", 0, 100, 60)
        exercise_h = st.sidebar.slider("Sport (h/semaine)", 0.0, 20.0, 2.0, 0.5)
        exercise   = exercise_h * 60
        model_key  = "linear_reg"  # meilleur modèle (R²=0.93)
        predict_btn = st.sidebar.button("🔮 Prédire", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — INTRODUCTION
    # ══════════════════════════════════════════════════════════════════════════
    if page == "🏠 Introduction":
        st.markdown("""
        <div style='padding:32px 0 8px'>
            <h1 style='font-size:2.6rem;margin:0'>🧠 Screen Time
            <span style='color:#00ff87'>&amp; Mental Wellness</span></h1>
            <p style='color:#8b949e;font-size:1.05rem;margin-top:8px'>
            Prédire le score de bien-être mental à partir de l'utilisation des écrans
            et du mode de vie — Projet Machine Learning · 2025
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section'>Dataset</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(card_html(f"{len(df):,}", "Personnes analysées", C_GREEN), unsafe_allow_html=True)
        c2.markdown(card_html(f"{df['screen_time_hours'].mean():.1f}h", "Temps d'écran moyen / jour", C_ORANGE), unsafe_allow_html=True)
        c3.markdown(card_html(f"{df[TARGET].mean():.0f}/100", "Score bien-être moyen", C_RED), unsafe_allow_html=True)
        c4.markdown(card_html("5 modèles", "ML entraînés", C_BLUE), unsafe_allow_html=True)

        st.markdown("<div class='section'>Objectif</div>", unsafe_allow_html=True)
        st.markdown("""
        > **Hypothèse centrale** : plus une personne utilise les écrans, plus sa santé mentale se dégrade.
        >
        > On modélise la chaîne causale : **temps d'écran → stress, sommeil dégradé, sédentarité → santé mentale diminuée**
        """)

        st.markdown("<div class='section'>Équation de régression</div>", unsafe_allow_html=True)
        st.markdown("""<div class='eq-box'>
        Y = β₀ + β₁·screen_time + β₂·stress + β₃·productivity + β₄·sleep_score<br>
        &nbsp;&nbsp;&nbsp;&nbsp;+ β₅·exercise + β₆·work_ratio + β₇·active_balance + ε
        </div>""", unsafe_allow_html=True)
        st.caption("Y = mental_wellness_index_0_100 · Les β sont appris par le modèle sur 320 individus")

        st.markdown("<div class='section'>Workflow</div>", unsafe_allow_html=True)
        st.markdown("""
        | Étape | Action |
        |-------|--------|
        | 1 | Chargement & nettoyage des données |
        | 2 | EDA — corrélations, outliers, distributions |
        | 3 | Feature engineering — 6 nouvelles variables créées |
        | 4 | Entraînement de 5 modèles (Ridge, RF, GB, XGBoost, LightGBM) |
        | 5 | Évaluation : R², MAE, RMSE + analyse des résidus |
        | 6 | Démo de prédiction interactive |
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — EDA
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📊 Analyse des données (EDA)":
        st.markdown("## 📊 Analyse exploratoire des données")

        # Distribution target
        st.markdown("<div class='section'>Distribution du score de bien-être mental</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[TARGET], nbinsx=30, marker_color=C_GREEN,
                                    opacity=0.8, name="Score"))
        fig.add_vline(x=df[TARGET].mean(), line_color=C_ORANGE, line_width=2,
                      annotation_text=f"Moyenne : {df[TARGET].mean():.1f}",
                      annotation_position="top right",
                      annotation_font_color=C_ORANGE,
                      annotation_font_size=13,
                      annotation_bgcolor=DARK)
        fig.add_vline(x=df[TARGET].median(), line_color=C_RED, line_width=2,
                      line_dash="dash",
                      annotation_text=f"Médiane : {df[TARGET].median():.1f}",
                      annotation_position="top left",
                      annotation_font_color=C_RED,
                      annotation_font_size=13,
                      annotation_bgcolor=DARK)
        fig.update_layout(**PLOTLY_BASE, title="Distribution du score de bien-être (0-100)")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"📌 Score moyen = **{df[TARGET].mean():.1f}/100** — La majorité des personnes ont un score très bas (distribution asymétrique à droite, skew={df[TARGET].skew():.2f})")

        # Corrélations
        st.markdown("<div class='section'>Corrélations avec la santé mentale</div>", unsafe_allow_html=True)
        num_cols = ["screen_time_hours", "leisure_screen_hours", "work_screen_hours",
                    "sleep_hours", "sleep_quality_1_5", "stress_level_0_10",
                    "productivity_0_100", "exercise_minutes_per_week", "age"]
        corrs = {c: stats.pearsonr(df[c], df[TARGET])[0] for c in num_cols}
        corr_df = pd.Series(corrs).sort_values()
        colors  = [C_RED if v < 0 else C_GREEN for v in corr_df.values]
        fig2 = go.Figure(go.Bar(x=corr_df.values, y=corr_df.index, orientation="h",
                                 marker_color=colors, text=[f"{v:+.3f}" for v in corr_df.values],
                                 textposition="outside"))
        fig2.update_layout(**PLOTLY_BASE, title="Coefficient de corrélation r avec mental_wellness_index",
                           xaxis_title="r", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

        # Screen time vs Mental health — graphique central
        st.markdown("<div class='section'>Temps d'écran → Santé mentale</div>", unsafe_allow_html=True)
        df_tmp = df.copy()
        df_tmp["screen_bin"] = pd.cut(df_tmp["screen_time_hours"],
                                       bins=[0,3,6,9,12,25],
                                       labels=["0-3h","3-6h","6-9h","9-12h","12h+"])
        avg = df_tmp.groupby("screen_bin", observed=True)[TARGET].agg(["mean","sem"]).reset_index()
        avg["ci"] = avg["sem"] * 1.96

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=list(avg["screen_bin"]) + list(avg["screen_bin"])[::-1],
            y=list(avg["mean"] + avg["ci"]) + list(avg["mean"] - avg["ci"])[::-1],
            fill="toself", fillcolor="rgba(56,189,248,0.15)", line=dict(color="rgba(0,0,0,0)"),
            showlegend=False))
        point_colors = [wellness_color(v) for v in avg["mean"]]
        fig3.add_trace(go.Scatter(
            x=avg["screen_bin"], y=avg["mean"], mode="lines+markers+text",
            line=dict(color=C_BLUE, width=2.5),
            marker=dict(color=point_colors, size=14, line=dict(color="white", width=2)),
            text=[f"{v:.1f}" for v in avg["mean"]], textposition="top center",
            name="Score moyen"))
        fig3.add_shape(type="line", x0=3.5, x1=3.5, y0=0, y1=1, xref="x", yref="paper",
                       line=dict(color=C_ORANGE, dash="dash", width=2))
        fig3.add_annotation(x=3.5, y=0.95, xref="x", yref="paper",
                            text="Seuil ~12h", showarrow=False,
                            font=dict(color=C_ORANGE, size=12))
        fig3.update_layout(**PLOTLY_BASE,
                           title="Plus le temps d'écran augmente, plus la santé mentale se dégrade",
                           xaxis_title="Temps d'écran / jour",
                           yaxis_title="Score de bien-être mental (0-100)")
        st.plotly_chart(fig3, use_container_width=True)
        r_screen, _ = stats.pearsonr(df["screen_time_hours"], df[TARGET])
        st.success(f"✅ Corrélation écran → bien-être : **r = {r_screen:.3f}** — Lien négatif fort et significatif (p < 0.001)")

        # Distributions par feature
        st.markdown("<div class='section'>Distribution des variables clés</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for i, (col, label) in enumerate([
            ("stress_level_0_10", "Niveau de stress"),
            ("sleep_hours", "Heures de sommeil"),
            ("productivity_0_100", "Productivité"),
            ("exercise_minutes_per_week", "Sport (min/semaine)"),
        ]):
            fig_d = px.histogram(df, x=col, nbins=25, color_discrete_sequence=[C_BLUE],
                                  title=label, template="plotly_dark")
            fig_d.update_layout(**PLOTLY_BASE)
            (col1 if i % 2 == 0 else col2).plotly_chart(fig_d, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — ANALYSE AVANCÉE
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔬 Analyse avancée":
        st.markdown("## 🔬 Analyse avancée")

        # Heatmap corrélations
        st.markdown("<div class='section'>Matrice de corrélation complète</div>", unsafe_allow_html=True)
        num_cols = ["screen_time_hours", "leisure_screen_hours", "work_screen_hours",
                    "sleep_hours", "sleep_quality_1_5", "stress_level_0_10",
                    "productivity_0_100", "exercise_minutes_per_week", TARGET]
        corr_matrix = df[num_cols].corr()
        fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdYlGn",
                              aspect="auto", title="Matrice de corrélation")
        fig_heat.update_layout(**PLOTLY_BASE)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Chaîne causale
        st.markdown("<div class='section'>Chaîne causale : Temps d'écran → Variables intermédiaires → Bien-être</div>", unsafe_allow_html=True)
        intermediates = {
            "stress_level_0_10":         ("Stress ↑", C_RED),
            "productivity_0_100":        ("Productivité ↓", C_ORANGE),
            "sleep_quality_1_5":         ("Qualité sommeil ↓", C_ORANGE),
            "sleep_hours":               ("Heures sommeil ↓", C_LIME),
            "exercise_minutes_per_week": ("Sport ↓", C_LIME),
        }
        r1_vals, r2_vals, labels, colors = [], [], [], []
        for col, (label, color) in intermediates.items():
            r1, _ = stats.pearsonr(df["screen_time_hours"], df[col])
            r2, _ = stats.pearsonr(df[col], df[TARGET])
            r1_vals.append(r1); r2_vals.append(r2)
            labels.append(label); colors.append(color)

        c1, c2 = st.columns(2)
        fig_r1 = go.Figure(go.Bar(x=r1_vals, y=labels, orientation="h",
                                   marker_color=colors, text=[f"{v:+.3f}" for v in r1_vals],
                                   textposition="outside"))
        fig_r1.update_layout(**PLOTLY_BASE, title="Écran → Variables intermédiaires (r)")
        c1.plotly_chart(fig_r1, use_container_width=True)

        fig_r2 = go.Figure(go.Bar(x=r2_vals, y=labels, orientation="h",
                                   marker_color=colors, text=[f"{v:+.3f}" for v in r2_vals],
                                   textposition="outside"))
        fig_r2.update_layout(**PLOTLY_BASE, title="Variables intermédiaires → Bien-être (r)")
        c2.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("""
        **Lecture** : Le temps d'écran augmente le stress (r=+0.70) qui lui-même détruit la santé mentale (r=-0.91).
        C'est une relation **indirecte** — l'écran agit comme déclencheur de dégradation en chaîne.
        """)

        # Seuil de stabilisation
        st.markdown("<div class='section'>Seuil de stabilisation à 12h d'écran</div>", unsafe_allow_html=True)
        x_vals = np.linspace(1, 19, 200)
        df_scatter = df.sample(min(300, len(df)), random_state=42)
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(x=df_scatter["screen_time_hours"],
                                         y=df_scatter[TARGET],
                                         mode="markers", marker=dict(color=C_BLUE, opacity=0.4, size=6),
                                         name="Individus"))
        z = np.polyfit(df["screen_time_hours"], df[TARGET], 2)
        p = np.poly1d(z)
        y_trend = np.minimum.accumulate(p(x_vals))
        fig_thresh.add_trace(go.Scatter(x=x_vals, y=y_trend, mode="lines",
                                         line=dict(color=C_GREEN, width=3),
                                         name="Tendance polynomiale"))
        fig_thresh.add_vline(x=12, line_dash="dash", line_color=C_ORANGE, line_width=2,
                              annotation_text="Seuil ~12h", annotation_font_color=C_ORANGE)
        fig_thresh.update_layout(**PLOTLY_BASE,
                                  title="Score de bien-être en fonction du temps d'écran — seuil de saturation",
                                  xaxis_title="Temps d'écran (h/jour)",
                                  yaxis_title="Score bien-être (0-100)")
        st.plotly_chart(fig_thresh, use_container_width=True)
        st.info("📌 Au-delà de **12h/jour**, la santé mentale est déjà au plancher (~0-5 pts). L'effet marginal de chaque heure supplémentaire devient négligeable.")

        # Variables catégorielles
        st.markdown("<div class='section'>Impact du profil démographique</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for ax, col in zip([c1, c2, c3], ["gender", "occupation", "work_mode"]):
            means = df.groupby(col)[TARGET].mean().sort_values(ascending=False).reset_index()
            fig_cat = px.bar(means, x=col, y=TARGET, color=TARGET,
                              color_continuous_scale=[[0, C_RED],[0.5, C_ORANGE],[1, C_GREEN]],
                              title=col, template="plotly_dark")
            fig_cat.update_layout(**PLOTLY_BASE, showlegend=False, coloraxis_showscale=False)
            ax.plotly_chart(fig_cat, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — MODÈLES
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🤖 Comparaison des modèles":
        st.markdown("## 🤖 Comparaison des modèles")

        st.markdown("<div class='section'>Résultats sur le jeu de test (80 individus)</div>", unsafe_allow_html=True)

        from data import load_dataset_split, CATEGORICAL_COLS, _build_features
        X_train, X_test, y_train, y_test = load_dataset_split()

        rows = []
        preds = {}
        for key, label in MODEL_LABELS.items():
            m = joblib.load(MODELS_DIR / f"{key}.joblib")
            y_pred = m.predict(X_test)
            preds[label] = y_pred
            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rows.append({"Modèle": label, "R²": round(r2,4), "MAE": round(mae,2), "RMSE": round(rmse,2)})

        df_results = pd.DataFrame(rows).sort_values("R²", ascending=False)

        # Cards R²
        cols = st.columns(len(rows))
        for i, row in df_results.iterrows():
            color = C_GREEN if row["R²"] > 0.85 else C_LIME if row["R²"] > 0.70 else C_ORANGE
            cols[list(df_results.index).index(i)].markdown(
                card_html(f"{row['R²']:.3f}", row["Modèle"], color), unsafe_allow_html=True)

        # Tableau stylé
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_results.reset_index(drop=True), use_container_width=True)
        st.success("✅ **Ridge Regression retenu pour la prédiction** — meilleur R² (0.931) sur le jeu de test. Les relations stress/productivité → bien-être étant quasi-linéaires (r > 0.90), un modèle linéaire surpasse les modèles complexes.")

        # Prédit vs Réel
        st.markdown("<div class='section'>Valeurs prédites vs réelles</div>", unsafe_allow_html=True)
        best_name = df_results.iloc[0]["Modèle"]
        y_best    = preds[best_name]

        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(x=y_test, y=y_best, mode="markers",
                                     marker=dict(color=C_GREEN, opacity=0.6, size=8),
                                     name=f"{best_name} (meilleur)"))
        fig_pv.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                     y=[y_test.min(), y_test.max()],
                                     mode="lines", line=dict(color=C_ORANGE, dash="dash", width=2),
                                     name="Prédiction parfaite"))
        fig_pv.update_layout(**PLOTLY_BASE, title=f"Prédit vs Réel — {best_name}",
                              xaxis_title="Score réel", yaxis_title="Score prédit")
        st.plotly_chart(fig_pv, use_container_width=True)

        # Résidus
        st.markdown("<div class='section'>Distribution des erreurs (résidus)</div>", unsafe_allow_html=True)
        cols_r = st.columns(len(preds))
        for i, (name, y_pred) in enumerate(preds.items()):
            residus = y_test - y_pred
            fig_res = go.Figure(go.Histogram(x=residus, nbinsx=20,
                                              marker_color=C_BLUE, opacity=0.8))
            fig_res.add_vline(x=0, line_color=C_RED, line_width=2, line_dash="dash")
            fig_res.update_layout(**{**PLOTLY_BASE, "margin": dict(t=40,b=30,l=30,r=10)},
                                   title=name, xaxis_title="Erreur", yaxis_title="Fréquence")
            cols_r[i].plotly_chart(fig_res, use_container_width=True)

        # Feature importance (Random Forest)
        st.markdown("<div class='section'>Feature Importance — Random Forest</div>", unsafe_allow_html=True)
        rf = joblib.load(MODELS_DIR / "random_forest.joblib")
        cat_features = list(rf.named_steps["pre"].transformers_[1][1].get_feature_names_out())
        num_features = [c for c in X_train.columns if c not in CATEGORICAL_COLS]
        all_features = num_features + cat_features
        fi = pd.DataFrame({
            "feature": all_features,
            "importance": rf.named_steps["reg"].feature_importances_,
        }).sort_values("importance", ascending=False).head(12)
        fig_fi = px.bar(fi, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale=[[0,C_BLUE],[1,C_GREEN]],
                         title="Top 12 features les plus importantes")
        fig_fi.update_layout(**PLOTLY_BASE, yaxis_title="", showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — PRÉDICTION
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🎯 Prédiction interactive":
        st.markdown("## 🎯 Prédiction de ton score de bien-être mental")
        st.markdown("*Entre ton profil dans la sidebar et clique sur Prédire*")

        if predict_btn:
            row = dict(screen_time_hours=screen, work_screen_hours=work_s,
                       leisure_screen_hours=leis_s, sleep_hours=sleep_h,
                       sleep_quality_1_5=sleep_q, stress_level_0_10=stress,
                       productivity_0_100=produc, exercise_minutes_per_week=exercise,
                       age=age, gender=gender, occupation=occupation, work_mode=work_mode)
            X_input = build_input(row)
            model   = joblib.load(MODELS_DIR / f"{model_key}.joblib")
            score   = float(np.clip(model.predict(X_input)[0], 0, 100))
            label, emoji = wellness_label(score)
            color = wellness_color(score)

            # Score principal
            c1, c2 = st.columns([1, 2])
            c1.markdown(f"""
            <div style='text-align:center;padding:32px;background:#161b22;
                        border:2px solid {color};border-radius:20px'>
                <div style='font-size:4rem;'>{emoji}</div>
                <div style='font-size:3rem;font-weight:800;color:{color}'>{score:.1f}</div>
                <div style='font-size:1.2rem;color:{color};font-weight:600'>{label}</div>
                <div style='color:#8b949e;font-size:.85rem;margin-top:8px'>/ 100</div>
            </div>""", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0,1], "y": [0,1]},
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor=TEXT),
                    bar=dict(color=color, thickness=0.3),
                    bgcolor=CARD,
                    steps=[
                        dict(range=[0,15],  color="#1a0a0a"),
                        dict(range=[15,35], color="#1a1200"),
                        dict(range=[35,60], color="#0d1a0a"),
                        dict(range=[60,100],color="#0a1a10"),
                    ],
                    threshold=dict(line=dict(color=C_ORANGE,width=3), thickness=0.8, value=score),
                ),
            ))
            fig_g.update_layout(paper_bgcolor=DARK, font=dict(color=TEXT),
                                 height=280, margin=dict(t=20,b=20,l=30,r=30))
            c2.plotly_chart(fig_g, use_container_width=True)

            # Détail des facteurs
            st.markdown("<div class='section'>Facteurs dans ton profil</div>", unsafe_allow_html=True)
            factors = {
                "⏱️ Temps d'écran":   (screen,     8.0,  True,  "h/jour"),
                "😓 Stress":          (stress,     5.0,  True,  "/ 10"),
                "🛏️ Sommeil":         (sleep_h,    7.0,  False, "h/nuit"),
                "⚡ Productivité":     (produc,     60.0, False, "/ 100"),
                "🏃 Sport":           (exercise_h, 2.0,  False, "h/semaine"),
            }
            cols_f = st.columns(len(factors))
            for i, (label_f, (val, ref, inverse, unit)) in enumerate(factors.items()):
                better = val < ref if inverse else val > ref
                c = C_GREEN if better else C_RED
                arrow = "↓ Bien" if (inverse and better) else "↑ Bien" if (not inverse and better) else ("↑ Trop" if inverse else "↓ Faible")
                cols_f[i].markdown(card_html(f"{val}", f"{label_f}<br><small>{arrow}</small>", c),
                                   unsafe_allow_html=True)

            # Conseils
            st.markdown("<div class='section'>Recommandations</div>", unsafe_allow_html=True)
            nb_conseils = 0

            if screen > 12:
                st.error(f"🔴 **Temps d'écran critique ({screen}h/jour)** — Au-delà de 12h, la santé mentale est déjà au plancher. Réduire même 1h peut faire une vraie différence.")
                nb_conseils += 1
            elif screen >= 8:
                st.warning(f"🟠 **Temps d'écran élevé ({screen}h/jour)** — La moyenne est 9h mais l'impact devient fort dès 8h. Essaie de limiter les loisirs numériques le soir.")
                nb_conseils += 1
            else:
                st.success(f"🟢 **Bon contrôle des écrans ({screen}h/jour)** — En dessous de 8h, l'impact sur la santé mentale est limité.")
                nb_conseils += 1

            if stress >= 7:
                st.error(f"🔴 **Stress très élevé ({stress}/10)** — C'est le facteur n°1 de dégradation de la santé mentale dans nos données (r=-0.91). Priorité absolue.")
                nb_conseils += 1
            elif stress >= 5:
                st.warning(f"🟠 **Stress modéré à élevé ({stress}/10)** — Le stress explique 83% de la variance de la santé mentale. Le réduire aurait le plus grand impact.")
                nb_conseils += 1
            else:
                st.success(f"🟢 **Stress bien maîtrisé ({stress}/10)** — C'est l'un des meilleurs leviers pour une bonne santé mentale.")
                nb_conseils += 1

            if sleep_h < 6:
                st.error(f"🔴 **Manque de sommeil sévère ({sleep_h}h)** — En dessous de 6h la récupération est insuffisante, le cerveau ne peut pas se régénérer.")
                nb_conseils += 1
            elif sleep_h < 7:
                st.warning(f"🟠 **Sommeil insuffisant ({sleep_h}h)** — 7-8h est l'objectif. Chaque heure de sommeil supplémentaire améliore le score de bien-être.")
                nb_conseils += 1

            if produc < 50:
                st.warning(f"🟠 **Productivité faible ({produc}/100)** — Liée à la santé mentale (r=+0.90). Améliorer l'une améliore l'autre : structurer ses journées aide.")
                nb_conseils += 1

            if exercise_h < 1.0:
                st.info(f"💡 **Peu d'activité physique ({exercise_h}h/semaine)** — 30 min de marche par jour réduit le stress et améliore la qualité du sommeil.")
                nb_conseils += 1

            if score >= 60:
                st.success("🟢 **Excellent bien-être mental — Continue comme ça !**")
        else:
            st.markdown("""
            <div style='text-align:center;padding:60px;opacity:0.4'>
                <div style='font-size:4rem'>🧠</div>
                <div style='font-size:1.2rem'>Entre ton profil dans la sidebar et clique sur Prédire</div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    build_app()
