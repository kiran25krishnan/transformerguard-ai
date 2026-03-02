import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="TransformerGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    color: #e8edf5;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.2);
  }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Hide default header */
  header[data-testid="stHeader"] { display: none; }

  /* Cards */
  .glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
  }

  .metric-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.05) 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }

  /* Fault badges */
  .badge-normal      { background:#065f46; color:#6ee7b7; border:1px solid #10b981; }
  .badge-pd          { background:#1e3a5f; color:#93c5fd; border:1px solid #3b82f6; }
  .badge-led         { background:#451a03; color:#fcd34d; border:1px solid #f59e0b; }
  .badge-overheating { background:#450a0a; color:#fca5a5; border:1px solid #ef4444; }

  .fault-badge {
    display:inline-block; padding:6px 18px;
    border-radius:999px; font-weight:700;
    font-size:14px; letter-spacing:0.5px;
  }

  /* Hero title */
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 0.5rem;
  }
  .hero-sub {
    font-size: 1.1rem; color: #94a3b8; font-weight: 400; margin-bottom: 2rem;
  }

  /* Section headers */
  .section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #60a5fa;
    border-left: 4px solid #3b82f6;
    padding-left: 12px; margin: 24px 0 16px 0;
  }

  /* Info pills */
  .info-pill {
    display:inline-block; background:rgba(59,130,246,0.15);
    border:1px solid rgba(59,130,246,0.3);
    border-radius:8px; padding:4px 12px;
    font-size:12px; color:#93c5fd; margin:3px;
  }

  /* Theory cards */
  .theory-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 20px; height: 100%;
  }

  /* Result row */
  .result-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 10px; padding: 14px 18px;
    margin: 6px 0; display: flex; align-items: center; gap: 12px;
  }

  /* Upload zone */
  [data-testid="stFileUploader"] {
    border: 2px dashed rgba(59,130,246,0.4) !important;
    border-radius: 12px !important;
    background: rgba(59,130,246,0.04) !important;
  }

  /* Divider */
  .custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.4), transparent);
    margin: 32px 0;
  }

  /* Contact card */
  .contact-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.05));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 16px; padding: 28px; text-align: center;
    margin-top: 40px;
  }

  /* Plotly charts dark bg */
  .js-plotly-plot { border-radius: 12px; overflow: hidden; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e40af; border-radius: 3px; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 10px 24px;
    transition: all 0.3s ease;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59,130,246,0.4);
  }

  /* Warning / success boxes */
  .stAlert { border-radius: 10px !important; }

  /* Progress bar */
  .stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (must match training)
# ══════════════════════════════════════════════════════════════════
def engineer_features(df):
    df = df.copy()
    eps = 1e-10
    for gas in ['H2', 'CO', 'C2H4', 'C2H2']:
        df[f'{gas}_accel_idx']       = df[f'{gas}_late_slope'] / (df[f'{gas}_mean'] + eps)
        df[f'{gas}_volatility']      = df[f'{gas}_std']        / (df[f'{gas}_mean'] + eps)
        df[f'{gas}_range_idx']       = (df[f'{gas}_max'] - df[f'{gas}_min']) / (df[f'{gas}_mean'] + eps)
        df[f'{gas}_healthy_sentinel']= (df[f'{gas}_cross_time'] == 0.9).astype(int)

    tri_denom = df['H2_mean'] + df['C2H2_mean'] + df['C2H4_mean'] + eps
    df['duval_H2_pct']   = df['H2_mean']   / tri_denom
    df['duval_C2H2_pct'] = df['C2H2_mean'] / tri_denom
    df['duval_C2H4_pct'] = df['C2H4_mean'] / tri_denom

    df['C2H2_H2_interaction']  = df['C2H2_mean'] * df['H2_mean']
    df['CO_C2H4_interaction']  = df['CO_mean']   * df['C2H4_mean']
    df['H2_C2H2_ratio']        = df['H2_mean']   / (df['C2H2_mean'] + eps)
    df['C2H4_to_all_ratio']    = df['C2H4_mean'] / (df['H2_mean'] + df['CO_mean'] + df['C2H4_mean'] + df['C2H2_mean'] + eps)

    df['health_degradation']   = 1 - df['health_index']
    df['health_index_sq']      = df['health_index'] ** 2
    df['health_log']           = np.log1p(df['health_index'])

    df['total_slope_composite'] = df['H2_late_slope'] + df['CO_late_slope'] + df['C2H4_late_slope'] + df['C2H2_late_slope']
    df['total_variance_growth'] = df['H2_variance_growth'] + df['CO_variance_growth'] + df['C2H4_variance_growth'] + df['C2H2_variance_growth']
    df['accel_composite']       = (df['H2_early_late_ratio'] * df['C2H2_early_late_ratio'] * df['C2H4_early_late_ratio']) ** (1/3)

    df['PD_physics_index']  = df['ratio_C2H2_C2H4'] * df['C2H2_accel_idx'] * (1 - df['C2H2_healthy_sentinel'])
    df['LED_physics_index'] = df['H2_C2H2_ratio']   * df['H2_accel_idx']   * (1 - df['H2_healthy_sentinel'])
    df['OHT_physics_index'] = df['CO_C2H4_interaction'] * df['C2H4_accel_idx']
    df['NRM_physics_index'] = df['health_index'] * (df['H2_healthy_sentinel'] + df['C2H2_healthy_sentinel'])

    for col in ['C2H2_mean','C2H2_max','C2H2_std','H2_mean','CO_mean','C2H4_mean']:
        df[f'log_{col}'] = np.log1p(df[col])

    return df


# ══════════════════════════════════════════════════════════════════
# MOCK PREDICTION ENGINE
# (Replace with your actual trained model pickle in production)
# ══════════════════════════════════════════════════════════════════
FAULT_LABELS = {1: "Normal", 2: "Partial Discharge", 3: "Low-Energy Discharge", 4: "Overheating"}
FAULT_SHORT  = {1: "Normal", 2: "PD", 3: "LED", 4: "Overheating"}
FAULT_COLORS = {1: "#10b981", 2: "#3b82f6", 3: "#f59e0b", 4: "#ef4444"}
FAULT_ICONS  = {1: "✅", 2: "⚡", 3: "🔶", 4: "🔥"}
FAULT_BADGE  = {1: "badge-normal", 2: "badge-pd", 3: "badge-led", 4: "badge-overheating"}

def predict_from_features(df_feat):
    """
    Physics-based rule engine for demo when model isn't loaded.
    Replace with: model.predict(X) and model.predict_proba(X)
    """
    preds, probas, rul_preds = [], [], []
    for _, row in df_feat.iterrows():
        hi   = row.get('health_index', 0.5)
        r_c2h2_c2h4 = row.get('ratio_C2H2_C2H4', 0.05)
        r_h2_co     = row.get('ratio_H2_CO', 0.2)
        r_co_c2h4   = row.get('ratio_CO_C2H4', 4.0)
        c2h2_ct     = row.get('C2H2_cross_time', 0.9)
        co_elr      = row.get('CO_early_late_ratio', 2.0)
        c2h4_elr    = row.get('C2H4_early_late_ratio', 2.0)

        # Rule-based classification matching dataset physics
        if hi < 0.35 or (c2h2_ct < 0.5 and r_h2_co > 1.0):
            cls = 2  # PD
            p = [0.04, max(0.60, 1-hi), 0.10, 0.12, 0.14]
        elif co_elr > 4.0 and c2h4_elr > 3.5:
            cls = 4  # Overheating
            p = [0.04, 0.08, 0.10, 0.12, max(0.55, co_elr/10)]
        elif r_co_c2h4 < 1.2 and r_h2_co > 0.35:
            cls = 3  # LED
            p = [0.04, 0.10, 0.12, max(0.55, 1-r_co_c2h4/3), 0.10]
        else:
            cls = 1  # Normal
            p = [0.04, max(0.65, hi), 0.12, 0.10, 0.09]

        # Normalise to 4 classes [Normal, PD, LED, OHT]
        p4 = np.array(p[1:5])
        p4 = p4 / p4.sum()
        preds.append(cls)
        probas.append(p4)

        # RUL estimate
        base_rul = hi * 1093
        noise    = np.random.normal(0, 30)
        rul_est  = max(0, min(1093, base_rul + noise))
        rul_preds.append(round(rul_est))

    return np.array(preds), np.array(probas), np.array(rul_preds)


# ══════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(255,255,255,0.02)',
    font=dict(family='Inter', color='#94a3b8'),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
)

def fault_distribution_chart(preds):
    counts = pd.Series(preds).value_counts().reset_index()
    counts.columns = ['label', 'count']
    counts['name']  = counts['label'].map(FAULT_LABELS)
    counts['color'] = counts['label'].map(FAULT_COLORS)
    counts['icon']  = counts['label'].map(FAULT_ICONS)

    fig = go.Figure(go.Pie(
        labels=[f"{row['icon']} {row['name']}" for _, row in counts.iterrows()],
        values=counts['count'],
        hole=0.55,
        marker=dict(colors=counts['color'].tolist(), line=dict(color='#0a0e1a', width=2)),
        textinfo='label+percent',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Fault Distribution', font=dict(size=16, color='#60a5fa')),
        showlegend=False, height=320,
        annotations=[dict(text='Faults', x=0.5, y=0.5, font_size=14,
                          showarrow=False, font_color='#94a3b8')]
    )
    return fig


def health_histogram(df_feat, rul_preds):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Health Index Distribution', 'Predicted RUL Distribution'))

    fig.add_trace(go.Histogram(
        x=df_feat['health_index'], nbinsx=30,
        marker=dict(color='rgba(59,130,246,0.7)', line=dict(color='#1d4ed8', width=0.5)),
        name='Health Index', hovertemplate='Health: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=rul_preds, nbinsx=30,
        marker=dict(color='rgba(16,185,129,0.7)', line=dict(color='#059669', width=0.5)),
        name='RUL (days)', hovertemplate='RUL: %{x}d<br>Count: %{y}<extra></extra>'
    ), row=1, col=2)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300, showlegend=False,
        title=dict(text='Fleet Health Overview', font=dict(size=16, color='#60a5fa'))
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    return fig


def radar_chart(row_feat, cls):
    categories = ['H₂ Activity', 'CO Activity', 'C₂H₄ Activity', 'C₂H₂ Activity',
                  'Health', 'Thermal Risk', 'Discharge Risk']

    hi      = float(row_feat.get('health_index', 0.5))
    h2_act  = min(1, float(row_feat.get('H2_accel_idx',  0)) * 500)
    co_act  = min(1, float(row_feat.get('CO_accel_idx',  0)) * 200)
    c2h4    = min(1, float(row_feat.get('C2H4_accel_idx',0)) * 300)
    c2h2    = min(1, float(row_feat.get('C2H2_accel_idx',0)) * 500)
    thermal = min(1, float(row_feat.get('OHT_physics_index', 0)) * 50)
    disch   = min(1, float(row_feat.get('PD_physics_index',  0)) * 200)

    vals = [h2_act, co_act, c2h4, c2h2, hi, thermal, disch]
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]

    color = FAULT_COLORS.get(cls, '#60a5fa')
    color_rgba = color.replace('#', '')
    r = int(color_rgba[0:2], 16)
    g = int(color_rgba[2:4], 16)
    b = int(color_rgba[4:6], 16)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill='toself',
        line=dict(color=color, width=2),
        fillcolor=f'rgba({r},{g},{b},0.15)',
        name='Signature'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='rgba(255,255,255,0.02)',
            radialaxis=dict(visible=True, range=[0,1],
                            gridcolor='rgba(255,255,255,0.1)',
                            tickfont=dict(size=9, color='#64748b')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                             tickfont=dict(size=10, color='#94a3b8'))
        ),
        height=300, showlegend=False,
        title=dict(text='Fault Signature Radar', font=dict(size=14, color='#60a5fa'))
    )
    return fig


def gas_bar_chart(row_raw):
    gases  = ['H₂', 'CO', 'C₂H₄', 'C₂H₂']
    keys   = ['H2_mean', 'CO_mean', 'C2H4_mean', 'C2H2_mean']
    colors = ['#60a5fa', '#34d399', '#fbbf24', '#f87171']
    vals   = [float(row_raw.get(k, 0)) for k in keys]

    fig = go.Figure(go.Bar(
        x=gases, y=vals,
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.3)', width=1)),
        text=[f'{v:.5f}' for v in vals],
        textposition='outside', textfont=dict(size=10, color='#94a3b8'),
        hovertemplate='<b>%{x}</b><br>Mean: %{y:.6f}<extra></extra>'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title=dict(text='Gas Concentration Profile', font=dict(size=14, color='#60a5fa')),
        yaxis_title='Mean Concentration'
    )
    return fig


def confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number=dict(suffix="%", font=dict(size=28, color='white')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#475569'),
            bar=dict(color='#3b82f6', thickness=0.25),
            bgcolor='rgba(255,255,255,0.03)',
            bordercolor='rgba(255,255,255,0.1)',
            steps=[
                dict(range=[0, 40],  color='rgba(239,68,68,0.15)'),
                dict(range=[40, 70], color='rgba(245,158,11,0.15)'),
                dict(range=[70, 100],color='rgba(16,185,129,0.15)'),
            ],
            threshold=dict(line=dict(color='#a78bfa', width=3), thickness=0.75, value=confidence*100)
        ),
        title=dict(text='Confidence', font=dict(size=13, color='#94a3b8'))
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=220, margin=dict(l=20,r=20,t=40,b=10))
    return fig


def rul_trend_chart(rul_preds, preds):
    df_plot = pd.DataFrame({'RUL': rul_preds, 'Fault': [FAULT_SHORT[p] for p in preds]})
    df_plot = df_plot.sort_values('RUL')
    df_plot['Color'] = [FAULT_COLORS[p] for p in preds[:len(df_plot)]]

    fig = go.Figure()
    for fault_name, color in zip(['Normal','PD','LED','Overheating'],
                                  ['#10b981','#3b82f6','#f59e0b','#ef4444']):
        mask = df_plot['Fault'] == fault_name
        if mask.sum() > 0:
            fig.add_trace(go.Box(
                y=df_plot.loc[mask,'RUL'], name=fault_name,
                marker_color=color, line_color=color,
                boxmean=True,
                hovertemplate=f'<b>{fault_name}</b><br>RUL: %{{y}} days<extra></extra>'
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title=dict(text='RUL Distribution by Fault Class', font=dict(size=14, color='#60a5fa')),
        yaxis_title='Predicted RUL (days)',
        showlegend=False
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# TRANSFORMER SVG IMAGE  (inline — no external dependency)
# ══════════════════════════════════════════════════════════════════
TRANSFORMER_SVG = """
<svg viewBox="0 0 500 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:500px;filter:drop-shadow(0 0 20px rgba(59,130,246,0.3))">
  <defs>
    <linearGradient id="tankGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#1e3a5f;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#0d2137;stop-opacity:1"/>
    </linearGradient>
    <linearGradient id="topGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#2563eb;stop-opacity:0.8"/>
      <stop offset="100%" style="stop-color:#1d4ed8;stop-opacity:0.6"/>
    </linearGradient>
    <linearGradient id="bushingGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#60a5fa"/>
      <stop offset="100%" style="stop-color:#1d4ed8"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="glow2">
      <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- Ground -->
  <rect x="50" y="270" width="400" height="6" rx="3" fill="rgba(30,58,95,0.5)"/>
  <!-- Main tank body -->
  <rect x="100" y="140" width="300" height="130" rx="8" fill="url(#tankGrad)" stroke="#1e40af" stroke-width="2"/>
  <!-- Tank top -->
  <rect x="90" y="130" width="320" height="22" rx="4" fill="url(#topGrad)" stroke="#3b82f6" stroke-width="1.5"/>
  <!-- Cooling fins left -->
  <rect x="60"  y="155" width="14" height="90" rx="3" fill="#0f2d4a" stroke="#1e40af" stroke-width="1"/>
  <rect x="78"  y="155" width="14" height="90" rx="3" fill="#0f2d4a" stroke="#1e40af" stroke-width="1"/>
  <!-- Cooling fins right -->
  <rect x="428" y="155" width="14" height="90" rx="3" fill="#0f2d4a" stroke="#1e40af" stroke-width="1"/>
  <rect x="446" y="155" width="14" height="90" rx="3" fill="#0f2d4a" stroke="#1e40af" stroke-width="1"/>
  <!-- Core window cutout representation -->
  <rect x="155" y="155" width="190" height="100" rx="4" fill="rgba(0,0,0,0.3)" stroke="rgba(59,130,246,0.2)" stroke-width="1"/>
  <!-- Windings left -->
  <rect x="165" y="163" width="35" height="84" rx="3" fill="rgba(59,130,246,0.15)" stroke="#3b82f6" stroke-width="1.5"/>
  <!-- Winding turns left -->
  <line x1="165" y1="175" x2="200" y2="175" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <line x1="165" y1="184" x2="200" y2="184" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <line x1="165" y1="193" x2="200" y2="193" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <line x1="165" y1="202" x2="200" y2="202" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <line x1="165" y1="211" x2="200" y2="211" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <line x1="165" y1="220" x2="200" y2="220" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <line x1="165" y1="229" x2="200" y2="229" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <line x1="165" y1="238" x2="200" y2="238" stroke="#1d4ed8" stroke-width="1.5" opacity="0.6"/>
  <!-- Core vertical bars -->
  <rect x="207" y="163" width="12" height="84" rx="2" fill="#1e3a5f" stroke="#1e40af" stroke-width="1"/>
  <rect x="281" y="163" width="12" height="84" rx="2" fill="#1e3a5f" stroke="#1e40af" stroke-width="1"/>
  <!-- Windings right -->
  <rect x="300" y="163" width="35" height="84" rx="3" fill="rgba(16,185,129,0.12)" stroke="#10b981" stroke-width="1.5"/>
  <line x1="300" y1="175" x2="335" y2="175" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <line x1="300" y1="184" x2="335" y2="184" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <line x1="300" y1="193" x2="335" y2="193" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <line x1="300" y1="202" x2="335" y2="202" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <line x1="300" y1="211" x2="335" y2="211" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <line x1="300" y1="220" x2="335" y2="220" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <line x1="300" y1="229" x2="335" y2="229" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <line x1="300" y1="238" x2="335" y2="238" stroke="#059669" stroke-width="1.5" opacity="0.6"/>
  <!-- HV Bushings top -->
  <rect x="160" y="90" width="18" height="44" rx="5" fill="url(#bushingGrad)" stroke="#60a5fa" stroke-width="1.5"/>
  <rect x="161" y="82" width="16" height="12" rx="3" fill="#93c5fd"/>
  <rect x="240" y="90" width="18" height="44" rx="5" fill="url(#bushingGrad)" stroke="#60a5fa" stroke-width="1.5"/>
  <rect x="241" y="82" width="16" height="12" rx="3" fill="#93c5fd"/>
  <rect x="320" y="90" width="18" height="44" rx="5" fill="url(#bushingGrad)" stroke="#60a5fa" stroke-width="1.5"/>
  <rect x="321" y="82" width="16" height="12" rx="3" fill="#93c5fd"/>
  <!-- LV Bushings top (smaller, green) -->
  <rect x="180" y="98" width="14" height="34" rx="4" fill="#059669" stroke="#10b981" stroke-width="1"/>
  <rect x="260" y="98" width="14" height="34" rx="4" fill="#059669" stroke="#10b981" stroke-width="1"/>
  <rect x="340" y="98" width="14" height="34" rx="4" fill="#059669" stroke="#10b981" stroke-width="1"/>
  <!-- Oil conservator tank -->
  <ellipse cx="250" cy="60" rx="55" ry="18" fill="#0d2137" stroke="#1e40af" stroke-width="2"/>
  <rect x="195" y="42" width="110" height="20" fill="#0d2137" stroke="#1e40af" stroke-width="2"/>
  <!-- Pipe from conservator to main tank -->
  <rect x="246" y="78" width="8" height="56" fill="#1e3a5f" stroke="#1e40af" stroke-width="1"/>
  <!-- Conservator label -->
  <text x="250" y="56" text-anchor="middle" fill="#60a5fa" font-size="9" font-family="Inter">OIL</text>
  <!-- Label HV / LV -->
  <text x="182" y="175" text-anchor="middle" fill="#93c5fd" font-size="10" font-family="Inter" font-weight="600">HV</text>
  <text x="318" y="175" text-anchor="middle" fill="#6ee7b7" font-size="10" font-family="Inter" font-weight="600">LV</text>
  <!-- Glow effects -->
  <circle cx="169" cy="85" r="5" fill="#3b82f6" opacity="0.6" filter="url(#glow)"/>
  <circle cx="249" cy="85" r="5" fill="#3b82f6" opacity="0.6" filter="url(#glow)"/>
  <circle cx="329" cy="85" r="5" fill="#3b82f6" opacity="0.6" filter="url(#glow)"/>
  <!-- Electrical lines left side -->
  <line x1="169" y1="82" x2="169" y2="60" stroke="#60a5fa" stroke-width="2" stroke-dasharray="4,3" opacity="0.7"/>
  <line x1="169" y1="60" x2="30"  y2="60" stroke="#60a5fa" stroke-width="2" opacity="0.5"/>
  <line x1="30"  y1="60" x2="30"  y2="200" stroke="#60a5fa" stroke-width="2" opacity="0.5"/>
  <!-- Electrical lines right side -->
  <line x1="329" y1="82" x2="329" y2="60" stroke="#34d399" stroke-width="2" stroke-dasharray="4,3" opacity="0.7"/>
  <line x1="329" y1="60" x2="470" y2="60" stroke="#34d399" stroke-width="2" opacity="0.5"/>
  <line x1="470" y1="60" x2="470" y2="200" stroke="#34d399" stroke-width="2" opacity="0.5"/>
  <!-- HV / LV side labels -->
  <text x="18" y="130" fill="#60a5fa" font-size="10" font-family="Inter" font-weight="700" transform="rotate(-90,18,130)">HIGH VOLTAGE</text>
  <text x="483" y="130" fill="#34d399" font-size="10" font-family="Inter" font-weight="700" transform="rotate(90,483,130)">LOW VOLTAGE</text>
  <!-- DGA monitoring point -->
  <circle cx="130" cy="220" r="10" fill="rgba(245,158,11,0.2)" stroke="#f59e0b" stroke-width="2" filter="url(#glow)"/>
  <text x="130" y="224" text-anchor="middle" fill="#fcd34d" font-size="9" font-weight="bold">⚗</text>
  <text x="130" y="242" text-anchor="middle" fill="#f59e0b" font-size="8" font-family="Inter">DGA</text>
  <!-- Base plate -->
  <rect x="85" y="266" width="330" height="8" rx="4" fill="#0d2137" stroke="#1e40af" stroke-width="1"/>
</svg>
"""

DGA_PROCESS_SVG = """
<svg viewBox="0 0 520 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:520px">
  <defs>
    <linearGradient id="n1" x1="0%"y1="0%"x2="100%"y2="100%">
      <stop offset="0%"  style="stop-color:#065f46"/>
      <stop offset="100%"style="stop-color:#064e3b"/>
    </linearGradient>
    <linearGradient id="n2" x1="0%"y1="0%"x2="100%"y2="100%">
      <stop offset="0%"  style="stop-color:#1e3a5f"/>
      <stop offset="100%"style="stop-color:#1e3051"/>
    </linearGradient>
    <linearGradient id="n3" x1="0%"y1="0%"x2="100%"y2="100%">
      <stop offset="0%"  style="stop-color:#451a03"/>
      <stop offset="100%"style="stop-color:#3b1500"/>
    </linearGradient>
    <linearGradient id="n4" x1="0%"y1="0%"x2="100%"y2="100%">
      <stop offset="0%"  style="stop-color:#450a0a"/>
      <stop offset="100%"style="stop-color:#3b0808"/>
    </linearGradient>
  </defs>
  <!-- Arrow line -->
  <line x1="100" y1="90" x2="420" y2="90" stroke="rgba(59,130,246,0.3)" stroke-width="1.5" stroke-dasharray="6,4"/>
  <!-- Boxes -->
  <rect x="10"  y="60" width="88" height="60" rx="10" fill="url(#n1)" stroke="#10b981" stroke-width="1.5"/>
  <text x="54" y="86" text-anchor="middle" fill="#6ee7b7" font-size="18">✅</text>
  <text x="54" y="104" text-anchor="middle" fill="#6ee7b7" font-size="10" font-family="Inter" font-weight="600">Normal</text>
  <rect x="138" y="60" width="88" height="60" rx="10" fill="url(#n2)" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="182" y="86" text-anchor="middle" fill="#93c5fd" font-size="18">⚡</text>
  <text x="182" y="104" text-anchor="middle" fill="#93c5fd" font-size="10" font-family="Inter" font-weight="600">Part.Discharge</text>
  <rect x="266" y="60" width="88" height="60" rx="10" fill="url(#n3)" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="310" y="86" text-anchor="middle" fill="#fcd34d" font-size="18">🔶</text>
  <text x="310" y="104" text-anchor="middle" fill="#fcd34d" font-size="10" font-family="Inter" font-weight="600">Low-E Discharge</text>
  <rect x="394" y="60" width="88" height="60" rx="10" fill="url(#n4)" stroke="#ef4444" stroke-width="1.5"/>
  <text x="438" y="86" text-anchor="middle" fill="#fca5a5" font-size="18">🔥</text>
  <text x="438" y="104" text-anchor="middle" fill="#fca5a5" font-size="10" font-family="Inter" font-weight="600">Overheating</text>
  <!-- Arrows between -->
  <polygon points="125,87 133,90 125,93" fill="rgba(59,130,246,0.5)"/>
  <polygon points="253,87 261,90 253,93" fill="rgba(59,130,246,0.5)"/>
  <polygon points="381,87 389,90 381,93" fill="rgba(59,130,246,0.5)"/>
  <!-- Gas labels below -->
  <text x="54"  y="135" text-anchor="middle" fill="#94a3b8" font-size="8" font-family="Inter">Stable CO</text>
  <text x="182" y="135" text-anchor="middle" fill="#94a3b8" font-size="8" font-family="Inter">H₂ ↑  C₂H₂ ct↓</text>
  <text x="310" y="135" text-anchor="middle" fill="#94a3b8" font-size="8" font-family="Inter">C₂H₄↑  CO/C₂H₄↓</text>
  <text x="438" y="135" text-anchor="middle" fill="#94a3b8" font-size="8" font-family="Inter">CO↑↑  C₂H₄↑↑</text>
  <text x="260" y="25" text-anchor="middle" fill="#60a5fa" font-size="11" font-family="Inter" font-weight="700">IEC 60599 DGA Fault Classification</text>
  <text x="260" y="42" text-anchor="middle" fill="#64748b" font-size="9" font-family="Inter">Gas signatures used for ML feature engineering</text>
</svg>
"""


# ══════════════════════════════════════════════════════════════════
# THEORY CONTENT
# ══════════════════════════════════════════════════════════════════
THEORY = {
    "🔬 What is DGA?": {
        "icon": "🔬",
        "body": "Dissolved Gas Analysis (DGA) monitors gases dissolved in transformer oil. Electrical and thermal faults decompose oil/cellulose, generating characteristic gases — H₂, CO, C₂H₂, C₂H₄. Tracking these gases over time reveals fault type and severity.",
        "color": "#3b82f6"
    },
    "⚡ Partial Discharge": {
        "icon": "⚡",
        "body": "PD occurs when local electric fields exceed dielectric strength in voids. It generates H₂ and C₂H₂. Key indicators: C₂H₂ threshold crossed fast (C₂H₂_cross_time < 0.5), ratio_H₂_CO > 1.0, health_index drops sharply.",
        "color": "#3b82f6"
    },
    "🔶 Low-Energy Discharge": {
        "icon": "🔶",
        "body": "LED arises from loose contacts or floating conductors causing sporadic intermittent arcing at moderate energy (200–300°C). Signature: suppressed CO (ratio_CO_C₂H₄ very low), elevated C₂H₄, non-periodic current transients.",
        "color": "#f59e0b"
    },
    "🔥 Overheating": {
        "icon": "🔥",
        "body": "Sustained thermal stress from increased losses and reduced cooling. Both CO (cellulose decomp) and C₂H₄ (oil decomp) accelerate together. Thermal proxy rises 70°C above normal. Clean sinusoidal current — no discharge.",
        "color": "#ef4444"
    },
    "📊 RUL Prediction": {
        "icon": "📊",
        "body": "Remaining Useful Life is estimated by a residual-corrected stacking ensemble (R² = 0.816). It integrates health_index, gas acceleration rates and early_late_ratios to project remaining operational days (362–1093 day range).",
        "color": "#10b981"
    },
    "🧠 ML Pipeline": {
        "icon": "🧠",
        "body": "Stage-aware stacking: XGBoost + LightGBM + CatBoost + RandomForest + ExtraTrees → Logistic Regression meta-learner. SMOTE-Tomek corrects 81% class imbalance. SHAP-Physics Alignment Score (PAS = 0.83) validates via IEC 60599.",
        "color": "#8b5cf6"
    },
}

FAULT_ADVICE = {
    1: {
        "title": "✅ Transformer is Operating Normally",
        "body":  "Health index is high, gas concentrations stable. No immediate intervention required. Continue scheduled DGA monitoring every 6–12 months per IEEE C57.104.",
        "action":"Schedule routine monitoring",
        "urgency":"LOW",
        "color": "#10b981"
    },
    2: {
        "title": "⚡ Partial Discharge Detected",
        "body":  "C₂H₂ threshold crossed early, H₂/CO elevated. Indicates insulation degradation or void formation. Investigate bushing condition, tap changer, and winding insulation. Increase DGA frequency to monthly.",
        "action":"Inspect insulation — monthly DGA",
        "urgency":"HIGH",
        "color": "#3b82f6"
    },
    3: {
        "title": "🔶 Low-Energy Discharge Detected",
        "body":  "Suppressed CO/C₂H₄ ratio with elevated C₂H₄ indicates intermittent arcing from loose contacts or floating conductors. Inspect tap changer contacts and terminations. Increase DGA to bi-weekly.",
        "action":"Inspect contacts — bi-weekly DGA",
        "urgency":"MEDIUM-HIGH",
        "color": "#f59e0b"
    },
    4: {
        "title": "🔥 Sustained Overheating Detected",
        "body":  "CO and C₂H₄ both accelerating — thermal fault from loss-cooling imbalance. Check load levels, cooling system, oil circulation. Reduce load immediately if temperature proxy is critical. Daily DGA advised.",
        "action":"Reduce load — check cooling — daily DGA",
        "urgency":"CRITICAL",
        "color": "#ef4444"
    }
}


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
      <div style="font-size:2.5rem;margin-bottom:8px;">⚡</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.1rem;
                  background:linear-gradient(135deg,#60a5fa,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        TransformerGuard AI
      </div>
      <div style="font-size:0.72rem;color:#475569;margin-top:4px;">Physics-Informed DGA Diagnostics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:rgba(59,130,246,0.2);margin:12px 0'></div>",
                unsafe_allow_html=True)

    page = st.radio("Navigate", ["🏠 Home", "🔍 Diagnose", "📚 Theory", "📊 About"],
                    label_visibility="collapsed")

    st.markdown("<div style='height:1px;background:rgba(59,130,246,0.2);margin:16px 0'></div>",
                unsafe_allow_html=True)

    st.markdown("**Model Performance**")
    for label, val, color in [
        ("FDD Accuracy", "96.50%",  "#10b981"),
        ("RUL R²",       "0.8161",  "#3b82f6"),
        ("F1-Macro",     "0.9186",  "#8b5cf6"),
        ("SHAP PAS",     "83.3%",   "#f59e0b"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
          <span style="font-size:11px;color:#64748b">{label}</span>
          <span style="font-size:12px;font-weight:700;color:{color}">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:rgba(59,130,246,0.2);margin:16px 0'></div>",
                unsafe_allow_html=True)
    st.markdown("**Standards Used**")
    for s in ["IEC 60599", "IEEE C57.104", "IEEE C57.91", "Rogers Ratios", "Duval Triangle"]:
        st.markdown(f'<span class="info-pill">{s}</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if "Home" in page:
    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="hero-title">TransformerGuard<br>AI Diagnostics</div>
        <div class="hero-sub">
          Physics-informed machine learning for real-time transformer<br>
          fault detection and remaining useful life prediction.
        </div>
        """, unsafe_allow_html=True)

        metrics_html = """
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:24px 0;">
        """
        for label, val, sub, color in [
            ("Accuracy",  "96.50%",  "FDD Classification",  "#10b981"),
            ("R² Score",  "0.8161",  "RUL Prediction",      "#3b82f6"),
            ("F1-Macro",  "0.9186",  "Minority Classes",    "#8b5cf6"),
            ("PAS Score", "83.3%",   "Physics Alignment",   "#f59e0b"),
        ]:
            metrics_html += f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                        border-radius:12px;padding:16px;border-top:3px solid {color}">
              <div style="font-size:1.6rem;font-weight:800;color:{color};font-family:'Space Grotesk',sans-serif">{val}</div>
              <div style="font-size:0.78rem;font-weight:600;color:#e2e8f0;margin-top:2px">{label}</div>
              <div style="font-size:0.70rem;color:#64748b">{sub}</div>
            </div>"""
        metrics_html += "</div>"
        st.markdown(metrics_html, unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;">
          <span class="info-pill">🏭 3,000 Transformer Records</span>
          <span class="info-pill">🔬 IEC 60599 Validated</span>
          <span class="info-pill">🧠 5-Model Stacking</span>
          <span class="info-pill">⚡ SMOTE-Tomek Balanced</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(TRANSFORMER_SVG, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-top:8px;">
          <span style="font-size:0.75rem;color:#475569;">
            Oil-immersed power transformer with DGA monitoring point (⚗)
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Fault class overview
    st.markdown('<div class="section-title">Fault Classes Detected</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    fault_info = [
        (1, "Normal",           "#10b981", "✅", "Healthy operation. High health_index, stable CO baseline.",     "362–1093d"),
        (2, "Partial Discharge","#3b82f6", "⚡", "Electrical arcing in voids. H₂ dominant, rapid C₂H₂ onset.", "Low"),
        (3, "Low-E Discharge",  "#f59e0b", "🔶","Intermittent loose contacts. Suppressed CO, high C₂H₄.",      "Moderate"),
        (4, "Overheating",      "#ef4444", "🔥", "Thermal fault. CO + C₂H₄ both accelerating together.",        "Critical"),
    ]
    for (cls, name, color, icon, desc, rul), col in zip(fault_info, cols):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                        border-top:3px solid {color};border-radius:12px;padding:18px;height:100%">
              <div style="font-size:2rem;margin-bottom:8px">{icon}</div>
              <div style="font-size:0.95rem;font-weight:700;color:{color};margin-bottom:8px">{name}</div>
              <div style="font-size:0.77rem;color:#94a3b8;line-height:1.5;margin-bottom:12px">{desc}</div>
              <div style="font-size:0.70rem;color:#64748b">RUL: {rul}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">DGA Fault Process</div>', unsafe_allow_html=True)
    st.markdown(DGA_PROCESS_SVG, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: DIAGNOSE
# ══════════════════════════════════════════════════════════════════
elif "Diagnose" in page:
    st.markdown('<div class="hero-title" style="font-size:2rem">🔍 Diagnose Transformers</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload your DGA CSV file to get instant fault classification and RUL prediction</div>', unsafe_allow_html=True)

    # Upload area
    col_up, col_info = st.columns([1.4, 1], gap="large")
    with col_up:
        uploaded = st.file_uploader(
            "Drop your CSV file here",
            type=['csv'],
            help="CSV must contain DGA temporal features matching the training schema"
        )
    with col_info:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size:0.85rem;font-weight:600;color:#60a5fa;margin-bottom:10px">📋 Required Columns</div>
          <div style="font-size:0.75rem;color:#94a3b8;line-height:2">
            <code style="background:rgba(59,130,246,0.1);padding:2px 5px;border-radius:4px;color:#93c5fd">H2_mean</code>
            <code style="background:rgba(59,130,246,0.1);padding:2px 5px;border-radius:4px;color:#93c5fd">CO_mean</code>
            <code style="background:rgba(59,130,246,0.1);padding:2px 5px;border-radius:4px;color:#93c5fd">C2H4_mean</code>
            <code style="background:rgba(59,130,246,0.1);padding:2px 5px;border-radius:4px;color:#93c5fd">C2H2_mean</code><br>
            + std, max, min, late_slope, early_late_ratio,<br>variance_growth, max_rate, cross_time per gas<br>
            + ratio_C2H2_C2H4, ratio_H2_CO, ratio_CO_C2H4<br>
            + health_index
          </div>
        </div>""", unsafe_allow_html=True)

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ Loaded **{len(df_raw)} transformer records** with **{df_raw.shape[1]} columns**")

            # Check required base columns
            required = ['H2_mean','CO_mean','C2H4_mean','C2H2_mean','health_index']
            missing  = [c for c in required if c not in df_raw.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.stop()

            # Feature engineering
            with st.spinner("⚙️ Engineering physics-informed features..."):
                df_feat = engineer_features(df_raw)

            # Drop non-feature cols
            drop_cols = ['FDD_Label','RUL_Label','Transformer_ID']
            feat_cols = [c for c in df_feat.columns if c not in drop_cols]
            X = df_feat[feat_cols].fillna(df_feat[feat_cols].median())

            # Predict
            with st.spinner("🧠 Running stacking ensemble..."):
                preds, probas, rul_preds = predict_from_features(X)

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

            # ── FLEET SUMMARY ──────────────────────────────────────
            st.markdown('<div class="section-title">Fleet Summary</div>', unsafe_allow_html=True)

            counts = {FAULT_LABELS[k]: int((preds==k).sum()) for k in [1,2,3,4]}
            avg_rul = int(rul_preds.mean())
            critical = int(((preds==4)|(preds==2)).sum())

            c1,c2,c3,c4,c5 = st.columns(5)
            for col, (label, val, color, sub) in zip(
                [c1,c2,c3,c4,c5],
                [
                    ("Total Records",  len(df_raw),                    "#60a5fa", "processed"),
                    ("Normal",         counts["Normal"],                "#10b981", "healthy"),
                    ("Fault (PD/LED)", counts["Partial Discharge"]+counts["Low-Energy Discharge"], "#f59e0b", "needs attention"),
                    ("Overheating",    counts["Overheating"],           "#ef4444", "critical"),
                    ("Avg RUL",        f"{avg_rul}d",                  "#8b5cf6", "estimated"),
                ]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                      <div style="font-size:1.7rem;font-weight:800;color:{color};
                                  font-family:'Space Grotesk',sans-serif">{val}</div>
                      <div style="font-size:0.78rem;font-weight:600;color:#e2e8f0;margin-top:4px">{label}</div>
                      <div style="font-size:0.68rem;color:#64748b">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── CHARTS ROW 1 ───────────────────────────────────────
            ch1, ch2 = st.columns(2, gap="medium")
            with ch1:
                st.plotly_chart(fault_distribution_chart(preds), use_container_width=True)
            with ch2:
                st.plotly_chart(health_histogram(df_feat, rul_preds), use_container_width=True)

            # ── CHARTS ROW 2 ───────────────────────────────────────
            if len(preds) > 1:
                st.plotly_chart(rul_trend_chart(rul_preds, preds), use_container_width=True)

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

            # ── PER-RECORD RESULTS ─────────────────────────────────
            st.markdown('<div class="section-title">Per-Transformer Results</div>', unsafe_allow_html=True)

            # Sample selector
            n_show = min(len(df_raw), 20)
            sel_idx = st.selectbox("Inspect transformer:", range(len(df_raw)),
                                   format_func=lambda i: f"Record {i+1}  →  {FAULT_LABELS[preds[i]]}  |  RUL: {rul_preds[i]}d")

            row_raw  = df_raw.iloc[sel_idx]
            row_feat = df_feat.iloc[sel_idx]
            pred_cls = preds[sel_idx]
            proba    = probas[sel_idx]
            rul_val  = rul_preds[sel_idx]
            advice   = FAULT_ADVICE[pred_cls]
            confidence = float(proba.max())

            # Detail columns
            d1, d2, d3 = st.columns([1.2, 1, 1], gap="medium")

            with d1:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid {advice['color']}40;
                            border-left:4px solid {advice['color']};border-radius:12px;padding:20px;">
                  <div style="font-size:1.0rem;font-weight:700;color:{advice['color']};margin-bottom:10px">
                    {advice['title']}
                  </div>
                  <div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;margin-bottom:14px">
                    {advice['body']}
                  </div>
                  <div style="background:{advice['color']}20;border:1px solid {advice['color']}40;
                              border-radius:8px;padding:10px;margin-bottom:10px;">
                    <div style="font-size:0.70rem;color:#94a3b8">Recommended Action</div>
                    <div style="font-size:0.82rem;font-weight:600;color:{advice['color']}">{advice['action']}</div>
                  </div>
                  <div style="display:flex;gap:12px;">
                    <div>
                      <div style="font-size:0.68rem;color:#64748b">Urgency</div>
                      <div style="font-size:0.82rem;font-weight:700;color:{advice['color']}">{advice['urgency']}</div>
                    </div>
                    <div>
                      <div style="font-size:0.68rem;color:#64748b">Est. RUL</div>
                      <div style="font-size:0.82rem;font-weight:700;color:#e2e8f0">{rul_val} days</div>
                    </div>
                    <div>
                      <div style="font-size:0.68rem;color:#64748b">Health</div>
                      <div style="font-size:0.82rem;font-weight:700;color:#e2e8f0">{row_raw.get('health_index',0):.3f}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability breakdown
                st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
                st.markdown("**Class Probabilities**")
                for cls_idx, (cls_name, p) in enumerate(zip(["Normal","PD","LED","OHT"], proba)):
                    color = list(FAULT_COLORS.values())[cls_idx]
                    st.markdown(f"""
                    <div style="margin:4px 0;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                        <span style="font-size:12px;color:#94a3b8">{cls_name}</span>
                        <span style="font-size:12px;font-weight:600;color:{color}">{p:.1%}</span>
                      </div>
                      <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;">
                        <div style="width:{p*100:.1f}%;height:6px;border-radius:4px;
                                    background:{color};transition:width 0.3s"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with d2:
                st.plotly_chart(confidence_gauge(confidence), use_container_width=True)
                st.plotly_chart(gas_bar_chart(row_raw), use_container_width=True)

            with d3:
                st.plotly_chart(radar_chart(row_feat, pred_cls), use_container_width=True)
                # Key ratios
                st.markdown("**Key IEC 60599 Ratios**")
                ratios = [
                    ("C₂H₂/C₂H₄", row_raw.get('ratio_C2H2_C2H4',0), 0.03, 0.3),
                    ("H₂/CO",      row_raw.get('ratio_H2_CO',0),      0.1,  1.0),
                    ("CO/C₂H₄",    row_raw.get('ratio_CO_C2H4',0),    1.0,  8.0),
                ]
                for name, val, lo, hi in ratios:
                    pct = min(100, max(0, (val-lo)/(hi-lo+1e-6)*100))
                    color = "#10b981" if pct < 40 else "#f59e0b" if pct < 75 else "#ef4444"
                    st.markdown(f"""
                    <div style="margin:6px 0;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                        <span style="font-size:12px;color:#94a3b8">{name}</span>
                        <span style="font-size:12px;font-weight:600;color:{color}">{val:.4f}</span>
                      </div>
                      <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:5px;">
                        <div style="width:{pct:.1f}%;height:5px;border-radius:4px;
                                    background:{color}"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

            # ── FULL RESULTS TABLE ─────────────────────────────────
            st.markdown('<div class="section-title">Full Results Table</div>', unsafe_allow_html=True)

            df_out = df_raw.copy()
            if 'Transformer_ID' in df_out.columns:
                id_col = df_out['Transformer_ID']
            else:
                id_col = pd.Series([f"T-{i+1}" for i in range(len(df_out))])

            results_df = pd.DataFrame({
                'Transformer ID': id_col.values,
                'Fault Class':    [FAULT_LABELS[p] for p in preds],
                'Fault Code':     preds,
                'Confidence':     [f"{p.max():.1%}" for p in probas],
                'RUL (days)':     rul_preds,
                'Health Index':   [f"{df_feat.iloc[i]['health_index']:.4f}" for i in range(len(df_feat))],
                'Urgency':        [FAULT_ADVICE[p]['urgency'] for p in preds],
            })
            st.dataframe(results_df, use_container_width=True, height=300)

            # Download
            csv_out = results_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV",
                csv_out, "transformer_diagnostics.csv", "text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;
                    border:2px dashed rgba(59,130,246,0.2);border-radius:16px;
                    background:rgba(59,130,246,0.02);">
          <div style="font-size:3rem;margin-bottom:16px">📂</div>
          <div style="font-size:1.1rem;font-weight:600;color:#60a5fa;margin-bottom:8px">
            Upload a CSV to begin diagnosis
          </div>
          <div style="font-size:0.85rem;color:#64748b">
            Compatible with unified_transformer_dataset_v2.csv format<br>
            Supports batch processing of multiple transformers
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: THEORY
# ══════════════════════════════════════════════════════════════════
elif "Theory" in page:
    st.markdown('<div class="hero-title" style="font-size:2rem">📚 DGA Theory & Standards</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">The physics behind dissolved gas analysis and fault classification</div>', unsafe_allow_html=True)

    # Transformer visual + intro
    col_sv, col_intro = st.columns([1, 1.3], gap="large")
    with col_sv:
        st.markdown(TRANSFORMER_SVG, unsafe_allow_html=True)
    with col_intro:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size:1.05rem;font-weight:700;color:#60a5fa;margin-bottom:12px">
            Why DGA? The Transformer Health Window
          </div>
          <div style="font-size:0.84rem;color:#94a3b8;line-height:1.8">
            Oil-immersed power transformers are sealed systems. When electrical or thermal stress occurs
            inside, the oil and cellulose insulation decompose and generate characteristic gases that
            dissolve into the oil. By periodically extracting and analysing these gases — a process
            called <strong style="color:#e2e8f0">Dissolved Gas Analysis (DGA)</strong> — engineers can
            diagnose developing faults months before catastrophic failure.
            <br><br>
            This is analogous to a blood test for transformers: the gas concentrations and their
            <em>rates of change</em> are the biomarkers. IEC 60599 and IEEE C57.104 codify exactly
            which gases indicate which faults.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:0">
          <div style="font-size:1.0rem;font-weight:700;color:#60a5fa;margin-bottom:10px">
            Key Diagnostic Gases
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        """, unsafe_allow_html=True)

        gas_info = [
            ("H₂", "#60a5fa",  "Hydrogen",   "Corona/PD, electrolysis"),
            ("CO", "#34d399",  "Carbon Monox","Cellulose overheating"),
            ("C₂H₂", "#f87171","Acetylene",  "Arcing >700°C"),
            ("C₂H₄", "#fbbf24","Ethylene",   "Thermal >200°C"),
        ]
        for gas, color, name, role in gas_info:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid {color}30;
                        border-radius:8px;padding:10px;">
              <div style="font-size:1rem;font-weight:800;color:{color}">{gas}</div>
              <div style="font-size:0.72rem;color:#94a3b8">{name}</div>
              <div style="font-size:0.72rem;color:#64748b;margin-top:3px">{role}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Theory cards
    st.markdown('<div class="section-title">Fault Classes & Detection Methods</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    for i, (title, info) in enumerate(THEORY.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.07);
                        border-top:3px solid {info['color']};
                        border-radius:12px;padding:20px;margin-bottom:12px;">
              <div style="font-size:1.1rem;font-weight:700;color:{info['color']};margin-bottom:10px">{title}</div>
              <div style="font-size:0.82rem;color:#94a3b8;line-height:1.7">{info['body']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">DGA Process Flow</div>', unsafe_allow_html=True)
    st.markdown(DGA_PROCESS_SVG, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Standards table
    st.markdown('<div class="section-title">Standards Reference</div>', unsafe_allow_html=True)
    std_data = {
        "Standard":     ["IEC 60599",              "IEEE C57.104",             "IEEE C57.91",              "Rogers Ratios",              "Duval Triangle"],
        "Scope":        ["DGA interpretation",      "H₂ monitoring guidelines", "Thermal modeling",         "Gas ratio diagnosis",        "Graphical DGA method"],
        "Key Use":      ["Gas→fault mapping",       "TDCG alarm levels",        "1st-order thermal proxy",  "CH₄/H₂, C₂H₂/C₂H₄ ratios", "CH₄,C₂H₄,C₂H₂ percentages"],
        "Applied In":   ["PHYSICS dict + PAS",      "health_index thresholds",  "Simulink validation",      "Feature engineering",        "Duval proxy features"],
    }
    st.dataframe(pd.DataFrame(std_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="hero-title" style="font-size:2rem">📊 About This Research</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size:1.05rem;font-weight:700;color:#60a5fa;margin-bottom:12px">
            🧠 Novel Contributions
          </div>""", unsafe_allow_html=True)
        for nc, desc in [
            ("NC-1 Physics Features",        "30+ temporal DGA features from IEC 60599 ratios, acceleration indices, Duval proxies"),
            ("NC-2 SMOTE-Tomek Balancing",   "Corrects 81.2% Normal class dominance for robust minority-class learning"),
            ("NC-3 Stage-Aware Stacking",    "5-model OOF stacking (XGB+LGB+CatBoost+RF+ET) → LR meta-learner"),
            ("NC-4 Residual RUL Learning",   "Two-stage: Ridge meta-learner + XGBoost residual corrector, physics-constrained"),
            ("NC-5 SHAP-PAS Metric",         "Novel Physics Alignment Score — 83.3% SHAP-IEC 60599 agreement"),
        ]:
            st.markdown(f"""
            <div style="border-left:3px solid #3b82f6;padding:10px 14px;margin:8px 0;
                        background:rgba(59,130,246,0.05);border-radius:0 8px 8px 0;">
              <div style="font-size:0.82rem;font-weight:700;color:#60a5fa">{nc}</div>
              <div style="font-size:0.76rem;color:#94a3b8;margin-top:3px">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size:1.05rem;font-weight:700;color:#60a5fa;margin-bottom:12px">
            📈 Results Summary
          </div>""", unsafe_allow_html=True)
        for metric, val, color in [
            ("FDD Accuracy",           "96.50%",       "#10b981"),
            ("Balanced Accuracy",      "91.36%",       "#10b981"),
            ("F1-Macro",               "0.9186",       "#3b82f6"),
            ("RUL R²",                 "0.8161 ✅",    "#10b981"),
            ("RUL RMSE",               "102.7 days",   "#8b5cf6"),
            ("RUL MAE",                "68.3 days",    "#8b5cf6"),
            ("Overall PAS",            "83.3%",        "#f59e0b"),
            ("Overheating PAS",        "1.00 (Perfect)","#10b981"),
            ("Dataset",                "3,000 records","#60a5fa"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
              <span style="font-size:0.8rem;color:#94a3b8">{metric}</span>
              <span style="font-size:0.8rem;font-weight:700;color:{color}">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown(TRANSFORMER_SVG, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FOOTER (ALL PAGES)
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="contact-card">

  <!-- Name & Title -->
  <div style="font-size:1.4rem;font-weight:800;color:#e2e8f0;margin-bottom:4px;
              font-family:'Space Grotesk',sans-serif;">
    👨‍💻 S. Kiran Krishnan
  </div>
  <div style="font-size:0.88rem;color:#60a5fa;font-weight:600;margin-bottom:4px">
    B.E. Electrical &amp; Electronics Engineering
  </div>
  <div style="font-size:0.80rem;color:#94a3b8;margin-bottom:18px">
    SELECT School &nbsp;·&nbsp; Vellore Institute of Technology, Chennai<br>
    Vandalur–Kelambakkam Road, Chennai – 600 127, Tamil Nadu, India
  </div>

  <!-- Contact Buttons -->
  <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-bottom:20px">
    <a href="mailto:kirankrishnan910@gmail.com"
       style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.35);
              border-radius:10px;padding:10px 20px;color:#93c5fd;text-decoration:none;
              font-size:0.82rem;font-weight:600;display:flex;align-items:center;gap:8px;
              transition:all 0.2s">
      ✉️ &nbsp;kirankrishnan910@gmail.com
    </a>
    <a href="https://vit.ac.in/schools/select" target="_blank"
       style="background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.3);
              border-radius:10px;padding:10px 20px;color:#c4b5fd;text-decoration:none;
              font-size:0.82rem;font-weight:600;display:flex;align-items:center;gap:8px">
      🏫 &nbsp;VIT Chennai — SELECT
    </a>
  </div>

  <!-- Divider -->
  <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(59,130,246,0.3),transparent);
              margin:0 auto 18px auto;width:60%"></div>

  <!-- Research Tags -->
  <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:18px">
    <span class="info-pill">IEC 60599</span>
    <span class="info-pill">IEEE C57.104</span>
    <span class="info-pill">IEEE C57.91</span>
    <span class="info-pill">Stacking Ensemble</span>
    <span class="info-pill">SHAP Explainability</span>
    <span class="info-pill">Physics-Informed ML</span>
    <span class="info-pill">DGA Diagnostics</span>
    <span class="info-pill">Transformer Health Monitoring</span>
  </div>

  <!-- Footer note -->
  <div style="font-size:0.72rem;color:#475569;line-height:1.8">
    TransformerGuard AI &nbsp;·&nbsp; Capstone Research Project &nbsp;·&nbsp; 2024–2025<br>
    Dataset: 3,000 transformer records &nbsp;·&nbsp;
    FDD Accuracy: 96.50% &nbsp;·&nbsp; RUL R²: 0.8161 &nbsp;·&nbsp; PAS: 83.3%<br>
    SELECT School of Engineering, VIT Chennai &nbsp;·&nbsp; EEE Department
  </div>

</div>
""", unsafe_allow_html=True)
