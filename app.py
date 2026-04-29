import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import json
from feature_engineering import FEATURE_COLS, build_features

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyCity Profit Intelligence",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 100%; }

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(255,165,0,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(255,100,50,0.05) 0%, transparent 50%);
}

[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.5rem !important;
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(255,165,0,0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(255,165,0,0.08);
}
[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.45) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

[data-testid="stTabs"] [role="tablist"] {
    background: #13131f;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500;
    color: rgba(255,255,255,0.45) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.2s ease;
    border: none !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: rgba(255,255,255,0.8) !important;
    background: rgba(255,255,255,0.05) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #ff9500, #ff5f00) !important;
    color: #000 !important;
    font-weight: 600 !important;
}

.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.25s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff9500 0%, #ff5f00 100%) !important;
    color: #000 !important;
    box-shadow: 0 4px 20px rgba(255,149,0,0.3);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,149,0,0.45) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #ff9500, #ff5f00) !important;
}
[data-testid="stSlider"] label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.78rem !important;
    font-family: 'DM Sans', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

[data-testid="stSelectbox"] label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSelectbox"] > div > div {
    background: #13131f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
}

[data-testid="stInfo"] {
    background: rgba(255,149,0,0.08) !important;
    border: 1px solid rgba(255,149,0,0.2) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.75) !important;
}
[data-testid="stSuccess"] {
    background: rgba(0,210,120,0.1) !important;
    border: 1px solid rgba(0,210,120,0.25) !important;
    border-radius: 10px !important;
}
[data-testid="stWarning"] {
    background: rgba(255,200,0,0.08) !important;
    border: 1px solid rgba(255,200,0,0.2) !important;
    border-radius: 10px !important;
}

[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

[data-testid="stNumberInput"] > div > div > input {
    background: #13131f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #fff !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(255,149,0,0.4); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout (NO title key — set separately per chart) ──
BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='rgba(255,255,255,0.7)', size=12),
    title_font=dict(family='Syne', size=16, color='#ffffff'),
    title_x=0.01,
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickfont=dict(size=11)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickfont=dict(size=11)),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
    colorway=['#ff9500','#ff5f00','#ffc300','#ff8c00','#e65c00','#ffb347'],
    margin=dict(l=20, r=20, t=48, b=20),
)

def apply_theme(fig, title='', xaxis_title='', yaxis_title=''):
    """Apply dark theme to any plotly figure. Titles set separately to avoid conflicts."""
    fig.update_layout(**BASE_LAYOUT)
    if title:
        fig.update_layout(title=title)
    if xaxis_title:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)
    return fig

ORANGE_SCALE = [[0,'#1a0a00'],[0.25,'#7a3000'],[0.5,'#cc5500'],[0.75,'#ff8000'],[1,'#ffb347']]

# ── Data loaders ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('SkyCity Auckland Restaurants & Bars.csv')
    df = build_features(df)
    df['TotalNetProfit'] = (df['InStoreNetProfit'] + df['UberEatsNetProfit'] +
                            df['DoorDashNetProfit'] + df['SelfDeliveryNetProfit'])
    return df

@st.cache_resource
def load_model():
    model  = joblib.load('models/XGBoost.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

df            = load_data()
model, scaler = load_model()
X_all         = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.5rem; font-weight:800;
                    background:linear-gradient(135deg,#ff9500,#ff5f00);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1;'>
            SKYCITY
        </div>
        <div style='font-family:DM Sans,sans-serif; font-size:0.68rem;
                    color:rgba(255,255,255,0.35); letter-spacing:0.25em;
                    text-transform:uppercase; margin-top:4px;'>
            Auckland · Profit Intelligence
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.06); margin:1rem 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.68rem; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:0.15em; font-family:DM Sans;'>Overview</p>", unsafe_allow_html=True)

    total_restaurants = df['RestaurantID'].nunique()
    avg_profit_side   = df['TotalNetProfit'].mean()
    top_segment       = df.groupby('Segment')['TotalNetProfit'].mean().idxmax()
    top_cuisine       = df.groupby('CuisineType')['TotalNetProfit'].mean().idxmax()

    st.markdown(f"""
    <div style='display:flex; flex-direction:column; gap:8px; margin-bottom:1rem;'>
        <div style='background:rgba(255,149,0,0.08); border:1px solid rgba(255,149,0,0.15); border-radius:10px; padding:10px 14px;'>
            <div style='font-size:0.65rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em; font-family:DM Sans;'>Restaurants</div>
            <div style='font-size:1.3rem; font-weight:700; color:#fff; font-family:Syne;'>{total_restaurants:,}</div>
        </div>
        <div style='background:rgba(255,149,0,0.08); border:1px solid rgba(255,149,0,0.15); border-radius:10px; padding:10px 14px;'>
            <div style='font-size:0.65rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em; font-family:DM Sans;'>Avg Monthly Profit</div>
            <div style='font-size:1.3rem; font-weight:700; color:#ff9500; font-family:Syne;'>${avg_profit_side:,.0f}</div>
        </div>
        <div style='background:rgba(255,149,0,0.08); border:1px solid rgba(255,149,0,0.15); border-radius:10px; padding:10px 14px;'>
            <div style='font-size:0.65rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em; font-family:DM Sans;'>Top Segment</div>
            <div style='font-size:1rem; font-weight:600; color:#fff; font-family:Syne;'>{top_segment}</div>
        </div>
        <div style='background:rgba(255,149,0,0.08); border:1px solid rgba(255,149,0,0.15); border-radius:10px; padding:10px 14px;'>
            <div style='font-size:0.65rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em; font-family:DM Sans;'>Top Cuisine</div>
            <div style='font-size:1rem; font-weight:600; color:#fff; font-family:Syne;'>{top_cuisine}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.65rem; color:rgba(255,255,255,0.2); text-align:center; font-family:DM Sans;'>Powered by XGBoost · SHAP · Streamlit</p>", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:0.5rem 0 1.5rem;'>
    <div style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; color:#fff; line-height:1.1;'>
        Profit Intelligence
        <span style='background:linear-gradient(135deg,#ff9500,#ff5f00);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Dashboard
        </span>
    </div>
    <div style='font-family:DM Sans,sans-serif; font-size:0.88rem; color:rgba(255,255,255,0.4); margin-top:6px;'>
        Multi-channel restaurant analytics · SkyCity Auckland
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tabs = st.tabs(["  📊  Analytics  ", "  🔮  Predict  ", "  ⚙️  Simulate  ", "  🎯  Optimize  ", "  📋  Report  "])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — ANALYTICS
# ══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Restaurants",  f"{df['RestaurantID'].nunique():,}")
    k2.metric("Avg Monthly Profit", f"${df['TotalNetProfit'].mean():,.0f}")
    k3.metric("Avg Order Value",    f"${df['AOV'].mean():.2f}")
    k4.metric("Avg Monthly Orders", f"{df['MonthlyOrders'].mean():,.0f}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.box(df, x='Segment', y='TotalNetProfit', color='Segment',
                     points='outliers',
                     color_discrete_sequence=['#ff9500','#ff5f00','#ffc300','#e65c00','#ffb347'])
        apply_theme(fig, title='Net Profit Distribution by Segment')
        fig.update_traces(marker=dict(opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        channel_avg = {
            'In-Store':      df['InStoreShare'].mean(),
            'Uber Eats':     df['UE_share'].mean(),
            'DoorDash':      df['DD_share'].mean(),
            'Self-Delivery': df['SD_share'].mean()
        }
        fig4 = px.pie(values=list(channel_avg.values()),
                      names=list(channel_avg.keys()),
                      hole=0.55,
                      color_discrete_sequence=['#ff9500','#ff5f00','#ffc300','#cc4400'])
        apply_theme(fig4, title='Average Channel Mix')
        fig4.update_traces(textfont=dict(family='DM Sans', size=12),
                           marker=dict(line=dict(color='#0a0a0f', width=2)))
        st.plotly_chart(fig4, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        cuisine_df = (df.groupby('CuisineType')['TotalNetProfit']
                        .mean().reset_index()
                        .sort_values('TotalNetProfit', ascending=True))
        fig2 = px.bar(cuisine_df, x='TotalNetProfit', y='CuisineType',
                      orientation='h',
                      color='TotalNetProfit', color_continuous_scale=ORANGE_SCALE)
        apply_theme(fig2, title='Avg Profit by Cuisine Type')
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    with c4:
        fig3 = px.scatter(df, x='CommissionRate', y='TotalNetProfit',
                          color='Segment', size='MonthlyOrders',
                          hover_name='RestaurantName',
                          color_discrete_sequence=['#ff9500','#ff5f00','#ffc300','#e65c00','#ffb347'])
        apply_theme(fig3, title='Commission Rate vs Net Profit')
        st.plotly_chart(fig3, use_container_width=True)

    num_cols = ['AOV','COGSRate','OPEXRate','CommissionRate',
                'InStoreShare','UE_share','DD_share','SD_share','TotalNetProfit']
    corr = df[num_cols].corr().round(2)
    fig5 = px.imshow(corr, color_continuous_scale='RdYlGn',
                     text_auto=True, zmin=-1, zmax=1)
    apply_theme(fig5, title='Feature Correlation Heatmap')
    fig5.update_traces(textfont=dict(size=11))
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne; font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:0.3rem;'>
        Configure Restaurant Parameters
    </div>
    <div style='font-family:DM Sans; font-size:0.82rem; color:rgba(255,255,255,0.4); margin-bottom:1.5rem;'>
        Adjust the sliders to match your restaurant profile and click Predict.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<p style='font-size:0.72rem; color:rgba(255,149,0,0.8); text-transform:uppercase; letter-spacing:0.15em; font-weight:600; margin-bottom:1rem;'>🏪 Restaurant Profile</p>", unsafe_allow_html=True)
        aov            = st.slider("Avg Order Value ($)",         25.0, 60.0, 38.0, 0.5)
        monthly_orders = st.slider("Monthly Orders",              200,  3000, 1000, 50)
        growth         = st.slider("Growth Factor",               0.95, 1.10, 1.02, 0.01)
        cuisine        = st.selectbox("Cuisine Type",             df['CuisineType'].unique())
        segment        = st.selectbox("Segment",                  df['Segment'].unique())
        subregion      = st.selectbox("Subregion",                df['Subregion'].unique())

    with c2:
        st.markdown("<p style='font-size:0.72rem; color:rgba(255,149,0,0.8); text-transform:uppercase; letter-spacing:0.15em; font-weight:600; margin-bottom:1rem;'>💰 Cost Parameters</p>", unsafe_allow_html=True)
        cogs_rate     = st.slider("COGS Rate",                    0.15, 0.45, 0.25, 0.01)
        opex_rate     = st.slider("OPEX Rate",                    0.15, 0.60, 0.35, 0.01)
        commission    = st.slider("Commission Rate",              0.15, 0.40, 0.28, 0.01)
        delivery_cost = st.slider("Self-Delivery Cost/Order ($)", 0.5,  6.0,  2.5,  0.1)
        delivery_km   = st.slider("Delivery Radius (km)",         3,    18,   10,   1)

    with c3:
        st.markdown("<p style='font-size:0.72rem; color:rgba(255,149,0,0.8); text-transform:uppercase; letter-spacing:0.15em; font-weight:600; margin-bottom:1rem;'>📡 Channel Mix</p>", unsafe_allow_html=True)
        instore_share = st.slider("In-Store Share",      0.0, 1.0, 0.25, 0.01)
        ue_share      = st.slider("Uber Eats Share",     0.0, 1.0, 0.45, 0.01)
        dd_share      = st.slider("DoorDash Share",      0.0, 1.0, 0.15, 0.01)
        sd_share      = st.slider("Self-Delivery Share", 0.0, 1.0, 0.15, 0.01)
        total         = instore_share + ue_share + dd_share + sd_share
        delta_val     = round(total - 1.0, 3)
        color         = "#00d278" if abs(delta_val) <= 0.05 else "#ff4444"

        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                    border-radius:10px; padding:12px 16px; margin-top:8px;'>
            <div style='font-size:0.68rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em;'>Total Share</div>
            <div style='font-size:1.4rem; font-weight:700; color:{color}; font-family:Syne;'>{total:.2f}</div>
            <div style='font-size:0.72rem; color:{"rgba(0,210,120,0.7)" if abs(delta_val)<=0.05 else "rgba(255,68,68,0.7)"};'>
                {"✓ Balanced" if abs(delta_val)<=0.05 else f"Δ {delta_val:+.3f} from 1.0"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if abs(total - 1.0) > 0.05:
            st.warning("Channel shares should sum to 1.0")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if st.button("🔮  Predict Monthly Profit", type="primary"):
        le_cuisine   = joblib.load('models/le_CuisineType.pkl')
        le_segment   = joblib.load('models/le_Segment.pkl')
        le_subregion = joblib.load('models/le_Subregion.pkl')

        instore_orders = monthly_orders * instore_share
        ue_orders      = monthly_orders * ue_share
        dd_orders      = monthly_orders * dd_share
        sd_orders      = monthly_orders * sd_share
        instore_rev    = instore_orders * aov
        ue_rev         = ue_orders      * aov
        dd_rev         = dd_orders      * aov
        sd_rev         = sd_orders      * aov
        total_rev      = instore_rev + ue_rev + dd_rev + sd_rev

        row = {
            'AOV': aov, 'MonthlyOrders': monthly_orders, 'GrowthFactor': growth,
            'COGSRate': cogs_rate, 'OPEXRate': opex_rate, 'CommissionRate': commission,
            'DeliveryRadiusKM': delivery_km, 'DeliveryCostPerOrder': delivery_cost,
            'InStoreShare': instore_share, 'UE_share': ue_share,
            'DD_share': dd_share, 'SD_share': sd_share,
            'InStoreRevShare':  instore_rev / (total_rev + 1),
            'UERevShare':       ue_rev      / (total_rev + 1),
            'DDRevShare':       dd_rev      / (total_rev + 1),
            'SDRevShare':       sd_rev      / (total_rev + 1),
            'TotalCostRate':    cogs_rate + opex_rate,
            'Commission_UE':    commission * ue_share,
            'Commission_DD':    commission * dd_share,
            'DeliveryCost_SD':  delivery_cost * sd_share,
            'GrowthAdjOrders':  monthly_orders * growth,
            'GrowthAdjRevenue': total_rev * growth,
            'InStorePPO': instore_rev * (1 - cogs_rate - opex_rate) / (instore_orders + 1),
            'UE_PPO':     ue_rev * (1 - cogs_rate - opex_rate - commission) / (ue_orders + 1),
            'DD_PPO':     dd_rev * (1 - cogs_rate - opex_rate - commission) / (dd_orders + 1),
            'SD_PPO':     sd_rev * (1 - cogs_rate - opex_rate) / (sd_orders + 1),
            'CuisineType_enc': le_cuisine.transform([cuisine])[0],
            'Segment_enc':     le_segment.transform([segment])[0],
            'Subregion_enc':   le_subregion.transform([subregion])[0],
        }
        X_pred     = pd.DataFrame([row])[FEATURE_COLS]
        prediction = model.predict(X_pred)[0]

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(255,149,0,0.12),rgba(255,95,0,0.08));
                    border:1px solid rgba(255,149,0,0.3); border-radius:16px;
                    padding:1.8rem 2rem; margin:1.2rem 0; text-align:center;'>
            <div style='font-size:0.72rem; color:rgba(255,149,0,0.7); text-transform:uppercase;
                        letter-spacing:0.2em; font-family:DM Sans;'>Predicted Monthly Net Profit</div>
            <div style='font-size:3rem; font-weight:800; color:#ff9500; font-family:Syne;
                        line-height:1.1; margin:8px 0;'>${prediction:,.2f}</div>
            <div style='font-size:0.78rem; color:rgba(255,255,255,0.35); font-family:DM Sans;'>
                Based on XGBoost model · {len(FEATURE_COLS)} features
            </div>
        </div>
        """, unsafe_allow_html=True)

        breakdown = {
            'In-Store':      instore_rev * (1 - cogs_rate - opex_rate),
            'Uber Eats':     ue_rev * (1 - cogs_rate - opex_rate - commission),
            'DoorDash':      dd_rev * (1 - cogs_rate - opex_rate - commission),
            'Self-Delivery': sd_rev * (1 - cogs_rate - opex_rate) - (sd_orders * delivery_cost),
        }
        fig_bd = px.bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                        color=list(breakdown.values()),
                        color_continuous_scale=ORANGE_SCALE,
                        labels={'x': 'Channel', 'y': 'Estimated Profit ($)'})
        apply_theme(fig_bd, title='Estimated Profit Breakdown by Channel')
        fig_bd.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bd, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — SIMULATE
# ══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne; font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:0.3rem;'>
        Scenario Simulation
    </div>
    <div style='font-family:DM Sans; font-size:0.82rem; color:rgba(255,255,255,0.4); margin-bottom:1.5rem;'>
        Sweep any parameter across a range and observe predicted profit response.
    </div>
    """, unsafe_allow_html=True)

    ctrl_col, chart_col = st.columns([1, 2])

    with ctrl_col:
        st.markdown("<p style='font-size:0.72rem; color:rgba(255,149,0,0.8); text-transform:uppercase; letter-spacing:0.15em; font-weight:600;'>Sweep Settings</p>", unsafe_allow_html=True)
        sweep_param = st.selectbox("Parameter", [
            'CommissionRate','InStoreShare','UE_share','DD_share',
            'SD_share','DeliveryCostPerOrder','GrowthFactor','COGSRate','OPEXRate'
        ])
        sweep_min = st.number_input("Min value", value=0.10, step=0.01)
        sweep_max = st.number_input("Max value", value=0.50, step=0.01)
        n_points  = st.slider("Resolution (points)", 10, 100, 40)

    with chart_col:
        X_base    = X_all.copy()
        vals      = np.linspace(sweep_min, sweep_max, n_points)
        preds     = []
        for v in vals:
            X_mod = X_base.copy()
            X_mod[sweep_param] = v
            preds.append(model.predict(X_mod).mean())

        baseline  = model.predict(X_base).mean()
        preds_arr = np.array(preds)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=vals, y=preds_arr, mode='lines',
            line=dict(color='#ff9500', width=2.5),
            fill='tozeroy', fillcolor='rgba(255,149,0,0.07)',
            name='Predicted Avg Profit'
        ))
        fig_sim.add_hline(
            y=baseline, line_dash='dash', line_color='rgba(255,255,255,0.3)',
            annotation_text=f"Baseline: ${baseline:,.0f}",
            annotation_font=dict(color='rgba(255,255,255,0.5)', size=11)
        )
        apply_theme(fig_sim,
                    title=f'Profit Response · {sweep_param}',
                    xaxis_title=sweep_param,
                    yaxis_title='Avg Predicted Net Profit ($)')
        st.plotly_chart(fig_sim, use_container_width=True)

        if sweep_param == 'CommissionRate':
            zero_rates = [v for v, p in zip(vals, preds) if p < 0]
            if zero_rates:
                st.warning(f"⚠️ Profit turns negative above commission ≈ {min(zero_rates):.2f}")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZE
# ══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        st.markdown("""
        <div style='font-family:Syne; font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:0.3rem;'>
            Optimal Channel Mix Finder
        </div>
        <div style='font-family:DM Sans; font-size:0.82rem; color:rgba(255,255,255,0.4); margin-bottom:1rem;'>
            Grid-searches all valid channel share combinations to find the profit-maximising mix.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍  Run Optimization", type="primary"):
            with st.spinner("Searching channel combinations..."):
                from itertools import product as iproduct
                best_profit = -np.inf
                best_mix    = {}
                results_opt = []
                shares      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                X_opt       = X_all.mean().to_frame().T

                for ins, ue, dd in iproduct(shares, shares, shares):
                    sd = round(1.0 - ins - ue - dd, 2)
                    if sd < 0 or sd > 1: continue
                    if abs(ins + ue + dd + sd - 1.0) > 0.01: continue
                    X_c = X_opt.copy()
                    X_c['InStoreShare']    = ins
                    X_c['UE_share']        = ue
                    X_c['DD_share']        = dd
                    X_c['SD_share']        = sd
                    X_c['Commission_UE']   = X_c['CommissionRate'] * ue
                    X_c['Commission_DD']   = X_c['CommissionRate'] * dd
                    X_c['DeliveryCost_SD'] = X_c['DeliveryCostPerOrder'] * sd
                    p = model.predict(X_c[FEATURE_COLS])[0]
                    results_opt.append({'InStore': ins, 'UberEats': ue,
                                        'DoorDash': dd, 'SelfDelivery': sd, 'Profit': p})
                    if p > best_profit:
                        best_profit = p
                        best_mix    = {'In-Store': ins, 'Uber Eats': ue,
                                       'DoorDash': dd, 'Self-Delivery': sd}

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(0,210,120,0.1),rgba(0,180,100,0.06));
                        border:1px solid rgba(0,210,120,0.25); border-radius:14px;
                        padding:1.2rem 1.5rem; margin:0.8rem 0;'>
                <div style='font-size:0.68rem; color:rgba(0,210,120,0.7); text-transform:uppercase; letter-spacing:0.15em; font-family:DM Sans;'>Best Predicted Profit</div>
                <div style='font-size:2rem; font-weight:800; color:#00d278; font-family:Syne;'>${best_profit:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(pd.DataFrame([best_mix]), use_container_width=True, hide_index=True)

            res_df  = pd.DataFrame(results_opt).sort_values('Profit', ascending=False).head(30)
            fig_opt = px.scatter(res_df, x='InStore', y='UberEats',
                                 size='Profit', color='Profit',
                                 color_continuous_scale=ORANGE_SCALE,
                                 hover_data=['DoorDash','SelfDelivery','Profit'])
            apply_theme(fig_opt, title='Top Channel Mix Combinations')
            st.plotly_chart(fig_opt, use_container_width=True)

    with col_opt2:
        st.markdown("""
        <div style='font-family:Syne; font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:0.3rem;'>
            Feature Importance (SHAP)
        </div>
        <div style='font-family:DM Sans; font-size:0.82rem; color:rgba(255,255,255,0.4); margin-bottom:1rem;'>
            Which features drive profit predictions the most?
        </div>
        """, unsafe_allow_html=True)
        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_all.sample(200, random_state=42))
            mean_shap   = pd.DataFrame({
                'Feature':    FEATURE_COLS,
                'Importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('Importance', ascending=True).tail(15)

            fig_shap = px.bar(mean_shap, x='Importance', y='Feature', orientation='h',
                              color='Importance', color_continuous_scale=ORANGE_SCALE)
            apply_theme(fig_shap, title='Top 15 Features by SHAP Value')
            fig_shap.update_traces(marker_line_width=0)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

# ══════════════════════════════════════════════════════════════════
# TAB 5 — REPORT
# ══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne; font-size:1.1rem; font-weight:700; color:#fff; margin-bottom:1rem;'>
        Executive Summary Report
    </div>
    """, unsafe_allow_html=True)

    try:
        with open('models/results.json') as f:
            results = json.load(f)
        st.markdown("<p style='font-size:0.72rem; color:rgba(255,149,0,0.8); text-transform:uppercase; letter-spacing:0.15em; font-weight:600; margin-bottom:0.8rem;'>Model Performance</p>", unsafe_allow_html=True)
        perf_df = pd.DataFrame(results).T.reset_index()
        perf_df.columns = ['Model','RMSE','R²','MAE','CV-R²']
        perf_df = perf_df.round(4)
        st.dataframe(
            perf_df.style
                .highlight_max(subset=['R²','CV-R²'], color='rgba(0,210,120,0.25)')
                .highlight_min(subset=['RMSE','MAE'], color='rgba(0,210,120,0.25)'),
            use_container_width=True, hide_index=True
        )
    except:
        st.info("Run train_models.py first to generate results.json")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.72rem; color:rgba(255,149,0,0.8); text-transform:uppercase; letter-spacing:0.15em; font-weight:600; margin-bottom:0.8rem;'>Key Insights</p>", unsafe_allow_html=True)

    avg_profit = df['TotalNetProfit'].mean()
    best_seg   = df.groupby('Segment')['TotalNetProfit'].mean().idxmax()
    worst_seg  = df.groupby('Segment')['TotalNetProfit'].mean().idxmin()
    best_cui   = df.groupby('CuisineType')['TotalNetProfit'].mean().idxmax()
    corr_comm  = df['CommissionRate'].corr(df['TotalNetProfit'])

    ins1, ins2, ins3 = st.columns(3)
    ins1.metric("Avg Monthly Profit",     f"${avg_profit:,.2f}")
    ins2.metric("Commission Correlation", f"{corr_comm:.3f}")
    ins3.metric("Best Cuisine",           best_cui)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    seg_df = (df.groupby('Segment')['TotalNetProfit']
                .mean().reset_index()
                .sort_values('TotalNetProfit', ascending=False))
    fig_seg = px.bar(seg_df, x='Segment', y='TotalNetProfit',
                     color='TotalNetProfit', color_continuous_scale=ORANGE_SCALE)
    apply_theme(fig_seg, title='Average Profit by Segment')
    fig_seg.update_traces(marker_line_width=0)
    st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown(f"""
    <div style='background:#13131f; border:1px solid rgba(255,255,255,0.07);
                border-radius:14px; padding:1.5rem 2rem; margin-top:0.5rem;'>
        <table style='width:100%; border-collapse:collapse; font-family:DM Sans; font-size:0.85rem;'>
            <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
                <td style='padding:10px 0; color:rgba(255,255,255,0.4);'>Avg Monthly Net Profit</td>
                <td style='padding:10px 0; color:#fff; font-weight:500; text-align:right;'>${avg_profit:,.2f}</td>
            </tr>
            <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
                <td style='padding:10px 0; color:rgba(255,255,255,0.4);'>Best Performing Segment</td>
                <td style='padding:10px 0; color:#ff9500; font-weight:500; text-align:right;'>{best_seg}</td>
            </tr>
            <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
                <td style='padding:10px 0; color:rgba(255,255,255,0.4);'>Lowest Performing Segment</td>
                <td style='padding:10px 0; color:#ff4444; font-weight:500; text-align:right;'>{worst_seg}</td>
            </tr>
            <tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>
                <td style='padding:10px 0; color:rgba(255,255,255,0.4);'>Best Performing Cuisine</td>
                <td style='padding:10px 0; color:#ff9500; font-weight:500; text-align:right;'>{best_cui}</td>
            </tr>
            <tr>
                <td style='padding:10px 0; color:rgba(255,255,255,0.4);'>Commission ↔ Profit Correlation</td>
                <td style='padding:10px 0; color:#fff; font-weight:500; text-align:right;'>{corr_comm:.3f}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)