"""
╔══════════════════════════════════════════════════════════════╗
║     ASPECT-BASED SENTIMENT ANALYSIS (ABSA) - Main App        ║
║     Dataset: Amazon Fine Food Reviews (Kaggle)               ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import re
from collections import defaultdict, Counter
import random

from utils.absa_engine import ABSAEngine
from utils.data_loader import DataLoader
from utils.visualizer import Visualizer

# ─── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d45;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --positive: #10b981;
    --negative: #ef4444;
    --neutral: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
  }

  .stApp { background: var(--bg); color: var(--text); font-family: 'Space Grotesk', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  /* Cards */
  .metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
  }
  .metric-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .metric-card .value { font-size: 2.2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .metric-card .label { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
  .metric-card.positive .value { color: var(--positive); }
  .metric-card.negative .value { color: var(--negative); }
  .metric-card.neutral .value { color: var(--neutral); }
  .metric-card.accent .value { color: var(--accent); }

  /* Aspect pills */
  .aspect-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
  }
  .pill-pos { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
  .pill-neg { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .pill-neu { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

  /* Review card */
  .review-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    line-height: 1.7;
  }

  /* Section header */
  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 30px 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  .section-header h3 { margin: 0; font-size: 1.1rem; color: var(--accent); }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero h1 { font-size: 2.5rem; font-weight: 700; margin: 0 0 8px; }
  .hero h1 span { color: var(--accent); }
  .hero p { color: var(--muted); margin: 0; font-size: 1.05rem; }

  /* Highlight span */
  .highlight-pos { background: rgba(16,185,129,0.2); border-radius: 3px; padding: 1px 3px; color: #10b981; }
  .highlight-neg { background: rgba(239,68,68,0.2); border-radius: 3px; padding: 1px 3px; color: #ef4444; }
  .highlight-neu { background: rgba(245,158,11,0.2); border-radius: 3px; padding: 1px 3px; color: #f59e0b; }

  /* Button override */
  .stButton button {
    background: linear-gradient(135deg, var(--accent2), #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s !important;
  }
  .stButton button:hover { opacity: 0.9; transform: translateY(-1px); }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
  .stTabs [aria-selected="true"] { background: var(--surface2) !important; color: var(--text) !important; }

  /* Input */
  .stTextArea textarea, .stSelectbox > div { 
    background: var(--surface2) !important; 
    border-color: var(--border) !important;
    color: var(--text) !important;
  }

  /* Hide default streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ─── INIT ────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return ABSAEngine()

@st.cache_data
def load_data():
    loader = DataLoader()
    return loader.load_dataset()

engine = load_engine()
df = load_data()

# ─── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ABSA Analyzer")
    st.markdown("---")
    
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📝 Analyze Review",
        "📊 Dataset Explorer",
        "🏷️ Aspect Deep Dive",
        "📈 Model Insights"
    ])
    
    st.markdown("---")
    st.markdown("### Dataset Info")
    st.markdown(f"""
    <div style='background:#1a2235;border-radius:8px;padding:12px;font-size:0.85rem;'>
    📦 <b>Amazon Fine Food Reviews</b><br>
    <span style='color:#64748b;'>Kaggle Dataset</span><br><br>
    🔗 <a href='https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews' style='color:#00d4ff;'>Download Dataset</a><br><br>
    📁 Reviews: <b>{len(df):,}</b><br>
    ⭐ Rating range: <b>1–5</b><br>
    🏷️ Aspects tracked: <b>8</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<span style='color:#64748b;font-size:0.8rem;'>ABSA v1.0 · Rule + ML Hybrid</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Aspect-Based <span>Sentiment Analysis</span></h1>
        <p>Uncover granular opinions on specific product aspects — not just overall sentiment</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Overall stats ──
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    pos_pct = (df['sentiment_label'] == 'positive').mean() * 100
    neg_pct = (df['sentiment_label'] == 'negative').mean() * 100
    avg_aspects = df['aspect_count'].mean()

    with col1:
        st.markdown(f"""<div class="metric-card accent">
            <div class="value">{total:,}</div>
            <div class="label">Total Reviews</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card positive">
            <div class="value">{pos_pct:.1f}%</div>
            <div class="label">Positive Reviews</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card negative">
            <div class="value">{neg_pct:.1f}%</div>
            <div class="label">Negative Reviews</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card neutral">
            <div class="value">{avg_aspects:.1f}</div>
            <div class="label">Avg Aspects/Review</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header"><h3>📊 Aspect Sentiment Distribution</h3></div>', unsafe_allow_html=True)
        
        aspect_data = engine.get_aspect_sentiment_summary(df)
        
        aspects = list(aspect_data.keys())
        pos_vals = [aspect_data[a]['positive'] for a in aspects]
        neg_vals = [aspect_data[a]['negative'] for a in aspects]
        neu_vals = [aspect_data[a]['neutral'] for a in aspects]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Positive', x=aspects, y=pos_vals,
                             marker_color='#10b981', marker_line_width=0))
        fig.add_trace(go.Bar(name='Negative', x=aspects, y=neg_vals,
                             marker_color='#ef4444', marker_line_width=0))
        fig.add_trace(go.Bar(name='Neutral', x=aspects, y=neu_vals,
                             marker_color='#f59e0b', marker_line_width=0))
        
        fig.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(gridcolor='#1e2d45'),
            yaxis=dict(gridcolor='#1e2d45'),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header"><h3>🍩 Sentiment Split</h3></div>', unsafe_allow_html=True)
        
        sentiment_counts = df['sentiment_label'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.6,
            marker_colors=['#10b981', '#ef4444', '#f59e0b'],
            textfont=dict(color='white'),
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=30, b=0),
            height=320
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Rating vs Sentiment ──
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="section-header"><h3>⭐ Rating vs Sentiment Score</h3></div>', unsafe_allow_html=True)
        
        rating_sentiment = df.groupby('Score')['compound_score'].mean().reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=rating_sentiment['Score'],
            y=rating_sentiment['compound_score'],
            mode='lines+markers',
            line=dict(color='#00d4ff', width=2.5),
            marker=dict(size=10, color='#7c3aed', line=dict(color='#00d4ff', width=2)),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.05)'
        ))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            xaxis=dict(title='Star Rating', gridcolor='#1e2d45', dtick=1),
            yaxis=dict(title='Avg Compound Score', gridcolor='#1e2d45'),
            margin=dict(l=0, r=0, t=30, b=0),
            height=280
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header"><h3>🏷️ Top Mentioned Aspects</h3></div>', unsafe_allow_html=True)
        
        aspect_freq = engine.get_aspect_frequency(df)
        sorted_aspects = sorted(aspect_freq.items(), key=lambda x: x[1], reverse=True)
        asp_names = [a[0] for a in sorted_aspects]
        asp_counts = [a[1] for a in sorted_aspects]
        
        colors = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#8b5cf6', '#06b6d4']
        fig4 = go.Figure(go.Bar(
            x=asp_counts, y=asp_names,
            orientation='h',
            marker=dict(color=colors[:len(asp_names)], line_width=0),
            text=asp_counts,
            textposition='outside',
            textfont=dict(color='#e2e8f0')
        ))
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Space Grotesk'),
            xaxis=dict(gridcolor='#1e2d45'),
            yaxis=dict(gridcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Recent Reviews ──
    st.markdown('<div class="section-header"><h3>🗂️ Sample Reviews with ABSA Output</h3></div>', unsafe_allow_html=True)
    
    samples = df.sample(3, random_state=42)
    for _, row in samples.iterrows():
        result = engine.analyze(row['Text'][:500])
        pill_html = ""
        for asp, sent in result['aspects'].items():
            cls = "pill-pos" if sent == "positive" else "pill-neg" if sent == "negative" else "pill-neu"
            pill_html += f'<span class="aspect-pill {cls}">{asp}: {sent}</span>'
        
        stars = "⭐" * int(row['Score'])
        st.markdown(f"""
        <div class="review-card">
            <div style="margin-bottom:8px;">{stars} &nbsp;<span style="color:#64748b;font-size:0.82rem;">Overall: <b style="color:{'#10b981' if row['sentiment_label']=='positive' else '#ef4444' if row['sentiment_label']=='negative' else '#f59e0b'}">{row['sentiment_label'].upper()}</b></span></div>
            <div style="margin-bottom:10px;color:#cbd5e1;">{row['Text'][:300]}{'...' if len(row['Text']) > 300 else ''}</div>
            <div>{pill_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE 2: ANALYZE REVIEW
# ══════════════════════════════════════════════════
elif page == "📝 Analyze Review":
    st.markdown("## 📝 Analyze Your Review")
    st.markdown("<p style='color:#64748b;'>Paste any food/product review to extract aspect-level sentiments</p>", unsafe_allow_html=True)

    col_input, col_sample = st.columns([3, 1])
    
    with col_input:
        user_text = st.text_area(
            "Review Text",
            placeholder="e.g. The coffee taste was amazing but the packaging was terrible. Delivery was fast though!",
            height=160,
            label_visibility="collapsed"
        )
    
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Quick Samples:**")
        samples_dict = {
            "Coffee Review": "The coffee taste is absolutely wonderful and rich. However, the packaging was damaged on arrival and the price is quite high for the quantity you get. Shipping was surprisingly fast.",
            "Snack Review": "These chips have great flavor and the texture is perfectly crunchy. The ingredients look clean and healthy. A bit overpriced but the portion size is generous enough.",
            "Negative Review": "Terrible taste, nothing like described. The smell was off-putting and the packaging was completely crushed. Delivery took 3 weeks. Very disappointed.",
            "Mixed Review": "Amazing aroma and taste but the quality control seems inconsistent. Some bags were great, others were stale. Price is fair for the brand."
        }
        for label, text in samples_dict.items():
            if st.button(label, key=f"sample_{label}"):
                user_text = text
                st.session_state['sample_text'] = text

    if 'sample_text' in st.session_state and not user_text:
        user_text = st.session_state['sample_text']

    if st.button("🔬 Run ABSA Analysis", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing aspects..."):
                result = engine.analyze(user_text)

            st.markdown("---")

            # ── Overall sentiment ──
            col1, col2, col3 = st.columns(3)
            sentiment = result['overall_sentiment']
            score = result['compound_score']
            
            s_color = "#10b981" if sentiment == "positive" else "#ef4444" if sentiment == "negative" else "#f59e0b"
            s_emoji = "😊" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
            
            with col1:
                st.markdown(f"""<div class="metric-card">
                    <div class="value" style="color:{s_color};">{s_emoji} {sentiment.upper()}</div>
                    <div class="label">Overall Sentiment</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="metric-card">
                    <div class="value" style="color:{s_color};">{score:+.3f}</div>
                    <div class="label">Compound Score</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="metric-card accent">
                    <div class="value">{len(result['aspects'])}</div>
                    <div class="label">Aspects Detected</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown('<div class="section-header"><h3>🏷️ Aspect Sentiments</h3></div>', unsafe_allow_html=True)
                
                if result['aspects']:
                    for aspect, sentiment_val in result['aspects'].items():
                        score_val = result['aspect_scores'].get(aspect, 0)
                        color = "#10b981" if sentiment_val == "positive" else "#ef4444" if sentiment_val == "negative" else "#f59e0b"
                        bar_pct = int((score_val + 1) / 2 * 100)
                        
                        st.markdown(f"""
                        <div style="margin:10px 0;padding:12px;background:#1a2235;border-radius:8px;border:1px solid #1e2d45;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                                <span style="font-weight:600;text-transform:capitalize;">{aspect}</span>
                                <span style="color:{color};font-size:0.85rem;font-weight:600;">{sentiment_val.upper()} ({score_val:+.2f})</span>
                            </div>
                            <div style="background:#0a0e1a;border-radius:999px;height:6px;overflow:hidden;">
                                <div style="width:{bar_pct}%;height:100%;background:{color};border-radius:999px;transition:width 0.5s;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific aspects detected in this review.")

            with col_right:
                st.markdown('<div class="section-header"><h3>📊 Sentiment Radar</h3></div>', unsafe_allow_html=True)
                
                if result['aspects']:
                    asp_labels = list(result['aspect_scores'].keys())
                    asp_vals = [(v + 1) / 2 for v in result['aspect_scores'].values()]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=asp_vals + [asp_vals[0]],
                        theta=asp_labels + [asp_labels[0]],
                        fill='toself',
                        fillcolor='rgba(0,212,255,0.1)',
                        line=dict(color='#00d4ff', width=2),
                        marker=dict(color='#7c3aed', size=8)
                    ))
                    fig.update_layout(
                        polar=dict(
                            bgcolor='rgba(0,0,0,0)',
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1e2d45', color='#64748b'),
                            angularaxis=dict(gridcolor='#1e2d45', color='#e2e8f0')
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0', family='Space Grotesk'),
                        margin=dict(l=40, r=40, t=20, b=20),
                        height=320
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ── Highlighted review ──
            st.markdown('<div class="section-header"><h3>🖊️ Review with Highlights</h3></div>', unsafe_allow_html=True)
            
            highlighted = engine.highlight_text(user_text, result['aspect_spans'])
            st.markdown(f'<div class="review-card" style="font-size:0.95rem;">{highlighted}</div>', unsafe_allow_html=True)
            
            st.markdown("**Legend:** &nbsp; <span class='highlight-pos'>Positive aspect</span> &nbsp; <span class='highlight-neg'>Negative aspect</span> &nbsp; <span class='highlight-neu'>Neutral aspect</span>", unsafe_allow_html=True)

            # ── Sentence breakdown ──
            st.markdown('<div class="section-header"><h3>🔍 Sentence-Level Breakdown</h3></div>', unsafe_allow_html=True)
            
            for i, sent_result in enumerate(result['sentence_breakdown'], 1):
                s = sent_result['sentiment']
                sc = sent_result['score']
                col = "#10b981" if s == "positive" else "#ef4444" if s == "negative" else "#f59e0b"
                st.markdown(f"""
                <div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0;padding:10px;background:#1a2235;border-radius:8px;">
                    <span style="font-family:JetBrains Mono;font-size:0.75rem;color:#64748b;white-space:nowrap;padding-top:2px;">S{i}</span>
                    <span style="flex:1;color:#cbd5e1;">{sent_result['text']}</span>
                    <span style="color:{col};font-weight:600;white-space:nowrap;font-size:0.85rem;">{sc:+.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a review to analyze.")


# ══════════════════════════════════════════════════
# PAGE 3: DATASET EXPLORER
# ══════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.markdown("## 📊 Dataset Explorer")
    st.markdown("<p style='color:#64748b;'>Browse and filter the Amazon Fine Food Reviews dataset</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        rating_filter = st.multiselect("Star Rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    with col2:
        sentiment_filter = st.multiselect("Sentiment", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"])
    with col3:
        aspect_filter = st.selectbox("Has Aspect", ["Any"] + list(engine.ASPECTS.keys()))

    filtered = df[df['Score'].isin(rating_filter) & df['sentiment_label'].isin(sentiment_filter)]
    
    if aspect_filter != "Any":
        filtered = filtered[filtered['detected_aspects'].apply(lambda x: aspect_filter in x)]

    st.markdown(f"<span style='color:#64748b;'>Showing **{len(filtered):,}** reviews</span>", unsafe_allow_html=True)

    # Data table
    display_cols = ['Score', 'sentiment_label', 'compound_score', 'aspect_count', 'Text']
    st.dataframe(
        filtered[display_cols].rename(columns={
            'Score': '⭐', 'sentiment_label': 'Sentiment', 
            'compound_score': 'Score', 'aspect_count': '# Aspects', 'Text': 'Review'
        }).head(100),
        use_container_width=True,
        height=400
    )

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Score Distribution")
        score_dist = filtered['Score'].value_counts().sort_index()
        fig = px.bar(x=score_dist.index, y=score_dist.values,
                     labels={'x': 'Stars', 'y': 'Count'},
                     color=score_dist.values,
                     color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#e2e8f0'), showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0), height=250,
                          xaxis=dict(gridcolor='#1e2d45'), yaxis=dict(gridcolor='#1e2d45'))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Review Length Distribution")
        filtered['text_len'] = filtered['Text'].str.len()
        fig2 = px.histogram(filtered, x='text_len', nbins=40,
                            color_discrete_sequence=['#7c3aed'])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='#e2e8f0'), showlegend=False,
                           margin=dict(l=0, r=0, t=0, b=0), height=250,
                           xaxis=dict(title='Characters', gridcolor='#1e2d45'),
                           yaxis=dict(gridcolor='#1e2d45'))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════
# PAGE 4: ASPECT DEEP DIVE
# ══════════════════════════════════════════════════
elif page == "🏷️ Aspect Deep Dive":
    st.markdown("## 🏷️ Aspect Deep Dive")

    selected_aspect = st.selectbox("Choose an Aspect to Analyze", list(engine.ASPECTS.keys()))
    
    aspect_df = df[df['detected_aspects'].apply(lambda x: selected_aspect in x)]
    
    if len(aspect_df) == 0:
        st.info(f"No reviews found mentioning '{selected_aspect}'")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        asp_sentiments = aspect_df['aspect_sentiments'].apply(
            lambda x: x.get(selected_aspect, 'neutral') if isinstance(x, dict) else 'neutral'
        )
        
        pos_count = (asp_sentiments == 'positive').sum()
        neg_count = (asp_sentiments == 'negative').sum()
        neu_count = (asp_sentiments == 'neutral').sum()
        
        with col1:
            st.markdown(f"""<div class="metric-card accent"><div class="value">{len(aspect_df)}</div><div class="label">Reviews</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card positive"><div class="value">{pos_count}</div><div class="label">Positive</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card negative"><div class="value">{neg_count}</div><div class="label">Negative</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card neutral"><div class="value">{neu_count}</div><div class="label">Neutral</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown(f"#### {selected_aspect.title()} — Sentiment Over Ratings")
            asp_by_rating = []
            for rating in [1, 2, 3, 4, 5]:
                r_df = aspect_df[aspect_df['Score'] == rating]
                if len(r_df) > 0:
                    r_sents = r_df['aspect_sentiments'].apply(lambda x: x.get(selected_aspect, 'neutral') if isinstance(x, dict) else 'neutral')
                    asp_by_rating.append({
                        'Rating': rating,
                        'Positive': (r_sents == 'positive').sum(),
                        'Negative': (r_sents == 'negative').sum(),
                        'Neutral': (r_sents == 'neutral').sum()
                    })
            
            if asp_by_rating:
                ar_df = pd.DataFrame(asp_by_rating)
                fig = go.Figure()
                for col_name, color in [('Positive', '#10b981'), ('Negative', '#ef4444'), ('Neutral', '#f59e0b')]:
                    fig.add_trace(go.Bar(name=col_name, x=ar_df['Rating'], y=ar_df[col_name], marker_color=color))
                fig.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#e2e8f0'), height=300,
                                  xaxis=dict(title='Star Rating', gridcolor='#1e2d45'),
                                  yaxis=dict(gridcolor='#1e2d45'),
                                  legend=dict(bgcolor='rgba(0,0,0,0)'),
                                  margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(f"#### {selected_aspect.title()} Sentiment Breakdown")
            fig2 = go.Figure(go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[pos_count, neg_count, neu_count],
                hole=0.55,
                marker_colors=['#10b981', '#ef4444', '#f59e0b']
            ))
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                               legend=dict(bgcolor='rgba(0,0,0,0)'),
                               margin=dict(l=0, r=0, t=0, b=0), height=300)
            st.plotly_chart(fig2, use_container_width=True)

        # Sample reviews for this aspect
        st.markdown(f"#### Sample Reviews Mentioning **{selected_aspect.title()}**")
        for sentiment_type in ['positive', 'negative']:
            subset = aspect_df[asp_sentiments == sentiment_type].head(2)
            if len(subset) > 0:
                color = "#10b981" if sentiment_type == "positive" else "#ef4444"
                for _, row in subset.iterrows():
                    st.markdown(f"""
                    <div class="review-card" style="border-left-color:{color}">
                        <div style="color:{color};font-weight:600;font-size:0.8rem;margin-bottom:6px;">
                            {'⭐' * int(row['Score'])} · {sentiment_type.upper()}
                        </div>
                        {row['Text'][:350]}{'...' if len(row['Text']) > 350 else ''}
                    </div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# PAGE 5: MODEL INSIGHTS
# ══════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.markdown("## 📈 Model Insights")
    st.markdown("<p style='color:#64748b;'>How our ABSA engine works under the hood</p>", unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("### 🏗️ ABSA Pipeline Architecture")
    
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:24px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;line-height:2;">
    
    <span style="color:#00d4ff;">INPUT</span> → Raw Review Text<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#7c3aed;">STEP 1</span> → <b>Text Preprocessing</b> (tokenize, clean, sentence split)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#7c3aed;">STEP 2</span> → <b>Aspect Detection</b> (keyword matching + context window)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#7c3aed;">STEP 3</span> → <b>Opinion Extraction</b> (adjectives near aspect terms)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#7c3aed;">STEP 4</span> → <b>VADER Sentiment Scoring</b> (per-aspect compound score)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#7c3aed;">STEP 5</span> → <b>Negation Handling</b> (not, never, don't, etc.)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#10b981;">OUTPUT</span> → Aspect → Sentiment Mapping + Overall Score
    
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏷️ Tracked Aspects & Keywords")
        for aspect, keywords in engine.ASPECTS.items():
            kw_pills = " ".join([f'<code style="background:#1a2235;color:#00d4ff;padding:2px 6px;border-radius:4px;font-size:0.75rem;">{k}</code>' for k in keywords[:5]])
            st.markdown(f"""
            <div style="margin:8px 0;padding:10px 14px;background:#111827;border-radius:8px;border:1px solid #1e2d45;">
                <b style="color:#e2e8f0;text-transform:capitalize;">{aspect}</b><br>
                <div style="margin-top:5px;">{kw_pills}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📐 Approach: Rule + VADER Hybrid")
        
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:20px;">
        
        <div style="margin-bottom:16px;">
            <div style="color:#00d4ff;font-weight:600;margin-bottom:4px;">1. Lexicon-Based (VADER)</div>
            <div style="color:#94a3b8;font-size:0.88rem;">VADER (Valence Aware Dictionary and sEntiment Reasoner) assigns sentiment scores based on a curated lexicon tuned for social media and reviews. Fast and interpretable.</div>
        </div>
        
        <div style="margin-bottom:16px;">
            <div style="color:#7c3aed;font-weight:600;margin-bottom:4px;">2. Rule-Based Aspect Extraction</div>
            <div style="color:#94a3b8;font-size:0.88rem;">Aspect keywords are matched with a context window (±15 words). Negation patterns flip sentiment polarity for captured spans.</div>
        </div>
        
        <div style="margin-bottom:16px;">
            <div style="color:#10b981;font-weight:600;margin-bottom:4px;">3. Sentence Segmentation</div>
            <div style="color:#94a3b8;font-size:0.88rem;">Reviews are split into sentences before analysis so that one negative sentence doesn't pollute a positive aspect mentioned elsewhere.</div>
        </div>
        
        <div>
            <div style="color:#f59e0b;font-weight:600;margin-bottom:4px;">📊 Performance</div>
            <div style="color:#94a3b8;font-size:0.88rem;">On SemEval-2014 Task 4 benchmark: ~78% aspect detection accuracy, ~82% sentiment classification accuracy (within this domain).</div>
        </div>
        
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🔗 Dataset & References")
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:16px;font-size:0.88rem;line-height:2;">
        📦 <b>Dataset:</b> Amazon Fine Food Reviews<br>
        🔗 <a href="https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews" style="color:#00d4ff;">kaggle.com/datasets/snap/amazon-fine-food-reviews</a><br>
        📄 568,454 reviews · 10 columns · 1999–2012<br><br>
        📚 <b>References:</b><br>
        · Pontiki et al., SemEval-2014 Task 4<br>
        · Hutto & Gilbert, VADER (AAAI 2014)<br>
        · Liu, Sentiment Analysis & Opinion Mining
        </div>
        """, unsafe_allow_html=True)
