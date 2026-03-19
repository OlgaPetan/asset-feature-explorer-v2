"""
Asset Intelligence — Asset Feature Explorer
Combined app: exploration + combinations + insights + rulebook
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Asset Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RED, DARK, MID, LIGHT_BG = "#E8002D", "#1A1A1A", "#555555", "#FAFAF8"
BLUE, GREEN, AMBER, PURPLE = "#2D5BE3", "#27AE60", "#E67E22", "#8E44AD"

METRIC_COL_MAP = {
    "Attention":   "Attention_T2B",
    "Persuasion":  "Persuasion_T2B",
    "Likeability": "Likeability_Love_Like_T2B",
    "SCD Score":   "SCD_score",
}
METRIC_COLORS = {"Attention": BLUE, "Persuasion": RED, "Likeability": GREEN, "SCD Score": PURPLE}
RULE_COLORS = {
    "Conflict":           ("#FFF0F0", RED),
    "Heterogeneity":      ("#FFFBF0", AMBER),
    "Boundary Condition": ("#F8F0FF", PURPLE),
    "Opportunity":        ("#F0FFF4", GREEN),
    "Outlier":            ("#EBF3FB", BLUE),
    "Consensus":          ("#F0FFF4", GREEN),
    "Anti-pattern":       ("#FFF0F0", RED),
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Source Sans 3',sans-serif;font-size:15px;background:#FAFAF8;color:#1C1C1C;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2.5rem 4rem 2.5rem!important;max-width:1440px;}
.topbar{background:#E8002D;margin:0 -2.5rem 0 -2.5rem;padding:.65rem 2.5rem;display:flex;align-items:center;justify-content:space-between;}
.topbar-left{display:flex;align-items:center;gap:.9rem;}
.topbar-logo{font-family:'Merriweather',serif;font-size:1.05rem;font-weight:400;color:#fff;}
.topbar-pipe{width:1px;height:16px;background:rgba(255,255,255,.3);}
.topbar-sub{font-size:.76rem;color:rgba(255,255,255,.65);letter-spacing:.1em;text-transform:uppercase;}
.page-hero{padding:1.8rem 0 1.4rem 0;border-bottom:1px solid #EAE8E2;margin-bottom:1.8rem;}
.page-eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:#E8002D;margin-bottom:.4rem;}
.page-title{font-family:'Merriweather',serif;font-size:2rem;font-weight:300;color:#1C1C1C;line-height:1.2;}
.page-sub{font-size:.9rem;color:#7A7670;font-weight:300;max-width:640px;line-height:1.7;margin-top:.4rem;}
.section-label{font-size:.73rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#6A6660;margin-bottom:.7rem;margin-top:.2rem;}
.kpi-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.1rem 1.3rem;}
.kpi-val{font-family:'Merriweather',serif;font-size:2rem;font-weight:300;line-height:1;}
.kpi-label{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#AAA8A0;margin-top:.3rem;}
.kpi-sub{font-size:.8rem;color:#888;margin-top:.25rem;}
.score-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:.9rem 1.1rem;text-align:center;}
.score-card.active{border-color:#E8002D;background:#FFF8F8;}
.score-metric{font-size:.67rem;letter-spacing:.14em;text-transform:uppercase;color:#CCC8C2;margin-bottom:.3rem;}
.score-val{font-family:'Merriweather',serif;font-size:1.9rem;font-weight:300;color:#1C1C1C;line-height:1;}
.score-delta{font-size:.8rem;font-weight:600;margin-top:.2rem;}
.d-up{color:#2A8050;}.d-dn{color:#C03030;}.d-flat{color:#CCC8C2;}
.score-n{font-size:.7rem;color:#D8D4CE;margin-top:.18rem;}
.combo-card{background:#fff;border:1px solid #EAE8E2;border-left:3px solid #E8002D;border-radius:0 6px 6px 0;padding:1.4rem 1.6rem;margin-bottom:1.1rem;}
.combo-uplift{font-family:'Merriweather',serif;font-size:3rem;color:#E8002D;line-height:1;font-weight:300;}
.combo-meta{font-size:.86rem;color:#AAA89E;margin-top:.25rem;line-height:1.6;}
.pill-row{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.9rem;}
.pill{display:inline-flex;align-items:center;gap:.38rem;background:#FAFAF8;border:1px solid #EAE8E2;border-radius:3px;padding:.26rem .75rem;font-size:.81rem;color:#5A5650;}
.pill-n{background:#E8002D;color:#fff;border-radius:2px;width:17px;height:17px;display:inline-flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;flex-shrink:0;}
.pill-gain{color:#2A8050;font-size:.79rem;font-weight:600;}
.wf-row{display:flex;align-items:center;gap:.75rem;padding:.52rem 0;border-bottom:1px solid #F5F3EF;}
.wf-idx{width:20px;font-size:.75rem;color:#CCC8C2;text-align:center;flex-shrink:0;}
.wf-label{flex:1;font-size:.88rem;color:#3A3830;}
.wf-bar{width:130px;flex-shrink:0;}
.wf-n{width:50px;font-size:.74rem;color:#BBB8B2;text-align:right;flex-shrink:0;}
.wf-pp{width:62px;font-size:.86rem;font-weight:600;color:#2A8050;text-align:right;flex-shrink:0;}
.wf-col{font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;color:#CCC8C2;}
.explorer-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.4rem 1.6rem;margin-top:1.1rem;}
.explorer-title{font-family:'Merriweather',serif;font-size:1.25rem;font-weight:300;color:#1C1C1C;margin-bottom:.22rem;}
.explorer-sub{font-size:.86rem;color:#AAA89E;line-height:1.6;margin-bottom:1.3rem;}
.feat-group{font-size:.65rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#CCC8C2;margin:1.1rem 0 .65rem 0;padding-bottom:.3rem;border-bottom:1px solid #F0EEE8;}
.insight-strip{display:flex;flex-direction:column;gap:.38rem;margin:.45rem 0;}
.insight-warn{display:flex;align-items:flex-start;gap:.6rem;border-radius:4px;padding:.52rem .82rem;font-size:.82rem;line-height:1.5;border:1px solid;}
.insight-warn-icon{flex-shrink:0;margin-top:.04rem;}
.insight-warn-body{flex:1;}
.insight-warn-type{font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.16rem;}
.warn-conflict{background:#FFF0F0;border-color:#F0C0C0;}.warn-conflict .insight-warn-type{color:#C00020;}
.warn-heterogeneity{background:#FFFBF0;border-color:#EDD080;}.warn-heterogeneity .insight-warn-type{color:#906820;}
.warn-boundary{background:#F8F0FF;border-color:#D4B0F0;}.warn-boundary .insight-warn-type{color:#7030A0;}
.warn-opportunity{background:#F0FFF4;border-color:#90D8A8;}.warn-opportunity .insight-warn-type{color:#1A7040;}
.warn-outlier{background:#EBF3FB;border-color:#90B0E8;}.warn-outlier .insight-warn-type{color:#1A3080;}
.warn-consensus{background:#F0FFF4;border-color:#90D8A8;}.warn-consensus .insight-warn-type{color:#1A7040;}
.warn-antipattern{background:#FFF0F0;border-color:#F0C0C0;}.warn-antipattern .insight-warn-type{color:#C00020;}
.warn-insight{background:#FAFAF8;border-color:#E4E0DA;}.warn-insight .insight-warn-type{color:#6A6660;}
.insight-card{background:#fff;border:1px solid #E8E4DC;border-left:4px solid #E8002D;border-radius:6px;padding:13px 17px;margin-bottom:8px;}
.insight-card-feature{font-weight:600;font-size:.91rem;color:#1A1A1A;}
.insight-card-text{font-size:.83rem;color:#444;margin-top:3px;line-height:1.5;}
.insight-card-meta{display:flex;gap:7px;margin-top:8px;flex-wrap:wrap;}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.69rem;font-weight:600;letter-spacing:.03em;}
.badge-high{background:#EAFAF1;color:#27AE60;}.badge-medium{background:#FEF9E7;color:#E67E22;}
.badge-low{background:#F2F2F2;color:#888;}.badge-pos{background:#EAFAF1;color:#27AE60;}
.badge-neg{background:#FDEDEC;color:#E8002D;}.badge-metric{background:#EBF3FB;color:#2D5BE3;}
.badge-sig{background:#F5F0FF;color:#8E44AD;}.badge-flag{background:#FFF8E1;color:#F57F17;}
.badge-scope{background:#F5F5F5;color:#555;}
.rule-card{border-radius:6px;padding:11px 15px;margin-bottom:8px;border:1px solid rgba(0,0,0,.07);}
.rule-card-type{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}
.rule-card-text{font-size:.83rem;color:#333;line-height:1.5;}
.asset-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:.85rem .95rem;margin-bottom:.55rem;}
.asset-name{font-size:.8rem;font-weight:600;color:#1C1C1C;word-break:break-all;}
.asset-meta{font-size:.73rem;color:#AAA89E;margin-top:.18rem;}
.asset-scores{display:flex;gap:.55rem;margin-top:.45rem;flex-wrap:wrap;}
.asset-score-chip{font-size:.7rem;padding:2px 7px;border-radius:3px;font-weight:600;}
.chip-att{background:#EBF3FB;color:#2D5BE3;}.chip-pers{background:#FFF0F0;color:#E8002D;}
.chip-like{background:#EAFAF1;color:#27AE60;}.chip-scd{background:#F5F0FF;color:#8E44AD;}
.low-n{background:#FFFBF0;border:1px solid #EDD080;border-radius:4px;padding:.48rem .85rem;font-size:.81rem;color:#906820;margin-top:.65rem;}
hr.div{border:none;border-top:1px solid #EAE8E2;margin:1.6rem 0;}
.stSelectbox label{font-size:.78rem!important;color:#AAA89E!important;font-weight:400!important;letter-spacing:.05em!important;text-transform:uppercase!important;}
.stSelectbox>div>div{background:#FAFAF8!important;border:1px solid #E4E0DA!important;border-radius:4px!important;color:#1C1C1C!important;font-size:.88rem!important;}
.stButton>button{background:#fff!important;border:1px solid #DAD6D0!important;color:#6A6660!important;border-radius:4px!important;font-size:.86rem!important;width:100%!important;}
.stButton>button:hover{border-color:#E8002D!important;color:#E8002D!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid #EAE8E2;gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#AAA89E;font-size:.88rem;border-radius:0;padding:.52rem 1.2rem;border:none;border-bottom:2px solid transparent;margin-bottom:-1px;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#1C1C1C!important;border-bottom:2px solid #E8002D!important;font-weight:600!important;}
.scope-chip{display:inline-flex;align-items:center;background:#FFF0F0;border:1px solid #F0C0C0;border-radius:3px;padding:.18rem .68rem;font-size:.81rem;color:#C00020;margin:.18rem .22rem .18rem 0;font-weight:500;}
.footer{display:flex;justify-content:space-between;font-size:.72rem;color:#CCC8C2;padding:1.1rem 0 .7rem 0;border-top:1px solid #EAE8E2;margin-top:1.8rem;}
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_all():
    with open("precomputed_data.pkl", "rb") as f:
        payload = pickle.load(f)
    results = payload["results"]
    meta    = payload["meta"]
    df      = meta["df"].copy()

    see  = ["Experience_Recall_T2B_percentile","Brand_Linkage_T2B_percentile","Comprehension_T2B_percentile"]
    conn = ["Likeability_Love_Like_T2B_percentile","Uniqueness_T2B_percentile","Brand_Interest_T2B_percentile"]
    do   = ["Persuasion_T2B_percentile","Shareability_T2B_percentile"]
    df["SCD_score"] = (
        df[[c for c in see  if c in df.columns]].mean(axis=1, skipna=False) * 0.1 +
        df[[c for c in conn if c in df.columns]].mean(axis=1, skipna=False) * 0.3 +
        df[[c for c in do   if c in df.columns]].mean(axis=1, skipna=False) * 0.6
    ).round(4)

    camp_map = (df[["campaign_sk_id","campaign_display_name","campaign_code"]]
                .drop_duplicates("campaign_sk_id")
                .set_index("campaign_sk_id"))

    try:
        catalog  = pd.read_csv("insight_catalog.csv")
        rulebook = pd.read_csv("rulebook.csv")
        for d in [catalog, rulebook]:
            for c in d.select_dtypes(include="object").columns:
                d[c] = d[c].fillna("")
        catalog["evidence_uplift_pp"] = pd.to_numeric(catalog["evidence_uplift_pp"], errors="coerce")
        has_insights = True
    except FileNotFoundError:
        catalog = rulebook = None
        has_insights = False

    try:
        uplift_df = pd.read_csv("uplift_all_scopes.csv")
        uplift_df["uplift_pp"] = pd.to_numeric(uplift_df["uplift_pp"], errors="coerce")
    except FileNotFoundError:
        uplift_df = None

    return df, results, meta, camp_map, catalog, rulebook, uplift_df, has_insights

try:
    df_full, results, meta, camp_map, catalog, rulebook, uplift_df, has_insights = load_all()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}"); st.stop()

METRICS    = meta["metrics"]
FEAT_LABEL = meta["feat_label"]
BIN_FEATS  = meta["binary_feats"]
CAT_FEATS  = meta["cat_feats"]
ALL_FEATS  = BIN_FEATS + CAT_FEATS
SCOPE_MAP  = {"ou":"operating_unit_code","category":"category",
               "brand":"brand_name","market":"country_name"}


# ── Helpers ────────────────────────────────────────────────────────────────────
def badge(text, cls): return f'<span class="badge {cls}">{text}</span>'
def conf_badge(c):    return badge(c.upper(), f"badge-{c}")
def metric_badge(m):  return badge(m, "badge-metric")
def sig_badge(s):     return badge(s, "badge-sig")
def dir_badge(d):
    return badge(("▲ " if d=="positive" else "▼ ")+d, "badge-pos" if d=="positive" else "badge-neg")
def flag_badge(f):
    if not f or f=="": return ""
    return "".join(badge(x.strip(),"badge-flag") for x in str(f).split(",") if x.strip())
def uplift_html(v):
    if pd.isna(v): return ""
    return f'<span class="badge {"badge-pos" if v>=0 else "badge-neg"}">{"▲" if v>=0 else "▼"} {abs(v):.1f}pp</span>'

def get_scoped_df(scope_filters, campaign_id=None):
    sub = df_full.copy()
    for t, v in scope_filters:
        sub = sub[sub[SCOPE_MAP[t]]==v]
    if campaign_id:
        sub = sub[sub["campaign_sk_id"]==campaign_id]
    return sub

def get_scope_key(scope_filters):
    if not scope_filters: return "global||All"
    if len(scope_filters)==1:
        t, v = scope_filters[0]
        k = f"{t}||{v}"
        return k if k in results else None
    return None

def bar_html(frac):
    pct = max(2, min(100, frac*100))
    return (f'<div style="background:#F0EDE8;border-radius:2px;height:5px;">'
            f'<div style="width:{pct:.0f}%;height:5px;background:#E8002D;border-radius:2px;"></div></div>')

def compute_uplift(sub, feat, metric_col):
    if feat not in sub.columns or metric_col not in sub.columns:
        return None, None, None
    if feat in BIN_FEATS:
        g1 = sub.loc[sub[feat]==1, metric_col].dropna()
        g0 = sub.loc[sub[feat]==0, metric_col].dropna()
    else:
        mask = sub[feat].notna() & (sub[feat].astype(str).str.strip()!="")
        g1 = sub.loc[mask,  metric_col].dropna()
        g0 = sub.loc[~mask, metric_col].dropna()
    if len(g1)<5 or len(g0)<5: return None, None, None
    u = (g1.mean()-g0.mean())*100
    try: _, p = mannwhitneyu(g1, g0, alternative="two-sided")
    except: p = 1.0
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    return round(u,2), sig, len(g1)

def compute_feature_combinations(sub_df, target_feat, metric_col, top_n=5):
    """
    For a given feature, find the other binary features that combine best with it.
    Returns a list of dicts: {partner, combined_uplift, solo_uplift, synergy, n}
    combined_uplift = mean(both present) - global mean
    synergy         = combined_uplift - solo_uplift (extra gain from pairing)
    """
    if target_feat not in sub_df.columns or metric_col not in sub_df.columns:
        return []
    global_mean = sub_df[metric_col].dropna().mean() * 100
    # Solo uplift of target
    g1_t = sub_df.loc[sub_df[target_feat]==1, metric_col].dropna() if target_feat in BIN_FEATS            else sub_df.loc[sub_df[target_feat].notna() & (sub_df[target_feat].astype(str).str.strip()!=""), metric_col].dropna()
    solo_target = g1_t.mean() * 100 - global_mean if len(g1_t) >= 3 else None
    if solo_target is None:
        return []
    rows = []
    for feat in BIN_FEATS:
        if feat == target_feat or feat not in sub_df.columns:
            continue
        # Assets with BOTH features
        if target_feat in BIN_FEATS:
            mask_both = (sub_df[target_feat]==1) & (sub_df[feat]==1)
        else:
            mask_t = sub_df[target_feat].notna() & (sub_df[target_feat].astype(str).str.strip()!="")
            mask_both = mask_t & (sub_df[feat]==1)
        n_both = int(mask_both.sum())
        if n_both < 3:
            continue
        combined_uplift = sub_df.loc[mask_both, metric_col].dropna().mean() * 100 - global_mean
        solo_partner_g1 = sub_df.loc[sub_df[feat]==1, metric_col].dropna()
        solo_partner = solo_partner_g1.mean() * 100 - global_mean if len(solo_partner_g1) >= 3 else 0
        synergy = combined_uplift - max(solo_target, solo_partner)
        rows.append({
            "partner":          feat,
            "partner_label":    FEAT_LABEL.get(feat, feat),
            "combined_uplift":  round(combined_uplift, 2),
            "solo_target":      round(solo_target, 2),
            "solo_partner":     round(solo_partner, 2),
            "synergy":          round(synergy, 2),
            "n":                n_both,
        })
    rows.sort(key=lambda x: x["combined_uplift"], reverse=True)
    return rows[:top_n]


def render_alerts(feature=None, scope_filters=None, max_items=4):
    if not has_insights or rulebook is None: return False
    rb = rulebook.copy()
    if feature: rb = rb[rb["feature"]==feature]
    if scope_filters:
        smap = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
        mask = pd.Series(False, index=rb.index)
        for t,v in scope_filters:
            mask |= ((rb["scope"]==smap.get(t,t)) & (rb["scope_value"]==v))
        mask |= (rb["scope"]=="Global")
        rb = rb[mask]
    else:
        rb = rb[rb["scope"]=="Global"]
    rb = rb.sort_values("severity", key=lambda s: s.map({"high":0,"medium":1,"low":2}))
    items = rb.head(max_items)
    if items.empty: return False
    RCSS  = {"Conflict":"warn-conflict","Heterogeneity":"warn-heterogeneity",
              "Boundary Condition":"warn-boundary","Opportunity":"warn-opportunity",
              "Outlier":"warn-outlier","Consensus":"warn-consensus","Anti-pattern":"warn-antipattern"}
    RICON = {"Conflict":"⚡","Heterogeneity":"⚠","Boundary Condition":"◈",
              "Opportunity":"◎","Outlier":"↗","Consensus":"✓","Anti-pattern":"✕"}
    html = '<div class="insight-strip">'
    for _, r in items.iterrows():
        css = RCSS.get(r["rule_type"],"warn-insight")
        icon= RICON.get(r["rule_type"],"ℹ")
        html += (f'<div class="insight-warn {css}">'
                 f'<div class="insight-warn-icon">{icon}</div>'
                 f'<div class="insight-warn-body">'
                 f'<div class="insight-warn-type">{r["rule_type"]}</div>'
                 f'<div style="font-size:.82rem;color:#333">{r["text"]}</div>'
                 f'</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    return True

def catalog_insights(feature, scope_filters, metric_col):
    if not has_insights or catalog is None: return []
    ml_map = {"Attention_T2B":"Attention","Persuasion_T2B":"Persuasion",
              "Likeability_Love_Like_T2B":"Likeability","SCD_score":"SCD Score"}
    ml = ml_map.get(metric_col, metric_col)
    cat = catalog[(catalog["feature"]==feature) &
                  (catalog["metric_display"]==ml) &
                  (catalog["confidence"].isin(["high","medium"]))]
    out = []
    smap = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
    if scope_filters:
        for t,v in scope_filters:
            sl = smap.get(t,t)
            for _, r in cat[(cat["filter"]==sl)&(cat["filter_value"]==v)].head(1).iterrows():
                out.append(r["text"])
    if not out:
        for _, r in cat[cat["filter"]=="Global"].head(1).iterrows():
            out.append(r["text"])
    return out

def get_opts(feat, scope_key, metric_col, sub_df):
    fv = None
    if scope_key and scope_key in results and metric_col in results[scope_key]:
        fv = results[scope_key][metric_col].get("feature_options",{}).get(feat)
    raws, disp = ["__any__"], ["Any"]
    if fv:
        for row in fv:
            if row["n"]<5: continue
            raws.append(str(row["val"]))
            disp.append(f"{row['label']}  (n={row['n']:,})")
    elif feat in BIN_FEATS and feat in sub_df.columns:
        n1=int((sub_df[feat]==1).sum()); n0=int((sub_df[feat]==0).sum())
        raws+=["1","0"]; disp+=[f"Yes  (n={n1:,})",f"No  (n={n0:,})"]
    elif feat in sub_df.columns:
        for v in sorted(sub_df[feat].dropna().unique()):
            n=int((sub_df[feat]==v).sum())
            if n<5: continue
            raws.append(str(v)); disp.append(f"{str(v)[:35]}  (n={n:,})")
    return raws, disp

def apply_sel(sub_df, sel):
    mask = pd.Series(True, index=sub_df.index)
    for feat, val in sel.items():
        if val=="__any__" or feat not in sub_df.columns: continue
        mask &= (sub_df[feat]==(1 if val=="1" else 0)) if feat in BIN_FEATS else (sub_df[feat]==val)
    return mask

def score_sel(sub_df, sel):
    mask=apply_sel(sub_df, sel); n=int(mask.sum())
    return {mc:(round(sub_df.loc[mask,mc].dropna().mean()*100,2),n) if n>0 else (None,0)
            for mc in list(METRICS.keys())+["SCD_score"] if mc in sub_df.columns}

def default_sel(combo):
    sel={f:"__any__" for f in ALL_FEATS}
    if combo:
        for feat,val in combo:
            if feat in ALL_FEATS:
                sel[feat]="1" if val==1 else ("0" if val==0 else str(val))
    return sel

def make_heatmap(piv, title, figsize=(13,5)):
    cmap=LinearSegmentedColormap.from_list("rg",["#D32F2F","white","#2E7D32"],N=256)
    vals=piv.values.astype(float); vmax=max(np.nanpercentile(np.abs(vals),95),1.0)
    fig,ax=plt.subplots(figsize=figsize); fig.patch.set_facecolor(LIGHT_BG); ax.set_facecolor(LIGHT_BG)
    im=ax.imshow(vals,cmap=cmap,vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,fontsize=8.5,rotation=30,ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=8.5)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v=vals[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=7.5,
                        color="white" if abs(v)>vmax*.55 else DARK)
            else:
                ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fc="#E8E4DE",ec="white",lw=.5))
    plt.colorbar(im,ax=ax,shrink=.55,label="Uplift (pp)")
    ax.set_title(title,fontsize=10,fontweight="bold",color=DARK,pad=8)
    ax.spines[:].set_visible(False); plt.tight_layout()
    return fig


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">The Coca&#8209;Cola Company</div>
    <div class="topbar-pipe"></div>
    <div class="topbar-sub">Asset Intelligence</div>
  </div>
  <div class="topbar-sub">Asset Feature Explorer</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Page", [
        "Overview",
        "Combinations & Explorer",
        "Feature Deep-Dive",
        "Insight Catalog",
        "Rulebook",
        "Heatmaps",
    ])
    st.markdown("---")
    st.markdown(f"<small style='color:#888'>{len(df_full):,} assets · {len(results)} scopes</small>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED SCOPE + CAMPAIGN FILTER BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="padding:1.4rem 0 .4rem 0"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Scope &amp; Filters</div>', unsafe_allow_html=True)

SCOPE_TYPES = ["Global","OU","Category","Brand","Market"]
if "scope_types" not in st.session_state:
    st.session_state.scope_types = ["Global"]

sc = st.columns([1,1,1,1,1,3])
for i, s in enumerate(SCOPE_TYPES):
    with sc[i]:
        active = s in st.session_state.scope_types
        if st.button(f"· {s}" if active else s, key=f"sb_{s}"):
            if s=="Global":
                st.session_state.scope_types=["Global"]
                for k in ["sel_ou","sel_cat","sel_brand","sel_market"]:
                    st.session_state.pop(k,None)
            else:
                if "Global" in st.session_state.scope_types:
                    st.session_state.scope_types=[]
                if s in st.session_state.scope_types:
                    st.session_state.scope_types.remove(s)
                else:
                    st.session_state.scope_types.append(s)
            st.experimental_rerun()

scope_filters = []
KEY_MAP = {
    "OU":       ("ou",       sorted(df_full["operating_unit_code"].dropna().unique()), "sel_ou"),
    "Category": ("category", sorted(df_full["category"].dropna().unique()),             "sel_cat"),
    "Brand":    ("brand",    sorted(df_full["brand_name"].dropna().unique()),            "sel_brand"),
    "Market":   ("market",   sorted(df_full["country_name"].dropna().unique()),          "sel_market"),
}
active_types = st.session_state.scope_types
if active_types and "Global" not in active_types:
    vcols = st.columns(len(active_types)+1)
    for i, stype in enumerate(active_types):
        with vcols[i]:
            tk, opts, sk = KEY_MAP[stype]
            sv = st.selectbox(f"Select {stype}", opts, key=sk)
            scope_filters.append((tk, sv))
    sub_for_camp = get_scoped_df(scope_filters)
    with vcols[len(active_types)]:
        cdf = (sub_for_camp[["campaign_sk_id","campaign_display_name","campaign_code"]]
               .drop_duplicates("campaign_sk_id").sort_values("campaign_display_name"))
        clabels = ["All campaigns"] + [f"{r['campaign_display_name']} ({r['campaign_code']})"
                                        for _,r in cdf.iterrows()]
        cids = [None] + cdf["campaign_sk_id"].tolist()
        ci = st.selectbox("Campaign", range(len(clabels)),
                           format_func=lambda i: clabels[i], key="sel_campaign")
        selected_campaign_id = cids[ci]
else:
    cdf_g = (df_full[["campaign_sk_id","campaign_display_name","campaign_code"]]
             .drop_duplicates("campaign_sk_id").sort_values("campaign_display_name"))
    clabels_g = ["All campaigns"] + [f"{r['campaign_display_name']} ({r['campaign_code']})"
                                      for _,r in cdf_g.iterrows()]
    cids_g = [None] + cdf_g["campaign_sk_id"].tolist()
    ci_g = st.selectbox("Campaign (optional)", range(len(clabels_g)),
                         format_func=lambda i: clabels_g[i], key="sel_campaign_global")
    selected_campaign_id = cids_g[ci_g]

sub_df    = get_scoped_df(scope_filters, selected_campaign_id)
scope_key = get_scope_key(scope_filters) if not selected_campaign_id else None
min_n     = 15 if scope_filters else 30

if scope_filters:
    lmap  = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
    chips = "".join(f'<span class="scope-chip">{lmap[t]}: {v}</span>' for t,v in scope_filters)
    if selected_campaign_id and selected_campaign_id in camp_map.index:
        chips += f'<span class="scope-chip">Campaign: {camp_map.loc[selected_campaign_id,"campaign_display_name"]}</span>'
    st.markdown(f'<div style="margin:.55rem 0 0 0">{chips}</div>', unsafe_allow_html=True)
else:
    suffix = ""
    if selected_campaign_id and selected_campaign_id in camp_map.index:
        suffix = f' &nbsp;&middot;&nbsp; Campaign: {camp_map.loc[selected_campaign_id,"campaign_display_name"]}'
    st.markdown(f'<p style="font-size:.75rem;color:#CCC8C2;margin:.45rem 0 0 0">Global scope{suffix}</p>',
                unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

if len(sub_df) < 3:
    st.warning(f"Only {len(sub_df)} assets in this scope — too few for reliable analysis.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    scope_label = "Global" if not scope_filters else " · ".join(f"{t.upper()}={v}" for t,v in scope_filters)
    st.markdown(f"""
    <div class="page-hero">
      <div class="page-eyebrow">Asset Intelligence</div>
      <div class="page-title">Asset Feature Explorer</div>
      <div class="page-sub">How asset features drive Attention, Persuasion, Likeability and SCD.
        Scope: <strong>{scope_label}</strong> &nbsp;&middot;&nbsp; {len(sub_df):,} assets</div>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-val">{len(sub_df):,}</div>
          <div class="kpi-label">Assets in scope</div>
          <div class="kpi-sub">{sub_df["campaign_sk_id"].nunique():,} campaigns</div>
        </div>""", unsafe_allow_html=True)
    for ml, col in [("Attention",k2),("Persuasion",k3),("Likeability",k4),("SCD Score",k5)]:
        mc = METRIC_COL_MAP[ml]
        mc_color = METRIC_COLORS[ml]
        if mc in sub_df.columns:
            mean_v = sub_df[mc].dropna().mean()
            with col:
                st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {mc_color}">
                  <div class="kpi-val" style="color:{mc_color}">{mean_v*100:.1f}<span style="font-size:1.1rem">pp</span></div>
                  <div class="kpi-label">{ml}</div>
                  <div class="kpi-sub">Mean score in scope</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    if has_insights and scope_filters:
        with st.expander(f"Scope alerts", expanded=True):
            render_alerts(scope_filters=scope_filters)

    # ── Feature Impact table + chart ─────────────────────────────────────────
    st.markdown('<div class="section-label">How each feature impacts performance</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.82rem;color:#7A7670;margin-bottom:1rem;line-height:1.6;max-width:760px">
    <strong>What this shows:</strong> For each Asset feature, the table below shows whether
    including it in an ad is associated with higher or lower scores on Attention, Persuasion,
    Likeability and SCD. <strong>Uplift</strong> is the difference in average score between
    ads that have the feature and ads that don't, expressed in percentage points (pp).
    A green value means the feature is associated with better performance; red means worse.
    Stars (*, **, ***) indicate how statistically confident we are — *** means very confident,
    * means directional only, <em>ns</em> means the difference is not significant.
    <strong>Best combination</strong> shows which other feature, when paired with this one,
    produces the highest combined SCD score — and the expected uplift of that pairing.</div>""",
    unsafe_allow_html=True)

    # Build feature impact rows
    impact_rows = []
    for feat in BIN_FEATS:
        if feat not in sub_df.columns: continue
        row = {"Feature": FEAT_LABEL.get(feat, feat), "_feat": feat}
        has_any = False
        for ml, mc in [("Attention","Attention_T2B"),("Persuasion","Persuasion_T2B"),
                       ("Likeability","Likeability_Love_Like_T2B"),("SCD Score","SCD_score")]:
            u, sig, n = compute_uplift(sub_df, feat, mc)
            if u is not None:
                row[ml] = u; row[f"{ml}_sig"] = sig; row[f"{ml}_n"] = n
                has_any = True
            else:
                row[ml] = None; row[f"{ml}_sig"] = ""; row[f"{ml}_n"] = 0
        if not has_any: continue
        # Best combo
        combos = compute_feature_combinations(sub_df, feat, "SCD_score", top_n=1)
        if combos:
            row["Best combination"] = combos[0]["partner_label"]
            row["Expected impact"] = f'+{combos[0]["combined_uplift"]:.1f}pp SCD (n={combos[0]["n"]})'
        else:
            row["Best combination"] = "—"
            row["Expected impact"]  = "—"
        impact_rows.append(row)

    if impact_rows:
        impact_df = pd.DataFrame(impact_rows).sort_values(
            "SCD Score", key=lambda s: s.fillna(-999), ascending=False)

        # Render as styled HTML table
        def fmt_cell(val, sig):
            if val is None: return '<td style="color:#CCC;text-align:center">—</td>'
            color = GREEN if val >= 0 else RED
            arrow = "▲" if val >= 0 else "▼"
            sig_html = f'<span style="color:#8E44AD;font-size:.68rem;margin-left:3px">{sig}</span>' if sig not in ("","ns") else f'<span style="color:#CCC;font-size:.68rem;margin-left:3px">ns</span>'
            return (f'<td style="text-align:center;font-weight:600;color:{color};font-size:.85rem">'
                    f'{arrow} {abs(val):.1f}pp{sig_html}</td>')

        tbl = """<div style="overflow-x:auto;margin-bottom:1rem">
        <table style="width:100%;border-collapse:collapse;font-size:.84rem;font-family:'Source Sans 3',sans-serif">
        <thead><tr style="border-bottom:2px solid #EAE8E2">
          <th style="text-align:left;padding:.55rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Feature</th>
          <th style="text-align:center;padding:.55rem .7rem;color:#2D5BE3;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Attention</th>
          <th style="text-align:center;padding:.55rem .7rem;color:#E8002D;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Persuasion</th>
          <th style="text-align:center;padding:.55rem .7rem;color:#27AE60;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Likeability</th>
          <th style="text-align:center;padding:.55rem .7rem;color:#8E44AD;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">SCD Score</th>
          <th style="text-align:left;padding:.55rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Best combination</th>
          <th style="text-align:left;padding:.55rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Expected impact</th>
        </tr></thead><tbody>"""
        for i, row in impact_df.iterrows():
            bg = "#FAFAF8" if list(impact_df.index).index(i) % 2 == 0 else "#FFFFFF"
            scd_val = row.get("SCD Score")
            direction_icon = ("▲" if scd_val is not None and scd_val >= 0 else "▼") if scd_val is not None else ""
            feat_color = GREEN if scd_val is not None and scd_val >= 0 else (RED if scd_val is not None else DARK)
            tbl += f'<tr style="background:{bg};border-bottom:1px solid #F0EEE8">'
            tbl += f'<td style="padding:.5rem .7rem;font-weight:600;color:{feat_color}">{direction_icon} {row["Feature"]}</td>'
            for ml in ["Attention","Persuasion","Likeability","SCD Score"]:
                tbl += fmt_cell(row.get(ml), row.get(f"{ml}_sig",""))
            tbl += f'<td style="padding:.5rem .7rem;color:#555">{row["Best combination"]}</td>'
            tbl += f'<td style="padding:.5rem .7rem;color:#27AE60;font-weight:600">{row["Expected impact"]}</td>'
            tbl += "</tr>"
        tbl += """</tbody></table></div>
        <div style="font-size:.76rem;color:#AAA;margin-bottom:.5rem">
        *** p&lt;0.001 &nbsp;·&nbsp; ** p&lt;0.01 &nbsp;·&nbsp; * p&lt;0.05 &nbsp;·&nbsp;
        ns = not significant &nbsp;·&nbsp; pp = percentage points &nbsp;·&nbsp;
        Uplift = mean score with feature minus mean score without</div>"""
        st.markdown(tbl, unsafe_allow_html=True)

        # OU Impact section
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">OU impact — SCD score by feature</div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-size:.82rem;color:#7A7670;margin-bottom:.9rem;line-height:1.6;max-width:760px">
        The same feature can perform very differently across Operating Units.
        This table shows the SCD uplift for each binary feature within each OU.
        A red cell means the feature is associated with <em>lower</em> scores in that OU;
        green means <em>higher</em>. Grey means insufficient data.
        Use this to avoid briefing a globally positive feature in an OU where it actually hurts.</div>""",
        unsafe_allow_html=True)

        ou_list = sorted(sub_df["operating_unit_code"].dropna().unique())
        if len(ou_list) >= 2:
            ou_impact = {}
            for feat in BIN_FEATS:
                if feat not in sub_df.columns: continue
                ou_impact[feat] = {}
                for ou in ou_list:
                    ou_sub = sub_df[sub_df["operating_unit_code"]==ou]
                    u, sig, n = compute_uplift(ou_sub, feat, "SCD_score")
                    ou_impact[feat][ou] = (u, sig, n) if u is not None else None

            # Render OU impact table
            tbl2 = """<div style="overflow-x:auto;margin-bottom:1rem">
            <table style="width:100%;border-collapse:collapse;font-size:.82rem;font-family:'Source Sans 3',sans-serif">
            <thead><tr style="border-bottom:2px solid #EAE8E2">
              <th style="text-align:left;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Feature</th>"""
            for ou in ou_list:
                tbl2 += f'<th style="text-align:center;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">{ou}</th>'
            tbl2 += "</tr></thead><tbody>"
            for fi, feat in enumerate([f for f in BIN_FEATS if f in ou_impact]):
                bg2 = "#FAFAF8" if fi % 2 == 0 else "#FFFFFF"
                tbl2 += f'<tr style="background:{bg2};border-bottom:1px solid #F0EEE8">'
                tbl2 += f'<td style="padding:.45rem .7rem;font-weight:600;color:{DARK}">{FEAT_LABEL.get(feat,feat)}</td>'
                for ou in ou_list:
                    val = ou_impact[feat].get(ou)
                    if val is None:
                        tbl2 += '<td style="text-align:center;color:#DDD;font-size:.78rem">—</td>'
                    else:
                        u2, sig2, n2 = val
                        color2 = GREEN if u2 >= 0 else RED
                        bg_cell = "#F0FFF4" if u2 >= 3 else ("#FFF0F0" if u2 <= -3 else "transparent")
                        tbl2 += (f'<td style="text-align:center;background:{bg_cell};'
                                 f'font-weight:600;color:{color2};font-size:.82rem">'
                                 f'{"+" if u2>=0 else ""}{u2:.1f}'
                                 f'<span style="font-size:.65rem;color:#AAA;margin-left:2px">{sig2}</span></td>')
                tbl2 += "</tr>"
            tbl2 += """</tbody></table></div>
            <div style="font-size:.76rem;color:#AAA;margin-bottom:.5rem">
            Values are SCD uplift in pp. Strong green (≥+3pp) and strong red (≤-3pp) cells are highlighted.</div>"""
            st.markdown(tbl2, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#AAA;font-size:.84rem">Select a broader scope to see OU comparison.</p>',
                        unsafe_allow_html=True)

    col_left_alerts, col_right_alerts = st.columns([1, 1])
    with col_left_alerts:
        pass  # space
    with col_right_alerts:
        st.markdown('<div class="section-label">Priority alerts</div>', unsafe_allow_html=True)
        if has_insights and rulebook is not None:
            for _, r in rulebook[rulebook["severity"]=="high"].head(6).iterrows():
                bg, accent = RULE_COLORS.get(r["rule_type"],("#F9F9F9",MID))
                st.markdown(f"""<div class="rule-card" style="background:{bg};border-left:3px solid {accent}">
                  <div class="rule-card-type" style="color:{accent}">{r["rule_type"]}</div>
                  <div class="rule-card-text">{r["text"]}</div>
                  <div style="margin-top:5px">{badge(r.get("feature",""),"badge-metric")}
                    {badge(r.get("scope",""),"badge-scope")}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Top performing assets in this scope</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.82rem;color:#888;margin-bottom:.75rem">
    Assets ranked by SCD score. Click "View asset" to see the Asset.</div>""", unsafe_allow_html=True)
    for _, r in sub_df[sub_df["SCD_score"].notna()].sort_values("SCD_score",ascending=False).head(8).iterrows():
        feats_present = [FEAT_LABEL.get(f,f) for f in BIN_FEATS if r.get(f)==1]
        url  = r.get("asset_url","")
        name = r.get("asset_name", f"Asset {r.get('asset_sk_id','')}")
        link = f'<a href="{url}" target="_blank" style="color:{RED};font-size:.73rem;text-decoration:none;font-weight:600">▶ View asset</a>' if url else ""
        att  = f"{r.get('Attention_T2B',0)*100:.0f}" if pd.notna(r.get("Attention_T2B")) else "—"
        pers = f"{r.get('Persuasion_T2B',0)*100:.0f}" if pd.notna(r.get("Persuasion_T2B")) else "—"
        like = f"{r.get('Likeability_Love_Like_T2B',0)*100:.0f}" if pd.notna(r.get("Likeability_Love_Like_T2B")) else "—"
        scd  = f"{r.get('SCD_score',0):.2f}" if pd.notna(r.get("SCD_score")) else "—"
        st.markdown(f"""<div class="asset-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div class="asset-name">{name}</div>{link}
          </div>
          <div class="asset-meta">{r.get("brand_name","")} · {r.get("country_name","")} · {r.get("asset_category","")}</div>
          <div class="asset-meta" style="margin-top:.18rem">Features: {", ".join(feats_present) if feats_present else "—"}</div>
          <div class="asset-scores">
            <span class="asset-score-chip chip-att">Att {att}pp</span>
            <span class="asset-score-chip chip-pers">Pers {pers}pp</span>
            <span class="asset-score-chip chip-like">Like {like}pp</span>
            <span class="asset-score-chip chip-scd">SCD {scd}</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMBINATIONS & EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Combinations & Explorer":
    st.markdown(f"""
    <div class="page-hero">
      <div class="page-eyebrow">Best Combinations</div>
      <div class="page-title">Feature Combination Explorer</div>
      <div class="page-sub">The greedy algorithm finds the feature combination with the
        highest cumulative uplift for each metric. Then explore any combination in real time.</div>
    </div>""", unsafe_allow_html=True)

    if has_insights and scope_filters:
        with st.expander("Scope alerts", expanded=False):
            render_alerts(scope_filters=scope_filters)

    tab_a, tab_p, tab_l = st.tabs(["Attention","Persuasion","Likeability"])
    for metric_col, tab in zip(
        ["Attention_T2B","Persuasion_T2B","Likeability_Love_Like_T2B"],
        [tab_a, tab_p, tab_l]
    ):
        with tab:
            metric_label = METRICS.get(metric_col, metric_col)
            combo, steps, baseline_pp, n_total = None, [], 0, len(sub_df)
            if scope_key and scope_key in results and metric_col in results[scope_key]:
                c = results[scope_key][metric_col]
                combo, steps, baseline_pp, n_total = c["combo"], c["steps"], c["baseline_pp"], c["n_total"]

            if not steps:
                st.markdown('<p style="color:#CCC8C2;padding:1.5rem 0;font-size:.84rem">No combination found for this scope and selection.</p>',
                            unsafe_allow_html=True)
                continue

            final_mean   = steps[-1]["metric_mean"]
            total_uplift = steps[-1]["cumulative_uplift_pp"]
            final_n      = steps[-1]["n"]

            col_combo, col_info = st.columns([1.6,1])
            with col_combo:
                st.markdown(f"""
                <div class="combo-card">
                  <div style="display:flex;align-items:flex-end;gap:.9rem;margin-bottom:.25rem">
                    <span class="combo-uplift">+{total_uplift:.1f}</span>
                    <span style="font-family:'Merriweather',serif;font-size:1.35rem;color:#D0C8C0;font-weight:300">pp</span>
                    <div style="padding-bottom:.25rem">
                      <div style="font-size:.74rem;color:#AAA89E">cumulative uplift on {metric_label}</div>
                      <div style="font-size:.68rem;color:#CCC8C2;margin-top:.08rem">
                        Baseline {baseline_pp:.1f}pp &rarr; {final_mean:.1f}pp &nbsp;&middot;&nbsp;
                        {len(steps)} features &nbsp;&middot;&nbsp; n={final_n:,}
                      </div>
                    </div>
                  </div>
                """, unsafe_allow_html=True)
                pills = '<div class="pill-row">'
                for s in steps:
                    pills += (f'<div class="pill"><span class="pill-n">{s["step"]}</span>'
                              f'{s["label"]}<span class="pill-gain">+{s["step_gain_pp"]:.1f}pp</span></div>')
                st.markdown(pills+"</div></div>", unsafe_allow_html=True)
                if final_n < 30:
                    st.markdown(f'<div class="low-n">Only {final_n} assets — treat as directional.</div>', unsafe_allow_html=True)

                # Waterfall
                st.markdown('<div style="margin-top:1.3rem"></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="wf-row" style="border-bottom:1px solid #EAE8E2;padding-bottom:.22rem">
                  <div class="wf-idx"></div><div class="wf-label wf-col">Feature</div>
                  <div class="wf-bar wf-col">Gain</div><div class="wf-n wf-col">n</div>
                  <div class="wf-pp wf-col">Total</div>
                </div>
                <div class="wf-row">
                  <div class="wf-idx">—</div>
                  <div class="wf-label" style="color:#CCC8C2;font-style:italic">Baseline — {n_total:,} assets</div>
                  <div class="wf-bar">{bar_html(0)}</div>
                  <div class="wf-n">{n_total:,}</div>
                  <div class="wf-pp" style="color:#CCC8C2">{baseline_pp:.1f}</div>
                </div>""", unsafe_allow_html=True)
                max_gain = max(s["step_gain_pp"] for s in steps) if steps else 1
                for s in steps:
                    frac = s["step_gain_pp"]/max_gain if max_gain>0 else 0
                    st.markdown(f"""
                    <div class="wf-row">
                      <div class="wf-idx" style="color:#E8002D;font-weight:600">{s["step"]}</div>
                      <div class="wf-label">+ {s["label"]}</div>
                      <div class="wf-bar">{bar_html(frac)}</div>
                      <div class="wf-n">{s["n"]:,}</div>
                      <div class="wf-pp">+{s["cumulative_uplift_pp"]:.1f}</div>
                    </div>""", unsafe_allow_html=True)

            with col_info:
                st.markdown('<div class="section-label">How to read this</div>', unsafe_allow_html=True)
                st.markdown(f"""<div style="font-size:.82rem;color:#555;line-height:1.6;background:#fff;
                  border:1px solid #EAE8E2;border-radius:6px;padding:.9rem 1rem">
                  <p><strong>Baseline</strong> is the mean {metric_label} score across all
                  {n_total:,} assets in this scope ({baseline_pp:.1f}pp).</p>
                  <p>Each step adds one feature that improves the score the most on top of what
                  is already selected. The <strong>Gain</strong> is that step's individual
                  contribution. <strong>Total</strong> is the cumulative lift from baseline.</p>
                  <p style="margin-bottom:0"><strong>n</strong> is how many assets match all
                  selected features at that step — smaller means less certain.</p>
                </div>""", unsafe_allow_html=True)

                # OU view of the top combo feature
                if steps:
                    top_feat = steps[0]["feat"]
                    st.markdown('<div style="margin-top:1.1rem"></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="section-label">{FEAT_LABEL.get(top_feat,top_feat)} across OUs</div>',
                                unsafe_allow_html=True)
                    ou_rows = []
                    for ou in sorted(df_full["operating_unit_code"].dropna().unique()):
                        ou_sub = df_full[df_full["operating_unit_code"]==ou]
                        if scope_filters:
                            for t,v in scope_filters:
                                if t!="ou": ou_sub=ou_sub[ou_sub[SCOPE_MAP[t]]==v]
                        u2, sig2, n2 = compute_uplift(ou_sub, top_feat, metric_col)
                        if u2 is not None:
                            ou_rows.append({"OU":ou,"uplift":u2,"sig":sig2})
                    if ou_rows:
                        ou_df2 = pd.DataFrame(ou_rows).sort_values("uplift")
                        fig2,ax2 = plt.subplots(figsize=(5.5,3.5))
                        fig2.patch.set_facecolor(LIGHT_BG); ax2.set_facecolor(LIGHT_BG)
                        ax2.barh(ou_df2["OU"], ou_df2["uplift"],
                                  color=[GREEN if v>=0 else RED for v in ou_df2["uplift"]],
                                  alpha=.85, height=.6)
                        ax2.axvline(0, color=DARK, lw=1, ls="--")
                        for i2,(_, r2) in enumerate(ou_df2.iterrows()):
                            v2=r2["uplift"]
                            ax2.text(v2+(.1 if v2>=0 else -.1),i2,f"{v2:+.1f} {r2['sig']}",
                                     va="center",ha="left" if v2>=0 else "right",fontsize=7.5,color=DARK)
                        ax2.set_xlabel(f"{metric_label} uplift (pp)",fontsize=8)
                        ax2.set_title(f"Same feature, different OUs",fontsize=9,fontweight="bold",color=DARK)
                        ax2.spines[["top","right"]].set_visible(False)
                        ax2.spines[["left","bottom"]].set_color("#DDD")
                        ax2.grid(axis="x",alpha=.3); plt.tight_layout()
                        st.pyplot(fig2); plt.close()

                    st.markdown('<div style="margin-top:.9rem"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Alerts for top feature</div>', unsafe_allow_html=True)
                    had = render_alerts(feature=top_feat, scope_filters=scope_filters)
                    if not had:
                        st.markdown("<span style='color:#AAA;font-size:.81rem'>No alerts for this feature.</span>",
                                    unsafe_allow_html=True)

            # Explorer
            st.markdown("""<div class="explorer-card">
              <div class="explorer-title">Explore any combination</div>
              <div class="explorer-sub">Pre-filled with the best combination. Change any feature
                to see all metrics update in real time. The scoreboard compares your selection
                to the scope mean.</div>""", unsafe_allow_html=True)

            ss = f"sel_{metric_col}_{hash(str(scope_filters))}_{selected_campaign_id}"
            if ss not in st.session_state:
                st.session_state[ss] = default_sel(combo)
            cur = st.session_state[ss]; new = dict(cur); changed = False

            st.markdown('<div class="feat-group">Asset features</div>', unsafe_allow_html=True)
            rows_feats = [ALL_FEATS[i:i+5] for i in range(0, len(ALL_FEATS), 5)]
            for row_feats in rows_feats:
                cols = st.columns(5)
                for ci, feat in enumerate(row_feats):
                    with cols[ci]:
                        if feat not in sub_df.columns: new[feat]="__any__"; continue
                        raws, disps = get_opts(feat, scope_key, metric_col, sub_df)
                        cur_raw = cur.get(feat,"__any__")
                        try: cur_idx = raws.index(cur_raw)
                        except ValueError: cur_idx=0
                        chosen_idx = st.selectbox(
                            FEAT_LABEL.get(feat,feat), range(len(raws)),
                            format_func=lambda i,d=disps: d[i], index=cur_idx,
                            key=f"d_{metric_col}_{feat}_{hash(str(scope_filters))}_{selected_campaign_id}",
                        )
                        chosen_raw = raws[chosen_idx]
                        new[feat] = chosen_raw
                        if chosen_raw != cur_raw: changed=True
                for ci in range(len(row_feats),5): cols[ci].empty()

            if changed: st.session_state[ss]=new; st.experimental_rerun()

            rc1,rc2,_ = st.columns([1,1,7])
            with rc1:
                if st.button("↺ Reset",key=f"rst_{metric_col}_{hash(str(scope_filters))}_{selected_campaign_id}"):
                    st.session_state[ss]=default_sel(combo); st.experimental_rerun()
            with rc2:
                if st.button("✕ Clear",key=f"clr_{metric_col}_{hash(str(scope_filters))}_{selected_campaign_id}"):
                    st.session_state[ss]={f:"__any__" for f in ALL_FEATS}; st.experimental_rerun()

            # Scoreboard
            scores = score_sel(sub_df, cur)
            bases  = {mc:round(sub_df[mc].dropna().mean()*100,2)
                      for mc in list(METRICS.keys())+["SCD_score"] if mc in sub_df.columns}
            sb = '<div style="display:flex;gap:.7rem;margin-top:1.1rem;flex-wrap:wrap">'
            for mc2,ml2 in list(METRICS.items())+[("SCD_score","SCD Score")]:
                val,n2 = scores.get(mc2,(None,0))
                base   = bases.get(mc2,0)
                hi     = " active" if mc2==metric_col else ""
                mc_col = METRIC_COLORS.get(ml2,DARK)
                if val is not None:
                    d=val-base; dc="d-up" if d>.1 else ("d-dn" if d<-.1 else "d-flat")
                    sb += (f'<div class="score-card{hi}" style="border-top:2px solid {mc_col};min-width:115px">'
                           f'<div class="score-metric">{ml2}</div><div class="score-val">{val:.1f}</div>'
                           f'<div class="score-delta {dc}">{"+" if d>0 else ""}{d:.1f}pp vs mean</div>'
                           f'<div class="score-n">n={n2:,}</div></div>')
                else:
                    sb += (f'<div class="score-card{hi}" style="border-top:2px solid {mc_col};min-width:115px">'
                           f'<div class="score-metric">{ml2}</div>'
                           f'<div class="score-val" style="font-size:.95rem;color:#D8D4CE">—</div>'
                           f'<div class="score-n">n=0</div></div>')
            st.markdown(sb+"</div>", unsafe_allow_html=True)

            _, mn = scores.get(metric_col,(None,0))
            if mn is not None and 0 < mn < 30:
                st.markdown(f'<div class="low-n">Only {mn} assets match — directional only.</div>',
                            unsafe_allow_html=True)

            # Insights for active selection
            active_feats_sel = [f for f,v in cur.items() if v not in ("__any__",None,"")]
            if active_feats_sel and has_insights:
                texts = []
                for f in active_feats_sel[:4]:
                    texts += catalog_insights(f, scope_filters, metric_col)
                if texts:
                    st.markdown('<div style="margin-top:.9rem;border-top:1px solid #EAE8E2;padding-top:.9rem">'
                                '<div class="section-label">Insights for your selection</div>',
                                unsafe_allow_html=True)
                    for t in texts[:4]:
                        st.markdown(f'<div class="insight-warn warn-insight">'
                                    f'<div class="insight-warn-icon">ℹ</div>'
                                    f'<div style="font-size:.82rem;color:#333">{t}</div></div>',
                                    unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                if active_feats_sel:
                    render_alerts(feature=active_feats_sel[0], scope_filters=scope_filters, max_items=3)

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Deep-Dive":
    st.markdown(f"""
    <div class="page-hero">
      <div class="page-eyebrow">Feature Analysis</div>
      <div class="page-title">Feature Deep-Dive</div>
      <div class="page-sub">How one feature performs across all metrics, all OUs, and all
        feature values. Includes automated alerts and catalog insights.</div>
    </div>""", unsafe_allow_html=True)

    feat_opts   = [(f, FEAT_LABEL.get(f,f)) for f in ALL_FEATS if f in sub_df.columns]
    feat_labels = [fl for _,fl in feat_opts]
    feat_cols   = [fc for fc,_ in feat_opts]
    sel_idx     = st.selectbox("Select feature", range(len(feat_labels)),
                               format_func=lambda i: feat_labels[i])
    sel_feat     = feat_cols[sel_idx]
    sel_feat_lbl = feat_labels[sel_idx]

    # Metric cards
    st.markdown('<div class="section-label">Performance across all metrics — this scope</div>',
                unsafe_allow_html=True)
    mcols = st.columns(4)
    for i,(mc,ml) in enumerate(zip(
        ["Attention_T2B","Persuasion_T2B","Likeability_Love_Like_T2B","SCD_score"],
        ["Attention","Persuasion","Likeability","SCD Score"]
    )):
        u,sig,n = compute_uplift(sub_df, sel_feat, mc)
        mc_col = METRIC_COLORS[ml]
        with mcols[i]:
            if u is not None:
                color = GREEN if u>=0 else RED; arrow = "▲" if u>=0 else "▼"
                st.markdown(f"""<div class="score-card" style="border-top:3px solid {mc_col}">
                  <div class="score-metric">{ml}</div>
                  <div class="score-val" style="color:{color}">{arrow} {abs(u):.1f}pp</div>
                  <div class="score-delta">{sig} · n={n:,}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="score-card" style="border-top:3px solid {mc_col}">
                  <div class="score-metric">{ml}</div>
                  <div class="score-val" style="color:#CCC">—</div>
                  <div class="score-delta" style="color:#CCC">Insufficient data</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.1rem"></div>', unsafe_allow_html=True)
    col_chart, col_alerts = st.columns([1.55,1])

    with col_chart:
        st.markdown('<div class="section-label">Effect by OU</div>', unsafe_allow_html=True)
        metric_dd = st.selectbox("Metric",["SCD Score","Attention","Persuasion","Likeability"],key="dd_ou")
        mc_dd = METRIC_COL_MAP[metric_dd]
        ou_rows2 = []
        for ou in sorted(df_full["operating_unit_code"].dropna().unique()):
            ou_sub2 = df_full[df_full["operating_unit_code"]==ou]
            if scope_filters:
                for t,v in scope_filters:
                    if t!="ou": ou_sub2=ou_sub2[ou_sub2[SCOPE_MAP[t]]==v]
            u2,sig2,n2 = compute_uplift(ou_sub2, sel_feat, mc_dd)
            if u2 is not None: ou_rows2.append({"OU":ou,"uplift":u2,"sig":sig2,"n":n2})
        if ou_rows2:
            ou_df2 = pd.DataFrame(ou_rows2).sort_values("uplift")
            g_u,_,_ = compute_uplift(df_full, sel_feat, mc_dd)
            fig3,ax3 = plt.subplots(figsize=(9, max(3.5, len(ou_df2)*.46)))
            fig3.patch.set_facecolor(LIGHT_BG); ax3.set_facecolor(LIGHT_BG)
            ax3.barh(ou_df2["OU"], ou_df2["uplift"],
                      color=[GREEN if v>=0 else RED for v in ou_df2["uplift"]],
                      alpha=.85, height=.6)
            ax3.axvline(0,color=DARK,lw=1.2,ls="--")
            if g_u is not None:
                ax3.axvline(g_u,color=BLUE,lw=1.5,ls=":",alpha=.7,label=f"Global: {g_u:+.1f}pp")
                ax3.legend(fontsize=8)
            for i3,(_, r3) in enumerate(ou_df2.iterrows()):
                v3=r3["uplift"]
                ax3.text(v3+(.12 if v3>=0 else -.12),i3,f"{v3:+.1f} {r3['sig']}",
                         va="center",ha="left" if v3>=0 else "right",fontsize=8,color=DARK)
            ax3.set_xlabel(f"{sel_feat_lbl} uplift on {metric_dd} (pp)\n"
                           f"Positive bars = feature associated with higher scores in that OU; "
                           f"blue dotted = global average",fontsize=8)
            ax3.set_title(f"{sel_feat_lbl} — {metric_dd} by OU",fontsize=9,fontweight="bold",color=DARK)
            ax3.spines[["top","right"]].set_visible(False); ax3.spines[["left","bottom"]].set_color("#DDD")
            ax3.grid(axis="x",alpha=.3); plt.tight_layout()
            st.pyplot(fig3); plt.close()

        # Feature values (categorical)
        if sel_feat in CAT_FEATS and sel_feat in sub_df.columns:
            st.markdown('<div style="margin-top:1.1rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Performance by feature value</div>', unsafe_allow_html=True)
            st.markdown("""<div style="font-size:.81rem;color:#888;margin-bottom:.65rem">
            Mean SCD score for each specific value of this feature, compared to the scope average (dashed).
            Use this to identify which specific values within this feature drive the best results.</div>""",
            unsafe_allow_html=True)
            scope_mean = sub_df["SCD_score"].dropna().mean()*100
            val_rows = []
            for val in sub_df[sel_feat].dropna().unique():
                if str(val).strip()=="": continue
                sv = sub_df[sub_df[sel_feat]==val]
                if len(sv)<5: continue
                mean_scd = sv["SCD_score"].dropna().mean()*100
                val_rows.append({"value":str(val)[:45],"diff":mean_scd-scope_mean,"n":len(sv)})
            if val_rows:
                val_df = pd.DataFrame(val_rows).sort_values("diff")
                fig4,ax4 = plt.subplots(figsize=(9, max(3.5, len(val_df)*.38)))
                fig4.patch.set_facecolor(LIGHT_BG); ax4.set_facecolor(LIGHT_BG)
                ax4.barh(val_df["value"], val_df["diff"],
                          color=[GREEN if v>=0 else RED for v in val_df["diff"]],
                          alpha=.85, height=.65)
                ax4.axvline(0,color=DARK,lw=1.2,ls="--")
                for i4,(_, r4) in enumerate(val_df.iterrows()):
                    v4=r4["diff"]
                    ax4.text(v4+(.1 if v4>=0 else -.1),i4,f"{v4:+.1f}pp  n={r4['n']}",
                             va="center",ha="left" if v4>=0 else "right",fontsize=7.5,color=DARK)
                ax4.set_xlabel("Difference from scope mean SCD (pp)",fontsize=8.5)
                ax4.set_title(f"{sel_feat_lbl}: which values perform best",
                              fontsize=9,fontweight="bold",color=DARK)
                ax4.spines[["top","right"]].set_visible(False); ax4.spines[["left","bottom"]].set_color("#DDD")
                ax4.grid(axis="x",alpha=.3); plt.tight_layout()
                st.pyplot(fig4); plt.close()

    # Combinations table below the two columns
    st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Best feature combinations</div>', unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:.82rem;color:#7A7670;margin-bottom:.9rem;line-height:1.6;max-width:760px">
    Which other features combine best with <strong>{sel_feat_lbl}</strong>?
    The table below shows the top pairings ranked by combined SCD score.
    <strong>Combined uplift</strong> is the expected SCD gain when an asset has
    <em>both</em> this feature and the partner feature, compared to the scope average.
    <strong>Synergy</strong> is how much extra gain the pairing delivers on top of
    the best individual feature alone — a positive synergy means the two features
    amplify each other.</div>""", unsafe_allow_html=True)

    combo_metric_dd = st.selectbox("Metric for combinations",
                                    ["SCD Score","Attention","Persuasion","Likeability"],
                                    key="combo_metric_dd")
    combo_mc = METRIC_COL_MAP[combo_metric_dd]
    combos = compute_feature_combinations(sub_df, sel_feat, combo_mc, top_n=6)
    if combos:
        ctbl = """<div style="overflow-x:auto;margin-bottom:.8rem">
        <table style="width:100%;border-collapse:collapse;font-size:.84rem;font-family:'Source Sans 3',sans-serif">
        <thead><tr style="border-bottom:2px solid #EAE8E2">
          <th style="text-align:left;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Partner feature</th>
          <th style="text-align:center;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Combined uplift</th>
          <th style="text-align:center;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">This feature alone</th>
          <th style="text-align:center;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Partner alone</th>
          <th style="text-align:center;padding:.5rem .7rem;color:#8E44AD;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Synergy</th>
          <th style="text-align:center;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">n assets</th>
          <th style="text-align:left;padding:.5rem .7rem;color:#6A6660;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600">Interpretation</th>
        </tr></thead><tbody>"""
        for ci, c in enumerate(combos):
            bg_c = "#FAFAF8" if ci % 2 == 0 else "#FFFFFF"
            cu   = c["combined_uplift"]
            syn  = c["synergy"]
            cu_color  = GREEN if cu >= 0 else RED
            syn_color = GREEN if syn >= 0 else RED
            syn_arrow = "▲" if syn >= 0 else "▼"
            cu_arrow  = "▲" if cu >= 0 else "▼"
            if syn >= 1.5:
                interp = "Strong synergy — better together than alone"
                interp_color = "#1A7040"
            elif syn >= 0:
                interp = "Additive — modest extra gain from pairing"
                interp_color = "#555"
            else:
                interp = "Diminishing returns — one feature may be sufficient"
                interp_color = "#906820"
            ctbl += (f'<tr style="background:{bg_c};border-bottom:1px solid #F0EEE8">'
                     f'<td style="padding:.45rem .7rem;font-weight:600;color:{DARK}">{c["partner_label"]}</td>'
                     f'<td style="text-align:center;font-weight:600;color:{cu_color}">{cu_arrow} {abs(cu):.1f}pp</td>'
                     f'<td style="text-align:center;color:{MID}">{c["solo_target"]:+.1f}pp</td>'
                     f'<td style="text-align:center;color:{MID}">{c["solo_partner"]:+.1f}pp</td>'
                     f'<td style="text-align:center;font-weight:600;color:{syn_color}">{syn_arrow} {abs(syn):.1f}pp</td>'
                     f'<td style="text-align:center;color:#AAA">{c["n"]}</td>'
                     f'<td style="padding:.45rem .7rem;font-size:.79rem;font-style:italic;color:{interp_color}">{interp}</td>'
                     f'</tr>')
        ctbl += """</tbody></table></div>
        <div style="font-size:.76rem;color:#AAA;margin-bottom:.5rem">
        Combined uplift = mean score when both features present, vs scope mean.
        Synergy = combined uplift minus the better of the two individual uplifts.</div>"""
        st.markdown(ctbl, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#AAA;font-size:.84rem">Not enough data to compute combinations in this scope.</p>',
                    unsafe_allow_html=True)

    with col_alerts:
        st.markdown('<div class="section-label">Rulebook alerts</div>', unsafe_allow_html=True)
        had2 = render_alerts(feature=sel_feat, scope_filters=scope_filters, max_items=8)
        if not had2:
            st.markdown("<span style='color:#AAA;font-size:.82rem'>No alerts for this feature.</span>",
                        unsafe_allow_html=True)
        if has_insights and catalog is not None:
            st.markdown('<div style="margin-top:1.1rem"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Catalog insights</div>', unsafe_allow_html=True)
            feat_cat = catalog[(catalog["feature"]==sel_feat) &
                               (catalog["confidence"].isin(["high","medium"]))
                               ].sort_values(["filter","evidence_uplift_pp"],ascending=[True,False])
            for _, r in feat_cat.head(12).iterrows():
                v=r["evidence_uplift_pp"]; bc=GREEN if v>=0 else RED
                st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
                  <div style="display:flex;justify-content:space-between">
                    <div class="insight-card-feature">{r["filter"]}: {r["filter_value"]}</div>
                    <div>{uplift_html(v)}</div>
                  </div>
                  <div class="insight-card-text">{r["text"]}</div>
                  <div class="insight-card-meta">
                    {conf_badge(r["confidence"])}{metric_badge(r["metric_display"])}
                    {sig_badge(r["evidence_sig"])}{flag_badge(r["quality_flags"])}
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHT CATALOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Insight Catalog":
    if not has_insights or catalog is None:
        st.warning("insight_catalog.csv not found. Run the insight pipeline first.")
        st.stop()

    st.markdown(f"""
    <div class="page-hero">
      <div class="page-eyebrow">Insight Catalog</div>
      <div class="page-title">All Insights</div>
      <div class="page-sub">Every finding from the pipeline, tagged with evidence and confidence.
        Filter and sort to find what's relevant to you.</div>
    </div>""", unsafe_allow_html=True)

    cf1,cf2,cf3,cf4 = st.columns(4)
    with cf1: cat_scope = st.selectbox("Scope",["All"]+sorted(catalog["filter"].unique().tolist()))
    with cf2: cat_metric = st.multiselect("Metric",sorted(catalog["metric_display"].unique()),
                                           default=sorted(catalog["metric_display"].unique()))
    with cf3: cat_conf = st.multiselect("Confidence",["high","medium","low"],default=["high","medium"])
    with cf4: cat_dir  = st.radio("Direction",["Both","Positive","Negative"])
    cat_sort = st.selectbox("Sort by",["Highest uplift","Lowest uplift","Confidence","Feature name"])

    fc = catalog.copy()
    if cat_scope!="All":
        if scope_filters:
            smap={"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
            sl=smap.get(scope_filters[0][0],scope_filters[0][0]); sv=scope_filters[0][1]
            fc=fc[(fc["filter"]==sl)&(fc["filter_value"]==sv)]
        else: fc=fc[fc["filter"]==cat_scope]
    if cat_metric: fc=fc[fc["metric_display"].isin(cat_metric)]
    if cat_conf:   fc=fc[fc["confidence"].isin(cat_conf)]
    if cat_dir=="Positive": fc=fc[fc["direction"]=="positive"]
    elif cat_dir=="Negative": fc=fc[fc["direction"]=="negative"]
    if cat_sort=="Highest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=False)
    elif cat_sort=="Lowest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=True)
    elif cat_sort=="Confidence":
        fc["_co"]=fc["confidence"].map({"high":0,"medium":1,"low":2})
        fc=fc.sort_values(["_co","evidence_uplift_pp"],ascending=[True,False])
    elif cat_sort=="Feature name": fc=fc.sort_values("feature_display")

    st.markdown(f"<span style='color:{MID};font-size:.84rem'>{len(fc):,} insights</span>",
                unsafe_allow_html=True)

    vmode = st.radio("View",["Cards","Table"])
    if vmode=="Cards":
        pg=max(1,int(np.ceil(len(fc)/20)))
        pnum=st.slider("Page",1,pg,1) if pg>1 else 1
        for _,r in fc.iloc[(pnum-1)*20:pnum*20].iterrows():
            v=r["evidence_uplift_pp"]; bc=GREEN if v>=0 else RED
            st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
              <div style="display:flex;justify-content:space-between">
                <div class="insight-card-feature">{r["feature_display"]}</div>
                <div>{uplift_html(v)}</div>
              </div>
              <div class="insight-card-text">{r["text"]}</div>
              <div class="insight-card-meta">
                {conf_badge(r["confidence"])}{metric_badge(r["metric_display"])}
                {dir_badge(r["direction"])}{sig_badge(r["evidence_sig"])}
                <span class="badge badge-metric">n={int(r["evidence_n_has"]):,}</span>
                <span class="badge badge-metric">base {r["evidence_baseline_pp"]:.1f}pp</span>
                {flag_badge(r["quality_flags"])}
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        sc2=["feature_display","metric_display","filter","filter_value","evidence_uplift_pp",
             "evidence_sig","confidence","effect_size","evidence_n_has","quality_flags"]
        st.dataframe(fc[[c for c in sc2 if c in fc.columns]].rename(columns={
            "feature_display":"Feature","metric_display":"Metric","filter":"Scope",
            "filter_value":"Scope value","evidence_uplift_pp":"Uplift (pp)","evidence_sig":"Sig.",
            "confidence":"Confidence","effect_size":"Effect size","evidence_n_has":"n (has)",
            "quality_flags":"Flags"}), height=600)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RULEBOOK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Rulebook":
    if not has_insights or rulebook is None:
        st.warning("rulebook.csv not found. Run the insight pipeline first.")
        st.stop()

    st.markdown(f"""
    <div class="page-hero">
      <div class="page-eyebrow">Automated Alerts</div>
      <div class="page-title">Rulebook</div>
      <div class="page-sub">Seven types of pattern detected automatically. High-severity entries
        should be reviewed before writing any brief.</div>
    </div>""", unsafe_allow_html=True)

    rc = rulebook["rule_type"].value_counts()
    rcols2 = st.columns(min(len(rc),7))
    for i,(rt,cnt) in enumerate(rc.items()):
        bg,accent = RULE_COLORS.get(rt,("#F9F9F9",MID))
        with rcols2[i%7]:
            st.markdown(f"""<div style="background:{bg};border:1px solid rgba(0,0,0,.08);
              border-left:3px solid {accent};border-radius:6px;padding:.85rem .95rem;
              margin-bottom:.7rem;text-align:center">
              <div style="font-size:1.45rem;font-weight:700;color:{accent}">{cnt}</div>
              <div style="font-size:.66rem;font-weight:700;color:{accent};
                text-transform:uppercase;letter-spacing:.06em">{rt}</div>
            </div>""", unsafe_allow_html=True)

    rf1,rf2,rf3 = st.columns(3)
    with rf1: rt_f=st.multiselect("Rule type",sorted(rulebook["rule_type"].unique()),
                                   default=sorted(rulebook["rule_type"].unique()))
    with rf2: sv_f=st.multiselect("Severity",["high","medium","low"],default=["high","medium"])
    with rf3: sc_f=st.multiselect("Scope",sorted(rulebook["scope"].unique()),
                                   default=sorted(rulebook["scope"].unique()))

    rb_f=rulebook.copy()
    if rt_f: rb_f=rb_f[rb_f["rule_type"].isin(rt_f)]
    if sv_f: rb_f=rb_f[rb_f["severity"].isin(sv_f)]
    if sc_f: rb_f=rb_f[rb_f["scope"].isin(sc_f)]

    st.markdown(f"<span style='color:{MID};font-size:.84rem'>{len(rb_f)} entries</span>",
                unsafe_allow_html=True)

    SEV_COL  = {"high":RED,"medium":AMBER,"low":GREEN}
    RULE_EXP = {
        "Conflict":          "This feature improves one metric but hurts another — choosing it involves a trade-off.",
        "Heterogeneity":     "The global average hides a reversal. Do not brief from the global number alone.",
        "Boundary Condition":"This feature works differently in Video vs Print. Format matters when briefing.",
        "Opportunity":       "High positive effect but under-used in the portfolio. Test more assets with it.",
        "Outlier":           "This scope is unusually different from its peers. Investigate before briefing.",
        "Consensus":         "Safe to brief — positive across all three metrics with no trade-offs detected.",
        "Anti-pattern":      "Consistently negative across all metrics in this scope. Brief against it.",
    }
    for _,r in rb_f.iterrows():
        bg,accent = RULE_COLORS.get(r["rule_type"],("#F9F9F9",MID))
        sc2 = SEV_COL.get(r.get("severity","medium"),MID)
        exp = RULE_EXP.get(r["rule_type"],"")
        st.markdown(f"""<div class="rule-card" style="background:{bg};border-left:4px solid {accent}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div class="rule-card-type" style="color:{accent}">{r["rule_type"]}</div>
            <span style="font-size:.67rem;font-weight:700;color:{sc2};
              text-transform:uppercase;letter-spacing:.05em">{r.get("severity","").upper()}</span>
          </div>
          <div class="rule-card-text">{r["text"]}</div>
          <div style="font-size:.77rem;color:#888;margin-top:.3rem;font-style:italic">{exp}</div>
          <div style="display:flex;gap:5px;margin-top:.45rem;flex-wrap:wrap">
            {badge(r.get("feature",""),"badge-metric")}
            {badge(r.get("scope",""),"badge-scope")}
            {badge(r.get("scope_value","") or "Global","badge-low")}
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Heatmaps":
    if uplift_df is None:
        st.warning("uplift_all_scopes.csv not found. Run the insight pipeline first.")
        st.stop()

    st.markdown(f"""
    <div class="page-hero">
      <div class="page-eyebrow">Visual Overview</div>
      <div class="page-title">Uplift Heatmaps</div>
      <div class="page-sub">Red cells = feature associated with lower scores. Green = higher.
        White = no effect. Grey = insufficient data. Values are percentage points.</div>
    </div>""", unsafe_allow_html=True)

    FEAT_DISP = ({r["feature"]:r["feature_display"]
                  for _,r in catalog[["feature","feature_display"]].drop_duplicates().iterrows()}
                 if catalog is not None else {})
    MDISP = {"Attention_T2B":"Attention","Persuasion_T2B":"Persuasion",
              "Likeability_Love_Like_T2B":"Likeability","SCD_score":"SCD Score"}

    tab1,tab2,tab3,tab4 = st.tabs(["Global × Metrics","By OU","By Category","By Brand / Market"])

    with tab1:
        st.markdown("""<div style="font-size:.82rem;color:#888;margin-bottom:.7rem">
        Each cell is the uplift in pp for that feature × metric globally.
        A green cell means assets with this feature score higher; red means lower.
        Sorted by SCD Score impact from highest to lowest.</div>""", unsafe_allow_html=True)
        gu = uplift_df[uplift_df["scope"]=="Global"].copy()
        if not gu.empty:
            piv=gu.pivot_table(index="feature",columns="metric",values="uplift_pp")
            piv.index=[FEAT_DISP.get(f,f) for f in piv.index]
            piv.columns=[MDISP.get(c,c) for c in piv.columns]
            if "SCD Score" in piv.columns: piv=piv.sort_values("SCD Score",ascending=False)
            st.pyplot(make_heatmap(piv,"Global Feature Uplift (pp)",figsize=(12,8))); plt.close()

    with tab2:
        m_ou=st.selectbox("Metric",list(MDISP.values()),key="hm_ou")
        mc_ou={v:k for k,v in MDISP.items()}[m_ou]
        st.markdown(f"""<div style="font-size:.82rem;color:#888;margin-bottom:.7rem">
        How each feature performs in each OU on {m_ou}. OUs with very different colours from
        the global pattern are flagged in the Rulebook as Heterogeneity entries.</div>""",
        unsafe_allow_html=True)
        ou_u=uplift_df[(uplift_df["scope"]=="OU")&(uplift_df["metric"]==mc_ou)].copy()
        if not ou_u.empty:
            ou_u["fd"]=ou_u["feature"].apply(lambda f: FEAT_DISP.get(f,f))
            piv_ou=ou_u.pivot_table(index="scope_value",columns="fd",values="uplift_pp")
            st.pyplot(make_heatmap(piv_ou,f"OU Uplift — {m_ou} (pp)",figsize=(14,6))); plt.close()

    with tab3:
        m_cat=st.selectbox("Metric",list(MDISP.values()),key="hm_cat")
        mc_cat={v:k for k,v in MDISP.items()}[m_cat]
        st.markdown(f"""<div style="font-size:.82rem;color:#888;margin-bottom:.7rem">
        Features by product category on {m_cat}. Some features that work globally may
        reverse for specific categories — check the Rulebook for confirmed reversals.</div>""",
        unsafe_allow_html=True)
        cat_u=uplift_df[(uplift_df["scope"]=="Category")&(uplift_df["metric"]==mc_cat)].copy()
        if not cat_u.empty:
            cat_u["fd"]=cat_u["feature"].apply(lambda f: FEAT_DISP.get(f,f))
            piv_cat=cat_u.pivot_table(index="scope_value",columns="fd",values="uplift_pp")
            st.pyplot(make_heatmap(piv_cat,f"Category Uplift — {m_cat} (pp)",figsize=(14,6))); plt.close()

    with tab4:
        hc1,hc2=st.columns(2)
        with hc1: split_t=st.radio("Split by",["Brand","Market"])
        with hc2: m_bm=st.selectbox("Metric",list(MDISP.values()),key="hm_bm")
        mc_bm={v:k for k,v in MDISP.items()}[m_bm]
        st.markdown(f"""<div style="font-size:.82rem;color:#888;margin-bottom:.7rem">
        Feature effects by {split_t.lower()} on {m_bm}. Showing top 15 by data volume.</div>""",
        unsafe_allow_html=True)
        bm_u=uplift_df[(uplift_df["scope"]==split_t)&(uplift_df["metric"]==mc_bm)].copy()
        if not bm_u.empty:
            top_vals=bm_u.groupby("scope_value").size().nlargest(15).index
            bm_u=bm_u[bm_u["scope_value"].isin(top_vals)]
            bm_u["fd"]=bm_u["feature"].apply(lambda f: FEAT_DISP.get(f,f))
            piv_bm=bm_u.pivot_table(index="scope_value",columns="fd",values="uplift_pp")
            st.pyplot(make_heatmap(piv_bm,f"{split_t} Uplift — {m_bm} (pp)",figsize=(14,7))); plt.close()
        else:
            st.info("No data for this selection.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>Uplift = mean(present) &minus; mean(absent) &nbsp;&middot;&nbsp;
    Significance: Mann-Whitney U &nbsp;&middot;&nbsp; pp = percentage points &nbsp;&middot;&nbsp;
    SCD = 10% See + 30% Connect + 60% Do</span>
  <span>The Coca&#8209;Cola Company &nbsp;&middot;&nbsp; Asset Intelligence</span>
</div>
""", unsafe_allow_html=True)
