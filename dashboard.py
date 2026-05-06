import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="MIS Dashboard", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "output")

# ============================================================
# VERI YUKLEME
# ============================================================
@st.cache_data
def load_data():
    customer = pd.read_csv(os.path.join(DATA_DIR, "dm_dim_customer.csv"), sep=";")
    currency = pd.read_csv(os.path.join(DATA_DIR, "dm_dim_currency.csv"), sep=";")
    credit = pd.read_csv(os.path.join(DATA_DIR, "dm_dim_credit.csv"), sep=";")
    payment = pd.read_csv(os.path.join(DATA_DIR, "dm_dim_payment.csv"), sep=";")
    risk = pd.read_csv(os.path.join(DATA_DIR, "dw_credit_risk.csv"), sep=";")
    commitment = pd.read_csv(os.path.join(DATA_DIR, "dw_commitment.csv"), sep=";")
    cust_limit = pd.read_csv(os.path.join(DATA_DIR, "dw_cust_limit.csv"), sep=";")
    collateral = pd.read_csv(os.path.join(DATA_DIR, "dw_collateral.csv"), sep=";")
    usage = pd.read_csv(os.path.join(DATA_DIR, "dw_collateral_usage.csv"), sep=";")
    return customer, currency, credit, payment, risk, commitment, cust_limit, collateral, usage

customer, currency, credit, payment, risk, commitment, cust_limit, collateral, usage = load_data()

# Musteri bilgilerini krediye ekle
credit_full = credit.merge(customer[["mno", "kobi_flg", "sube", "portfoy", "region"]], on="mno", how="left")
credit_full["valor_dt"] = pd.to_datetime(credit_full["valor"])
credit_full["ceyrek"] = credit_full["valor_dt"].dt.to_period("Q").astype(str)

risk_full = risk.merge(credit[["krd_id", "krd_program", "odeme_sablonu", "yapilandirma", "teminat_tur_id", "teminat"]], on="krd_id", how="left")
risk_full = risk_full.merge(customer[["mno", "kobi_flg", "sube", "portfoy", "region"]], on="mno", how="left")
risk_full["valor_dt"] = pd.to_datetime(risk_full["valor"])
risk_full["ceyrek"] = risk_full["valor_dt"].dt.to_period("Q").astype(str)

# ============================================================
# NAVIGATION
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "ana_sayfa"

def go_to(page):
    st.session_state.page = page

# ============================================================
# FILTRE FONKSIYONU (sidebar)
# ============================================================
def render_filters(df, prefix="main"):
    st.sidebar.markdown("### Filtreler")
    portfoy = st.sidebar.multiselect("Portfoy", sorted(df["portfoy"].unique()), key=f"{prefix}_portfoy")
    sube = st.sidebar.multiselect("Sube", sorted(df["sube"].unique()), key=f"{prefix}_sube")
    bolge = st.sidebar.multiselect("Bolge", sorted(df["region"].unique()), key=f"{prefix}_bolge")
    kobi = st.sidebar.multiselect("KOBi Durumu", [0, 1], format_func=lambda x: "KOBi" if x == 1 else "Kurumsal", key=f"{prefix}_kobi")

    mask = pd.Series(True, index=df.index)
    if portfoy:
        mask &= df["portfoy"].isin(portfoy)
    if sube:
        mask &= df["sube"].isin(sube)
    if bolge:
        mask &= df["region"].isin(bolge)
    if kobi:
        mask &= df["kobi_flg"].isin(kobi)
    return mask

def back_button():
    st.sidebar.markdown("---")
    if st.sidebar.button("Ana Sayfaya Don", use_container_width=True):
        go_to("ana_sayfa")
        st.rerun()

# ============================================================
# ANA SAYFA - X JARFI DUZENI
# ============================================================
def page_ana_sayfa():
    st.markdown("""
    <style>
    div.stButton > button {
        height: 180px;
        font-size: 28px;
        font-weight: bold;
        border-radius: 16px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center; margin-bottom:60px;'>MIS Dashboard</h1>", unsafe_allow_html=True)

    # X jarfi: ust sol - ust sag / alt sol - alt sag
    row1 = st.columns([1, 0.3, 1])
    with row1[0]:
        if st.button("KREDI", use_container_width=True, key="btn_kredi"):
            go_to("kredi")
            st.rerun()
    with row1[2]:
        if st.button("TEMINAT", use_container_width=True, key="btn_teminat"):
            go_to("teminat")
            st.rerun()

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    row2 = st.columns([1, 0.3, 1])
    with row2[0]:
        if st.button("HAZINE", use_container_width=True, key="btn_hazine"):
            go_to("hazine")
            st.rerun()
    with row2[2]:
        if st.button("LIMIT", use_container_width=True, key="btn_limit"):
            go_to("limit")
            st.rerun()


# ============================================================
# KREDI SAYFASI
# ============================================================
def page_kredi():
    st.title("Kredi Risk ve Kullandirim")
    back_button()
    mask = render_filters(risk_full, prefix="kredi")
    df = risk_full[mask].copy()

    # --- KPI'lar ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Kredi Adedi", f"{len(df):,}")
    col2.metric("Toplam Anapara Risk", f"{df['anapara_risk'].sum()/1e6:,.1f} M")
    col3.metric("Toplam Faiz Risk", f"{df['faiz_risk'].sum()/1e6:,.1f} M")
    col4.metric("Yapilandirma Adedi", f"{df[df['yapilandirma']==1].shape[0]:,}")

    st.markdown("---")

    # 1. Programa Gore Risk ve Kullandirim
    st.subheader("Programa Gore Risk ve Kullandirim")
    prog = df.groupby("krd_program").agg(
        anapara_risk=("anapara_risk", "sum"),
        faiz_risk=("faiz_risk", "sum"),
        adet=("krd_id", "count"),
    ).reset_index()
    prog["anapara_risk_m"] = (prog["anapara_risk"] / 1e6).round(1)
    prog["faiz_risk_m"] = (prog["faiz_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            prog.melt(id_vars="krd_program", value_vars=["anapara_risk_m", "faiz_risk_m"],
                       var_name="tip", value_name="tutar"),
            x="krd_program", y="tutar", color="tip", barmode="group",
            labels={"krd_program": "Program", "tutar": "Milyon TL", "tip": "Risk Tipi"},
            color_discrete_map={"anapara_risk_m": "#EF553B", "faiz_risk_m": "#636EFA"},
        )
        fig.for_each_trace(lambda t: t.update(name="Anapara" if "anapara" in t.name else "Faiz"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(prog, values="adet", names="krd_program",
                     title="Kredi Adet Dagilimi", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2. KOBi Durumuna Gore
    st.subheader("KOBi Durumuna Gore Risk")
    kobi_df = df.copy()
    kobi_df["kobi_label"] = kobi_df["kobi_flg"].map({0: "Kurumsal", 1: "KOBi"})
    kobi_grp = kobi_df.groupby("kobi_label").agg(
        anapara_risk=("anapara_risk", "sum"),
        faiz_risk=("faiz_risk", "sum"),
        adet=("krd_id", "count"),
    ).reset_index()
    kobi_grp["anapara_risk_m"] = (kobi_grp["anapara_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(kobi_grp, x="kobi_label", y="anapara_risk_m", color="kobi_label",
                     labels={"kobi_label": "", "anapara_risk_m": "Anapara Risk (M TL)"},
                     color_discrete_map={"KOBi": "#00CC96", "Kurumsal": "#AB63FA"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(kobi_grp, values="adet", names="kobi_label",
                     title="Adet Dagilimi", hole=0.4,
                     color_discrete_map={"KOBi": "#00CC96", "Kurumsal": "#AB63FA"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Yapilandirilmis Krediler
    st.subheader("Yapilandirilmis Krediler")
    yapil = df.copy()
    yapil["yapil_label"] = yapil["yapilandirma"].map({0: "Normal", 1: "Yapilandirma"})
    yapil_grp = yapil.groupby("yapil_label").agg(
        anapara_risk=("anapara_risk", "sum"),
        adet=("krd_id", "count"),
    ).reset_index()
    yapil_grp["anapara_risk_m"] = (yapil_grp["anapara_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(yapil_grp, x="yapil_label", y="anapara_risk_m", color="yapil_label",
                     labels={"yapil_label": "", "anapara_risk_m": "Anapara Risk (M TL)"},
                     color_discrete_map={"Normal": "#636EFA", "Yapilandirma": "#EF553B"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # programa gore yapilandirma
        yapil_prog = yapil[yapil["yapilandirma"] == 1].groupby("krd_program").agg(
            anapara_risk=("anapara_risk", "sum"), adet=("krd_id", "count")
        ).reset_index()
        yapil_prog["anapara_risk_m"] = (yapil_prog["anapara_risk"] / 1e6).round(1)
        fig = px.bar(yapil_prog, x="krd_program", y="anapara_risk_m",
                     title="Yapilandirma - Programa Gore",
                     labels={"krd_program": "Program", "anapara_risk_m": "Risk (M TL)"},
                     color_discrete_sequence=["#EF553B"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 4. Odeme Sablonuna Gore Ceyreklik Risk Projeksiyonu
    st.subheader("Odeme Sablonuna Gore Ceyreklik Risk Projeksiyonu")
    proj = df.groupby(["ceyrek", "odeme_sablonu"])["anapara_risk"].sum().reset_index()
    proj["anapara_risk_m"] = (proj["anapara_risk"] / 1e6).round(1)

    fig = px.bar(proj, x="ceyrek", y="anapara_risk_m", color="odeme_sablonu",
                 barmode="stack",
                 labels={"ceyrek": "Ceyrek", "anapara_risk_m": "Anapara Risk (M TL)",
                          "odeme_sablonu": "Odeme Sablonu"})
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TEMINAT SAYFASI
# ============================================================
def page_teminat():
    st.title("Teminat Analizi")
    back_button()

    # Teminat verisine musteri bilgisi ekle
    coll_full = collateral.merge(customer[["mno", "kobi_flg", "sube", "portfoy", "region"]],
                                  left_on="cust_id", right_on="mno", how="left")
    mask = render_filters(coll_full, prefix="teminat")
    coll_f = coll_full[mask].copy()
    filtered_custs = coll_f["cust_id"].unique()

    # usage + risk birlestir (ayni filtre uygulanir)
    usage_risk = usage.merge(collateral[["collateral_id", "finans_kurulusu", "tutar", "teminat_tur_id", "vade", "cust_id"]],
                              on="collateral_id", how="left")
    usage_risk = usage_risk.merge(risk[["krd_id", "anapara_risk"]], left_on="credit_id", right_on="krd_id", how="left")
    usage_risk = usage_risk.merge(credit[["krd_id", "yapilandirma"]], on="krd_id", how="left")
    ur_f = usage_risk[usage_risk["cust_id"].isin(filtered_custs)].copy()

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Teminat Adedi", f"{len(coll_f):,}")
    col2.metric("Toplam Teminat Tutari", f"{coll_f['tutar'].sum()/1e6:,.1f} M")
    col3.metric("Toplam Risk Karsiligi", f"{ur_f['anapara_risk'].sum()/1e6:,.1f} M")

    st.markdown("---")

    # 1. Bankalarda Teminat Yogunlasmasi
    st.subheader("Bankalarda Teminat Yogunlasmasi")
    bank_conc = coll_f.groupby("finans_kurulusu").agg(
        toplam_tutar=("tutar", "sum"), adet=("collateral_id", "count")
    ).reset_index()
    bank_conc["tutar_m"] = (bank_conc["toplam_tutar"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(bank_conc.sort_values("tutar_m", ascending=True),
                     x="tutar_m", y="finans_kurulusu", orientation="h",
                     labels={"finans_kurulusu": "", "tutar_m": "Teminat Tutari (M TL)"},
                     color="finans_kurulusu")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(bank_conc, values="adet", names="finans_kurulusu",
                     title="Adet Bazinda Yogunlasma", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2. Risk Coverage Oranlari (teminat turu bazinda)
    st.subheader("Risk Coverage Oranlari")
    TEMINAT_MAP = {100: "Banka Teminat Mektubu", 200: "Ipotek", 300: "Kefalet"}
    cov = ur_f.groupby("teminat_tur_id").agg(
        teminat_toplam=("tutar", "sum"),
        risk_toplam=("anapara_risk", "sum"),
    ).reset_index()
    cov["teminat_adi"] = cov["teminat_tur_id"].map(TEMINAT_MAP)
    cov["coverage"] = (cov["teminat_toplam"] / cov["risk_toplam"].replace(0, np.nan) * 100).round(1)
    cov["teminat_m"] = (cov["teminat_toplam"] / 1e6).round(1)
    cov["risk_m"] = (cov["risk_toplam"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Teminat", x=cov["teminat_adi"], y=cov["teminat_m"], marker_color="#00CC96"))
        fig.add_trace(go.Bar(name="Risk", x=cov["teminat_adi"], y=cov["risk_m"], marker_color="#EF553B"))
        fig.update_layout(barmode="group", height=400,
                          yaxis_title="Milyon TL", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(cov, x="teminat_adi", y="coverage",
                     labels={"teminat_adi": "", "coverage": "Coverage (%)"},
                     color="coverage", color_continuous_scale=["#EF553B", "#FFA15A", "#00CC96"],
                     text="coverage")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Serbest Kalacak Teminat Tutarlari (vadeye gore)
    st.subheader("Serbest Kalacak Teminat Tutarlari")
    coll_f["vade_dt"] = pd.to_datetime(coll_f["vade"])
    coll_f["vade_ceyrek"] = coll_f["vade_dt"].dt.to_period("Q").astype(str)
    serbest = coll_f.groupby("vade_ceyrek")["tutar"].sum().reset_index()
    serbest["tutar_m"] = (serbest["tutar"] / 1e6).round(1)

    fig = px.area(serbest, x="vade_ceyrek", y="tutar_m",
                  labels={"vade_ceyrek": "Vade Ceyregi", "tutar_m": "Serbest Kalacak Tutar (M TL)"},
                  color_discrete_sequence=["#636EFA"])
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 4. Teminatlara Gore Yapilandirma
    st.subheader("Teminat Turune Gore Yapilandirma")
    yapil_tem = ur_f.copy()
    yapil_tem["yapil_label"] = yapil_tem["yapilandirma"].map({0: "Normal", 1: "Yapilandirma"})
    yapil_tem["teminat_adi"] = yapil_tem["teminat_tur_id"].map(TEMINAT_MAP)
    yt = yapil_tem.groupby(["teminat_adi", "yapil_label"])["anapara_risk"].sum().reset_index()
    yt["risk_m"] = (yt["anapara_risk"] / 1e6).round(1)

    fig = px.bar(yt, x="teminat_adi", y="risk_m", color="yapil_label",
                 barmode="stack",
                 labels={"teminat_adi": "", "risk_m": "Risk (M TL)", "yapil_label": ""},
                 color_discrete_map={"Normal": "#636EFA", "Yapilandirma": "#EF553B"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# HAZINE SAYFASI
# ============================================================
def page_hazine():
    st.title("Hazine - Kur Analizi")
    back_button()

    curr = currency.copy()
    curr["ay_dt"] = pd.to_datetime(curr["ay"] + "-01")

    # KPI: son ay kurlari
    son_ay = curr["ay"].max()
    son_kurlar = curr[curr["ay"] == son_ay]

    cols = st.columns(4)
    for i, pb in enumerate(["USD", "EUR", "TRY", "CNY"]):
        row = son_kurlar[son_kurlar["pb1"] == pb]
        if len(row) > 0:
            cols[i].metric(f"{pb}/TRY", f"{row['rate'].values[0]:,.4f}")

    st.markdown("---")

    # 1. Kurlarin Gelisimi
    st.subheader("Kur Gelisimi")
    fx = curr[curr["pb1"] != "TRY"].copy()

    fig = px.line(fx, x="ay_dt", y="rate", color="pb1",
                  labels={"ay_dt": "", "rate": "Kur (TL)", "pb1": "Para Birimi"},
                  markers=True)
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2. Kur Projeksiyonu (basit linear trend)
    st.subheader("Kur Projeksiyonu (Onumuzdeki 6 Ay)")

    proj_months = pd.date_range(curr["ay_dt"].max() + pd.DateOffset(months=1), periods=6, freq="MS")

    fig = go.Figure()
    colors = {"USD": "#636EFA", "EUR": "#EF553B", "CNY": "#00CC96"}

    for pb in ["USD", "EUR", "CNY"]:
        pb_data = curr[curr["pb1"] == pb].sort_values("ay_dt")
        x_num = np.arange(len(pb_data))
        y_val = pb_data["rate"].values
        # linear regression
        coeffs = np.polyfit(x_num, y_val, 1)
        # projeksiyon
        proj_x = np.arange(len(pb_data), len(pb_data) + 6)
        proj_y = np.polyval(coeffs, proj_x)

        # gercek veri
        fig.add_trace(go.Scatter(
            x=pb_data["ay_dt"], y=y_val,
            mode="lines+markers", name=f"{pb} (Gercek)",
            line=dict(color=colors[pb]),
        ))
        # projeksiyon
        fig.add_trace(go.Scatter(
            x=proj_months, y=proj_y,
            mode="lines+markers", name=f"{pb} (Projeksiyon)",
            line=dict(color=colors[pb], dash="dash"),
        ))

    fig.update_layout(height=500, yaxis_title="Kur (TL)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# LIMIT SAYFASI
# ============================================================
def page_limit():
    st.title("Limit Analizi")
    back_button()

    lim = cust_limit.merge(customer[["mno", "kobi_flg", "portfoy"]], on="mno", how="left")
    mask = render_filters(lim, prefix="limit")
    lim_f = lim[mask].copy()

    # Her musteri icin toplam risk hesapla
    cust_risk = risk.groupby("mno")["anapara_risk"].sum().reset_index().rename(columns={"anapara_risk": "toplam_risk"})
    lim_f = lim_f.merge(cust_risk, on="mno", how="left").fillna({"toplam_risk": 0})
    lim_f["doluluk"] = (lim_f["toplam_risk"] / lim_f["genel_limit"].replace(0, np.nan) * 100).round(1)

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Musteri Sayisi", f"{len(lim_f):,}")
    col2.metric("Toplam Genel Limit", f"{lim_f['genel_limit'].sum()/1e6:,.1f} M")
    col3.metric("Toplam Risk", f"{lim_f['toplam_risk'].sum()/1e6:,.1f} M")
    col4.metric("Ort. Doluluk", f"{lim_f['doluluk'].mean():,.1f}%")

    st.markdown("---")

    # 1. Limit vs Risk (scatter)
    st.subheader("Musteri Bazinda Limit vs Risk")
    fig = px.scatter(lim_f, x="genel_limit", y="toplam_risk",
                     hover_data=["mno", "sube", "doluluk"],
                     color="region",
                     labels={"genel_limit": "Genel Limit (TL)", "toplam_risk": "Toplam Risk (TL)",
                             "region": "Bolge"},
                     opacity=0.7)
    # referans cizgisi (limit = risk)
    max_val = max(lim_f["genel_limit"].max(), lim_f["toplam_risk"].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                             mode="lines", name="Limit = Risk",
                             line=dict(color="red", dash="dash")))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2. Limit Gruplarina Gore Doluluk
    st.subheader("Limit Gruplarina Gore Doluluk Oranlari")

    # Her grup icin toplam limit ve toplam risk
    grup_data = []
    for grup, col in [("1. Grup", "bir_grup_limit"), ("2. Grup", "iki_grup_limit"), ("3. Grup", "uc_grup_limit")]:
        toplam_limit = lim_f[col].sum()
        toplam_risk = lim_f["toplam_risk"].sum()
        # risk dagilimini gruba gore oranla
        oran = lim_f[col].sum() / lim_f["genel_limit"].sum() if lim_f["genel_limit"].sum() > 0 else 0
        grup_risk = toplam_risk * oran
        doluluk = (grup_risk / toplam_limit * 100) if toplam_limit > 0 else 0
        grup_data.append({
            "grup": grup,
            "limit_m": round(toplam_limit / 1e6, 1),
            "risk_m": round(grup_risk / 1e6, 1),
            "doluluk": round(doluluk, 1),
        })

    grup_df = pd.DataFrame(grup_data)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Limit", x=grup_df["grup"], y=grup_df["limit_m"], marker_color="#636EFA"))
        fig.add_trace(go.Bar(name="Risk", x=grup_df["grup"], y=grup_df["risk_m"], marker_color="#EF553B"))
        fig.update_layout(barmode="group", height=400, yaxis_title="Milyon TL")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=grup_df["doluluk"].mean(),
            title={"text": "Ortalama Doluluk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#636EFA"},
                "steps": [
                    {"range": [0, 50], "color": "#d4edda"},
                    {"range": [50, 80], "color": "#fff3cd"},
                    {"range": [80, 100], "color": "#f8d7da"},
                ],
            },
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Doluluk dagilimi histogram
    st.subheader("Musteri Bazinda Doluluk Dagilimi")
    fig = px.histogram(lim_f, x="doluluk", nbins=20,
                       labels={"doluluk": "Doluluk Orani (%)", "count": "Musteri Sayisi"},
                       color_discrete_sequence=["#636EFA"])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ROUTING
# ============================================================
pages = {
    "ana_sayfa": page_ana_sayfa,
    "kredi": page_kredi,
    "teminat": page_teminat,
    "hazine": page_hazine,
    "limit": page_limit,
}

pages[st.session_state.page]()
