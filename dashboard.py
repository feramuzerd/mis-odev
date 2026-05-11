import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

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
# PD MODELi - Temerr\u00fct Riski Skoru (Lojistik Regresyon)
# ============================================================
@st.cache_data
def compute_pd_scores(_customer, _credit, _risk):
    merged = _risk.merge(_credit[["krd_id", "krd_program", "yapilandirma"]], on="krd_id", how="left")
    merged = merged.merge(_customer[["mno", "kobi_flg", "kurulus_year", "region"]], on="mno", how="left")
    cust_agg = merged.groupby("mno").agg(
        kobi_flg=("kobi_flg", "first"),
        kurulus_year=("kurulus_year", "first"),
        region=("region", "first"),
        lcy_tutar=("lcy_tutar", "sum"),
        yapilandirma_count=("yapilandirma", "sum"),
        kredi_adet=("krd_id", "count"),
    ).reset_index()
    region_dummies = pd.get_dummies(cust_agg["region"], prefix="region")
    features = pd.concat([cust_agg[["kobi_flg", "kurulus_year", "lcy_tutar"]], region_dummies], axis=1)
    yapilandirma_oran = cust_agg["yapilandirma_count"] / cust_agg["kredi_adet"]
    y = (yapilandirma_oran > yapilandirma_oran.median()).astype(int)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features.values)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    proba = model.predict_proba(X_scaled)[:, 1]
    cust_agg["pd_skor"] = (proba * 100).round(1)
    return cust_agg[["mno", "pd_skor"]]

pd_scores = compute_pd_scores(customer, credit, risk)

# ============================================================
# NAVIGATION
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "ana_sayfa"
if "drill_region" not in st.session_state:
    st.session_state.drill_region = None
if "drill_sube" not in st.session_state:
    st.session_state.drill_sube = None
if "drill_portfoy" not in st.session_state:
    st.session_state.drill_portfoy = None

def go_to(page, region=None, sube=None, portfoy=None):
    st.session_state.page = page
    if region is not None:
        st.session_state.drill_region = region
    if sube is not None:
        st.session_state.drill_sube = sube
    if portfoy is not None:
        st.session_state.drill_portfoy = portfoy

# ============================================================
# FILTRE FONKSIYONU (sidebar)
# ============================================================
def render_filters(df, prefix="main"):
    st.sidebar.markdown("### Filtreler")

    # Filtre secenekleri her zaman customer tablosundan (tutarlilik icin)
    ref = customer.copy()

    # Once bolge secilir, diger filtreler buna gore daraltilir
    bolge = st.sidebar.multiselect("Bolge", sorted(ref["region"].unique()), key=f"{prefix}_bolge")
    ref_after_bolge = ref[ref["region"].isin(bolge)] if bolge else ref

    kobi = st.sidebar.multiselect("KOBi Durumu",
                                  sorted(ref_after_bolge["kobi_flg"].unique()),
                                  format_func=lambda x: "KOBi" if x == 1 else "Kurumsal",
                                  key=f"{prefix}_kobi")
    ref_after_kobi = ref_after_bolge[ref_after_bolge["kobi_flg"].isin(kobi)] if kobi else ref_after_bolge

    sube = st.sidebar.multiselect("Sube", sorted(ref_after_kobi["sube"].unique()), key=f"{prefix}_sube")
    ref_after_sube = ref_after_kobi[ref_after_kobi["sube"].isin(sube)] if sube else ref_after_kobi

    portfoy = st.sidebar.multiselect("Portfoy", sorted(ref_after_sube["portfoy"].unique()), key=f"{prefix}_portfoy")

    mask = pd.Series(True, index=df.index)
    if bolge:
        mask &= df["region"].isin(bolge)
    if kobi:
        mask &= df["kobi_flg"].isin(kobi)
    if sube:
        mask &= df["sube"].isin(sube)
    if portfoy:
        mask &= df["portfoy"].isin(portfoy)
    return mask

def back_button(target="ana_sayfa", label="Ana Sayfaya Don", **kwargs):
    st.sidebar.markdown("---")
    if st.sidebar.button(label, use_container_width=True, key=f"back_{target}"):
        go_to(target, **kwargs)
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

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    row3 = st.columns([1, 0.3, 1])
    with row3[1]:
        if st.button("MUSTERI", use_container_width=True, key="btn_musteri"):
            go_to("musteri")
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

    # --- Kredi Kullandirim: Region Bazinda (drill-down) ---
    st.subheader("Kredi Kullandirim - Bolge Bazinda")
    st.caption("Bolgeye tiklayin detay icin")
    kull_mask = pd.Series(True, index=credit_full.index)
    if st.session_state.get("kredi_portfoy"):
        kull_mask &= credit_full["portfoy"].isin(st.session_state["kredi_portfoy"])
    if st.session_state.get("kredi_sube"):
        kull_mask &= credit_full["sube"].isin(st.session_state["kredi_sube"])
    if st.session_state.get("kredi_bolge"):
        kull_mask &= credit_full["region"].isin(st.session_state["kredi_bolge"])
    if st.session_state.get("kredi_kobi"):
        kull_mask &= credit_full["kobi_flg"].isin(st.session_state["kredi_kobi"])
    kull_df = credit_full[kull_mask].copy()
    kull_region = kull_df.groupby("region").agg(
        kullandirim=("lcy_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    kull_region["kullandirim_m"] = (kull_region["kullandirim"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(kull_region.sort_values("kullandirim_m", ascending=True),
                     x="kullandirim_m", y="region", orientation="h",
                     labels={"region": "Bolge", "kullandirim_m": "Kullandirim (M TL)"},
                     color="region", text="kullandirim_m")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(kull_region, values="adet", names="region",
                     title="Kullandirim Adet Dagilimi", hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(kull_region))
    for i, row in enumerate(kull_region.sort_values("region").itertuples()):
        with cols[i]:
            if st.button(f"{row.region}", key=f"kull_reg_{row.region}", use_container_width=True):
                go_to("kredi_kull_region", region=row.region)
                st.rerun()

    st.markdown("---")

    # --- Kredi Risk: Region Bazinda (drill-down) ---
    st.subheader("Kredi Risk - Bolge Bazinda")
    st.caption("Bolgeye tiklayin detay icin")
    risk_region = df.groupby("region").agg(
        anapara_risk=("anapara_risk", "sum"),
        faiz_risk=("faiz_risk", "sum"),
        adet=("krd_id", "count"),
    ).reset_index()
    risk_region["anapara_risk_m"] = (risk_region["anapara_risk"] / 1e6).round(1)
    risk_region["faiz_risk_m"] = (risk_region["faiz_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            risk_region.melt(id_vars="region", value_vars=["anapara_risk_m", "faiz_risk_m"],
                             var_name="tip", value_name="tutar"),
            x="region", y="tutar", color="tip", barmode="group",
            labels={"region": "Bolge", "tutar": "Milyon TL", "tip": "Risk Tipi"},
            color_discrete_map={"anapara_risk_m": "#EF553B", "faiz_risk_m": "#636EFA"},
        )
        fig.for_each_trace(lambda t: t.update(name="Anapara" if "anapara" in t.name else "Faiz"))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(risk_region, values="adet", names="region",
                     title="Risk Adet Dagilimi", hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(risk_region))
    for i, row in enumerate(risk_region.sort_values("region").itertuples()):
        with cols[i]:
            if st.button(f"{row.region}", key=f"risk_reg_{row.region}", use_container_width=True):
                go_to("kredi_risk_region", region=row.region)
                st.rerun()

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

    st.markdown("---")

    # 5. Gelecek Odemelere Gore Kalan Kredi Riski Projeksiyonu
    st.subheader("Gelecek Odemelere Gore Kalan Kredi Riski")

    AY_ISIMLERI = {1: "Ocak", 2: "Subat", 3: "Mart", 4: "Nisan", 5: "Mayis", 6: "Haziran",
                   7: "Temmuz", 8: "Agustos", 9: "Eylul", 10: "Ekim", 11: "Kasim", 12: "Aralik"}

    SABLON_MAP = {
        "720_gun_4_taksit": (720, 4), "360_gun_2_taksit": (360, 2),
        "5_yil_9_taksit": (1825, 9), "7_yil_9_taksit": (2555, 9),
        "360_gun_tek_taksit": (360, 1),
    }

    ref_tarih = datetime(2026, 3, 1)

    # Her kredi icin gelecek taksit tarihlerini ve odeme tutarlarini hesapla
    odeme_rows = []
    for _, row in df.iterrows():
        sablon = SABLON_MAP.get(row["odeme_sablonu"])
        if sablon is None or row["anapara_risk"] <= 0:
            continue
        gun, taksit_sayisi = sablon
        valor_dt = pd.to_datetime(row["valor"])
        taksit_araligi = gun / taksit_sayisi

        gelecek_tarihler = []
        for t in range(taksit_sayisi):
            taksit_dt = valor_dt + timedelta(days=int(taksit_araligi * (t + 1)))
            if taksit_dt > ref_tarih:
                gelecek_tarihler.append(taksit_dt)

        if not gelecek_tarihler:
            continue

        taksit_tutar = row["anapara_risk"] / len(gelecek_tarihler)
        for dt in gelecek_tarihler:
            odeme_rows.append({"tarih": dt, "odeme": taksit_tutar})

    if odeme_rows:
        odeme_df = pd.DataFrame(odeme_rows)
        odeme_df["tarih"] = pd.to_datetime(odeme_df["tarih"])

        # Ay ve ceyrek bilgisi ekle
        odeme_df["yil"] = odeme_df["tarih"].dt.year
        odeme_df["ay"] = odeme_df["tarih"].dt.month

        # Ilk 4 ay siniri
        ay4_sinir = ref_tarih + timedelta(days=120)

        # Aylik odemeler (ilk 4 ay)
        aylik = odeme_df[odeme_df["tarih"] < ay4_sinir].copy()
        aylik_grp = aylik.groupby(["yil", "ay"])["odeme"].sum().reset_index()
        aylik_grp["donem"] = aylik_grp.apply(
            lambda r: f"{AY_ISIMLERI[int(r['ay'])]} {int(r['yil'])}", axis=1)
        aylik_grp["sira"] = aylik_grp["yil"] * 100 + aylik_grp["ay"]

        # Ceyreklik odemeler (4 aydan sonrasi)
        ceyreklik = odeme_df[odeme_df["tarih"] >= ay4_sinir].copy()
        ceyreklik["ceyrek"] = ceyreklik["tarih"].dt.to_period("Q")
        ceyrek_grp = ceyreklik.groupby("ceyrek")["odeme"].sum().reset_index()
        ceyrek_grp["donem"] = ceyrek_grp["ceyrek"].astype(str)
        ceyrek_grp["sira"] = range(90000, 90000 + len(ceyrek_grp))

        # Birlestir
        tum_donemler = pd.concat([
            aylik_grp[["donem", "odeme", "sira"]],
            ceyrek_grp[["donem", "odeme", "sira"]],
        ]).sort_values("sira").reset_index(drop=True)

        # Kumulatif kalan risk hesapla
        toplam_risk = df["anapara_risk"].sum()
        tum_donemler["kumulatif_odeme"] = tum_donemler["odeme"].cumsum()
        tum_donemler["kalan_risk"] = toplam_risk - tum_donemler["kumulatif_odeme"]
        tum_donemler["kalan_risk_m"] = (tum_donemler["kalan_risk"] / 1e6).round(1)
        tum_donemler["odeme_m"] = (tum_donemler["odeme"] / 1e6).round(1)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tum_donemler["donem"], y=tum_donemler["kalan_risk_m"],
                mode="lines+markers+text", name="Kalan Risk",
                line=dict(color="#EF553B", width=3),
                text=tum_donemler["kalan_risk_m"], textposition="top center",
                fill="tozeroy", fillcolor="rgba(239,85,59,0.1)",
            ))
            fig.update_layout(
                height=500, yaxis_title="Kalan Risk (M TL)", xaxis_title="",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(tum_donemler, x="donem", y="odeme_m",
                         labels={"donem": "", "odeme_m": "Odeme (M TL)"},
                         color_discrete_sequence=["#00CC96"], text="odeme_m")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Tablo
        with st.expander("Detay Tablosu"):
            tablo = tum_donemler[["donem", "odeme_m", "kalan_risk_m"]].copy()
            tablo.columns = ["Donem", "Donem Odemesi (M TL)", "Kalan Risk (M TL)"]
            baslangic_col = [""] * len(tablo)
            baslangic_col[0] = str(round(toplam_risk / 1e6, 1))
            tablo.insert(0, "Baslangic Risk (M TL)", baslangic_col)
            st.dataframe(tablo, use_container_width=True, hide_index=True)
    else:
        st.info("Gelecek donemde odeme bulunamadi.")

    st.markdown("---")

    # 6. Kredi Teklif Onerileri (Next Best Action)
    st.subheader("Kredi Teklif Onerileri")
    st.caption("Bos limiti yuksek ve PD skoru %50'den dusuk (guvenilir) musterilere yeni urun onerisi")

    # genel_limit - anapara_risk = bos_limit
    cust_risk_agg = risk.groupby("mno")["anapara_risk"].sum().reset_index().rename(
        columns={"anapara_risk": "toplam_risk"})
    nba_df = cust_limit[["mno", "genel_limit"]].merge(cust_risk_agg, on="mno", how="left")
    nba_df["toplam_risk"] = nba_df["toplam_risk"].fillna(0)
    nba_df["bos_limit"] = nba_df["genel_limit"] - nba_df["toplam_risk"]
    nba_df = nba_df.merge(customer[["mno", "unvan", "kobi_flg", "sube", "portfoy", "region"]], on="mno", how="left")
    nba_df = nba_df.merge(pd_scores, on="mno", how="left").dropna(subset=["pd_skor"])

    # Sidebar filtrelerini uygula
    if st.session_state.get("kredi_portfoy"):
        nba_df = nba_df[nba_df["portfoy"].isin(st.session_state["kredi_portfoy"])]
    if st.session_state.get("kredi_sube"):
        nba_df = nba_df[nba_df["sube"].isin(st.session_state["kredi_sube"])]
    if st.session_state.get("kredi_bolge"):
        nba_df = nba_df[nba_df["region"].isin(st.session_state["kredi_bolge"])]
    if st.session_state.get("kredi_kobi"):
        nba_df = nba_df[nba_df["kobi_flg"].isin(st.session_state["kredi_kobi"])]

    # Musterinin kullandigi kredi programlari
    cust_prog = credit.groupby("mno")["krd_program"].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 else "").reset_index()
    cust_prog.columns = ["mno", "mevcut_program"]
    cust_prog_list = credit.groupby("mno")["krd_program"].apply(set).reset_index()
    cust_prog_list.columns = ["mno", "program_seti"]
    nba_df = nba_df.merge(cust_prog, on="mno", how="left")
    nba_df = nba_df.merge(cust_prog_list, on="mno", how="left")

    # Filtre: PD < 20 ve bos_limit > 0
    guvenilir = nba_df[(nba_df["pd_skor"] < 50) & (nba_df["bos_limit"] > 0)].copy()

    PROG_LABEL = {"isletme_kredi": "Isletme Kredisi", "yatirim_kredi": "Yatirim Kredisi",
                   "reeskont": "Reeskont"}

    # Son ay USD kuru (reeskont limiti icin)
    _son_ay = currency["ay"].max()
    _usd_kur = currency[(currency["pb1"] == "USD") & (currency["ay"] == _son_ay)]["rate"].values[0]
    reeskont_limit_tl = 1_000_000 * _usd_kur

    # --- TAB: Kredi Teklifi vs Reeskont Teklifi ---
    tab_kredi, tab_reeskont = st.tabs(["Kredi Teklifi (Isletme / Yatirim)", "Reeskont Teklifi"])

    # ========== TAB 1: KREDI TEKLIFI ==========
    with tab_kredi:
        # Cross-sell: isletme -> yatirim, yatirim -> isletme
        ONERI_MAP = {
            "isletme_kredi": "Yatirim Kredisi",
            "yatirim_kredi": "Isletme Kredisi",
            "reeskont": "Yatirim Kredisi",
        }
        kredi_teklif = guvenilir.copy()
        kredi_teklif["oneri"] = kredi_teklif["mevcut_program"].map(ONERI_MAP).fillna("Isletme Kredisi")

        k1, k2, k3 = st.columns(3)
        k1.metric("Toplam Musteri", f"{len(nba_df):,}")
        k2.metric("Guvenilir (PD<%50)", f"{len(guvenilir):,}")
        k3.metric("Kredi Teklif Hazir", f"{len(kredi_teklif):,}")

        # Scatter
        nba_plot = nba_df.copy()
        nba_plot["durum"] = ((nba_plot["pd_skor"] < 50) & (nba_plot["bos_limit"] > 0)).map(
            {True: "Teklif Hazir", False: "Uygun Degil"})
        fig = px.scatter(nba_plot, x="bos_limit", y="pd_skor", color="durum",
                         color_discrete_map={"Teklif Hazir": "#00CC96", "Uygun Degil": "#ABB2B9"},
                         hover_data=["mno", "unvan", "sube"],
                         labels={"bos_limit": "Bos Limit (TL)", "pd_skor": "PD Skoru (%)", "durum": ""},
                         opacity=0.7)
        fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="PD Esigi (%50)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        if len(kredi_teklif) > 0:
            kt = kredi_teklif.copy()
            kt["bos_limit_m"] = (kt["bos_limit"] / 1e6).round(2)
            kt["genel_limit_m"] = (kt["genel_limit"] / 1e6).round(2)
            kt["toplam_risk_m"] = (kt["toplam_risk"] / 1e6).round(2)
            kt["kobi_label"] = kt["kobi_flg"].map({0: "Kurumsal", 1: "KOBi"})
            kt["mevcut"] = kt["mevcut_program"].map(PROG_LABEL)
            kt = kt.sort_values("bos_limit", ascending=False)

            st.dataframe(
                kt[["mno", "unvan", "sube", "region", "kobi_label",
                    "genel_limit_m", "toplam_risk_m", "bos_limit_m",
                    "pd_skor", "mevcut", "oneri"]]
                .rename(columns={
                    "mno": "MNO", "unvan": "Unvan", "sube": "Sube", "region": "Bolge",
                    "kobi_label": "Segment", "genel_limit_m": "Genel Limit (M)",
                    "toplam_risk_m": "Risk (M)", "bos_limit_m": "Bos Limit (M)",
                    "pd_skor": "PD (%)", "mevcut": "Mevcut Program", "oneri": "Urun Onerisi"
                }),
                use_container_width=True, height=400
            )

            c1, c2 = st.columns(2)
            with c1:
                oneri_grp = kt.groupby("oneri").agg(
                    musteri=("mno", "count"), potansiyel=("bos_limit", "sum")).reset_index()
                oneri_grp["potansiyel_m"] = (oneri_grp["potansiyel"] / 1e6).round(1)
                fig = px.bar(oneri_grp, x="oneri", y="musteri",
                             labels={"oneri": "Oneri", "musteri": "Musteri Sayisi"},
                             color="oneri", text="musteri")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=350, showlegend=False, title="Urun Onerisine Gore Musteri")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(oneri_grp, x="oneri", y="potansiyel_m",
                             labels={"oneri": "Oneri", "potansiyel_m": "Potansiyel (M TL)"},
                             color="oneri", text="potansiyel_m")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=350, showlegend=False, title="Potansiyel Hacim")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Uygun kredi teklif musterisi bulunamadi.")

    # ========== TAB 2: REESKONT TEKLIFI ==========
    with tab_reeskont:
        st.caption(f"Guvenilir (PD<%50), bos limiti olan musteriler. "
                   f"Maks. 1M USD ({reeskont_limit_tl/1e6:,.2f} M TL)")

        # Tum guvenilir musteriler
        reeskont_teklif = guvenilir.copy()

        # Reeskont tutari: bos_limit ile 1M USD limitinin kucugu
        reeskont_teklif["reeskont_tutar"] = reeskont_teklif["bos_limit"].values.clip(max=reeskont_limit_tl)
        reeskont_teklif = reeskont_teklif[reeskont_teklif["reeskont_tutar"] > 0].sort_values(
            "reeskont_tutar", ascending=False)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Guvenilir Musteri", f"{len(guvenilir):,}")
        k2.metric("Teklif Hazir", f"{len(reeskont_teklif):,}")
        k3.metric("Maks Tutar (1M USD)", f"{reeskont_limit_tl/1e6:,.2f} M TL")
        k4.metric("Toplam Potansiyel", f"{reeskont_teklif['reeskont_tutar'].sum()/1e6:,.1f} M TL")

        if len(reeskont_teklif) > 0:
            rt = reeskont_teklif.copy()
            rt["reeskont_m"] = (rt["reeskont_tutar"] / 1e6).round(2)
            rt["reeskont_usd"] = (rt["reeskont_tutar"] / _usd_kur / 1000).round(1)
            rt["bos_limit_m"] = (rt["bos_limit"] / 1e6).round(2)
            rt["genel_limit_m"] = (rt["genel_limit"] / 1e6).round(2)
            rt["toplam_risk_m"] = (rt["toplam_risk"] / 1e6).round(2)
            rt["kobi_label"] = rt["kobi_flg"].map({0: "Kurumsal", 1: "KOBi"})
            rt["mevcut"] = rt["mevcut_program"].map(PROG_LABEL)

            st.dataframe(
                rt[["mno", "unvan", "sube", "region", "kobi_label",
                    "genel_limit_m", "toplam_risk_m", "bos_limit_m",
                    "pd_skor", "mevcut", "reeskont_m", "reeskont_usd"]]
                .rename(columns={
                    "mno": "MNO", "unvan": "Unvan", "sube": "Sube", "region": "Bolge",
                    "kobi_label": "Segment", "genel_limit_m": "Genel Limit (M)",
                    "toplam_risk_m": "Risk (M)", "bos_limit_m": "Bos Limit (M)",
                    "pd_skor": "PD (%)", "mevcut": "Mevcut Program",
                    "reeskont_m": "Reeskont Teklif (M TL)", "reeskont_usd": "Reeskont (Bin USD)"
                }),
                use_container_width=True, height=400
            )

            c1, c2 = st.columns(2)
            with c1:
                bolge_re = rt.groupby("region").agg(
                    musteri=("mno", "count"), potansiyel=("reeskont_tutar", "sum")).reset_index()
                bolge_re["potansiyel_m"] = (bolge_re["potansiyel"] / 1e6).round(1)
                fig = px.bar(bolge_re.sort_values("potansiyel_m", ascending=True),
                             x="potansiyel_m", y="region", orientation="h",
                             labels={"region": "Bolge", "potansiyel_m": "Potansiyel (M TL)"},
                             color="region", text="potansiyel_m",
                             title="Bolge Bazinda Reeskont Potansiyeli")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                seg_re = rt.groupby("kobi_label").agg(
                    musteri=("mno", "count"), potansiyel=("reeskont_tutar", "sum")).reset_index()
                seg_re["potansiyel_m"] = (seg_re["potansiyel"] / 1e6).round(1)
                fig = px.pie(seg_re, values="musteri", names="kobi_label",
                             title="Segment Bazinda Reeskont Dagilimi", hole=0.4,
                             color="kobi_label",
                             color_discrete_map={"KOBi": "#00CC96", "Kurumsal": "#AB63FA"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Uygun reeskont teklif musterisi bulunamadi.")


# ============================================================
# KREDI KULLANDIRIM DRILL-DOWN SAYFALARI
# ============================================================
def page_kredi_kull_region():
    region = st.session_state.drill_region
    st.title(f"Kredi Kullandirim - {region} Bolgesi")
    back_button(target="kredi", label="Kredi Sayfasina Don")

    df = credit_full[credit_full["region"] == region].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Kredi Adedi", f"{len(df):,}")
    col2.metric("Toplam Kullandirim", f"{df['lcy_tutar'].sum()/1e6:,.1f} M TL")
    col3.metric("USD Kullandirim", f"{df['usd_tutar'].sum()/1e6:,.1f} M USD")

    st.markdown("---")

    # Sube bazinda
    st.subheader("Sube Bazinda Kullandirim")
    st.caption("Subeye tiklayin detay icin")
    sube_grp = df.groupby("sube").agg(
        kullandirim=("lcy_tutar", "sum"), usd_kullandirim=("usd_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    sube_grp["kullandirim_m"] = (sube_grp["kullandirim"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sube_grp.sort_values("kullandirim_m", ascending=True),
                     x="kullandirim_m", y="sube", orientation="h",
                     labels={"sube": "Sube", "kullandirim_m": "Kullandirim (M TL)"},
                     color="sube", text="kullandirim_m")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(sube_grp, values="adet", names="sube",
                     title="Adet Dagilimi", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(sube_grp))
    for i, row in enumerate(sube_grp.sort_values("sube").itertuples()):
        with cols[i]:
            if st.button(f"{row.sube}", key=f"kull_sube_{row.sube}", use_container_width=True):
                go_to("kredi_kull_sube", sube=row.sube)
                st.rerun()

    st.markdown("---")

    # Programa gore
    st.subheader("Programa Gore Kullandirim")
    prog = df.groupby("krd_program").agg(
        kullandirim=("lcy_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    prog["kullandirim_m"] = (prog["kullandirim"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(prog, x="krd_program", y="kullandirim_m",
                     labels={"krd_program": "Program", "kullandirim_m": "Kullandirim (M TL)"},
                     color="krd_program", text="kullandirim_m")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(prog, values="adet", names="krd_program",
                     title="Program Adet Dagilimi", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Para birimi dagilimi
    st.subheader("Para Birimi Dagilimi")
    pb_grp = df.groupby("para_birimi").agg(
        kullandirim=("lcy_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    pb_grp["kullandirim_m"] = (pb_grp["kullandirim"] / 1e6).round(1)
    fig = px.bar(pb_grp, x="para_birimi", y="kullandirim_m", color="para_birimi",
                 labels={"para_birimi": "Para Birimi", "kullandirim_m": "Kullandirim (M TL)"},
                 text="kullandirim_m")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def page_kredi_kull_sube():
    sube = st.session_state.drill_sube
    region = st.session_state.drill_region
    st.title(f"Kredi Kullandirim - {sube} Subesi")
    back_button(target="kredi_kull_region", label=f"{region} Bolgesine Don", region=region)

    df = credit_full[credit_full["sube"] == sube].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Kredi Adedi", f"{len(df):,}")
    col2.metric("Toplam Kullandirim", f"{df['lcy_tutar'].sum()/1e6:,.1f} M TL")
    col3.metric("USD Kullandirim", f"{df['usd_tutar'].sum()/1e6:,.1f} M USD")

    st.markdown("---")

    # Portfoy bazinda
    st.subheader("Portfoy Bazinda Kullandirim")
    st.caption("Portfoye tiklayin detay icin")
    port_grp = df.groupby("portfoy").agg(
        kullandirim=("lcy_tutar", "sum"), usd_kullandirim=("usd_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    port_grp["kullandirim_m"] = (port_grp["kullandirim"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(port_grp.sort_values("kullandirim_m", ascending=True),
                     x="kullandirim_m", y="portfoy", orientation="h",
                     labels={"portfoy": "Portfoy Yoneticisi", "kullandirim_m": "Kullandirim (M TL)"},
                     color="portfoy", text="kullandirim_m")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=max(400, len(port_grp) * 40), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(port_grp, values="adet", names="portfoy",
                     title="Adet Dagilimi", hole=0.4)
        fig.update_layout(height=max(400, len(port_grp) * 40))
        st.plotly_chart(fig, use_container_width=True)

    # Portfoy butonlari (3'er sutun)
    port_list = sorted(port_grp["portfoy"].unique())
    for row_start in range(0, len(port_list), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < len(port_list):
                with col:
                    if st.button(port_list[idx], key=f"kull_port_{port_list[idx]}", use_container_width=True):
                        go_to("kredi_kull_portfoy", portfoy=port_list[idx])
                        st.rerun()

    st.markdown("---")

    # Programa gore
    st.subheader("Programa Gore Kullandirim")
    prog = df.groupby("krd_program").agg(
        kullandirim=("lcy_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    prog["kullandirim_m"] = (prog["kullandirim"] / 1e6).round(1)
    fig = px.bar(prog, x="krd_program", y="kullandirim_m", color="krd_program",
                 labels={"krd_program": "Program", "kullandirim_m": "Kullandirim (M TL)"},
                 text="kullandirim_m")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def page_kredi_kull_portfoy():
    portfoy = st.session_state.drill_portfoy
    sube = st.session_state.drill_sube
    region = st.session_state.drill_region
    st.title(f"Kredi Kullandirim - {portfoy}")
    back_button(target="kredi_kull_sube", label=f"{sube} Subesine Don", sube=sube)

    df = credit_full[(credit_full["portfoy"] == portfoy) & (credit_full["sube"] == sube)].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Kredi Adedi", f"{len(df):,}")
    col2.metric("Toplam Kullandirim", f"{df['lcy_tutar'].sum()/1e6:,.1f} M TL")
    col3.metric("USD Kullandirim", f"{df['usd_tutar'].sum()/1e6:,.1f} M USD")

    st.markdown("---")

    # Programa gore
    st.subheader("Programa Gore Kullandirim")
    prog = df.groupby("krd_program").agg(
        kullandirim=("lcy_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    prog["kullandirim_m"] = (prog["kullandirim"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(prog, x="krd_program", y="kullandirim_m", color="krd_program",
                     labels={"krd_program": "Program", "kullandirim_m": "Kullandirim (M TL)"},
                     text="kullandirim_m")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(prog, values="adet", names="krd_program",
                     title="Program Adet Dagilimi", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Ceyreklik kullandirim trendi
    st.subheader("Ceyreklik Kullandirim Trendi")
    ceyrek_grp = df.groupby("ceyrek").agg(
        kullandirim=("lcy_tutar", "sum"), adet=("krd_id", "count")
    ).reset_index()
    ceyrek_grp["kullandirim_m"] = (ceyrek_grp["kullandirim"] / 1e6).round(1)
    fig = px.bar(ceyrek_grp, x="ceyrek", y="kullandirim_m",
                 labels={"ceyrek": "Ceyrek", "kullandirim_m": "Kullandirim (M TL)"},
                 text="kullandirim_m", color_discrete_sequence=["#636EFA"])
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Kredi detay tablosu
    st.subheader("Kredi Detaylari")
    st.dataframe(
        df[["krd_id", "mno", "lcy_tutar", "usd_tutar", "para_birimi", "valor", "vade",
            "krd_program", "odeme_sablonu", "yapilandirma"]].sort_values("lcy_tutar", ascending=False),
        use_container_width=True, height=400
    )


# ============================================================
# KREDI RISK DRILL-DOWN SAYFALARI
# ============================================================
def page_kredi_risk_region():
    region = st.session_state.drill_region
    st.title(f"Kredi Risk - {region} Bolgesi")
    back_button(target="kredi", label="Kredi Sayfasina Don")

    df = risk_full[risk_full["region"] == region].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kredi Adedi", f"{len(df):,}")
    col2.metric("Anapara Risk", f"{df['anapara_risk'].sum()/1e6:,.1f} M")
    col3.metric("Faiz Risk", f"{df['faiz_risk'].sum()/1e6:,.1f} M")
    col4.metric("Yapilandirma", f"{df[df['yapilandirma']==1].shape[0]:,}")

    st.markdown("---")

    # Sube bazinda
    st.subheader("Sube Bazinda Risk")
    st.caption("Subeye tiklayin detay icin")
    sube_grp = df.groupby("sube").agg(
        anapara_risk=("anapara_risk", "sum"),
        faiz_risk=("faiz_risk", "sum"),
        adet=("krd_id", "count"),
    ).reset_index()
    sube_grp["anapara_risk_m"] = (sube_grp["anapara_risk"] / 1e6).round(1)
    sube_grp["faiz_risk_m"] = (sube_grp["faiz_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            sube_grp.melt(id_vars="sube", value_vars=["anapara_risk_m", "faiz_risk_m"],
                          var_name="tip", value_name="tutar"),
            x="sube", y="tutar", color="tip", barmode="group",
            labels={"sube": "Sube", "tutar": "Milyon TL", "tip": "Risk Tipi"},
            color_discrete_map={"anapara_risk_m": "#EF553B", "faiz_risk_m": "#636EFA"},
        )
        fig.for_each_trace(lambda t: t.update(name="Anapara" if "anapara" in t.name else "Faiz"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(sube_grp, values="adet", names="sube",
                     title="Risk Adet Dagilimi", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(sube_grp))
    for i, row in enumerate(sube_grp.sort_values("sube").itertuples()):
        with cols[i]:
            if st.button(f"{row.sube}", key=f"risk_sube_{row.sube}", use_container_width=True):
                go_to("kredi_risk_sube", sube=row.sube)
                st.rerun()

    st.markdown("---")

    # Programa gore risk
    st.subheader("Programa Gore Risk")
    prog = df.groupby("krd_program").agg(
        anapara_risk=("anapara_risk", "sum"), faiz_risk=("faiz_risk", "sum"), adet=("krd_id", "count")
    ).reset_index()
    prog["anapara_risk_m"] = (prog["anapara_risk"] / 1e6).round(1)
    prog["faiz_risk_m"] = (prog["faiz_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Anapara", x=prog["krd_program"], y=prog["anapara_risk_m"], marker_color="#EF553B"))
        fig.add_trace(go.Bar(name="Faiz", x=prog["krd_program"], y=prog["faiz_risk_m"], marker_color="#636EFA"))
        fig.update_layout(barmode="group", height=400, yaxis_title="Milyon TL")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(prog, values="adet", names="krd_program",
                     title="Program Adet Dagilimi", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def page_kredi_risk_sube():
    sube = st.session_state.drill_sube
    region = st.session_state.drill_region
    st.title(f"Kredi Risk - {sube} Subesi")
    back_button(target="kredi_risk_region", label=f"{region} Bolgesine Don", region=region)

    df = risk_full[risk_full["sube"] == sube].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kredi Adedi", f"{len(df):,}")
    col2.metric("Anapara Risk", f"{df['anapara_risk'].sum()/1e6:,.1f} M")
    col3.metric("Faiz Risk", f"{df['faiz_risk'].sum()/1e6:,.1f} M")
    col4.metric("Yapilandirma", f"{df[df['yapilandirma']==1].shape[0]:,}")

    st.markdown("---")

    # Portfoy bazinda
    st.subheader("Portfoy Bazinda Risk")
    st.caption("Portfoye tiklayin detay icin")
    port_grp = df.groupby("portfoy").agg(
        anapara_risk=("anapara_risk", "sum"),
        faiz_risk=("faiz_risk", "sum"),
        adet=("krd_id", "count"),
    ).reset_index()
    port_grp["anapara_risk_m"] = (port_grp["anapara_risk"] / 1e6).round(1)
    port_grp["faiz_risk_m"] = (port_grp["faiz_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(port_grp.sort_values("anapara_risk_m", ascending=True),
                     x="anapara_risk_m", y="portfoy", orientation="h",
                     labels={"portfoy": "Portfoy Yoneticisi", "anapara_risk_m": "Anapara Risk (M TL)"},
                     color="portfoy", text="anapara_risk_m")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=max(400, len(port_grp) * 40), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(port_grp, values="adet", names="portfoy",
                     title="Risk Adet Dagilimi", hole=0.4)
        fig.update_layout(height=max(400, len(port_grp) * 40))
        st.plotly_chart(fig, use_container_width=True)

    # Portfoy butonlari
    port_list = sorted(port_grp["portfoy"].unique())
    for row_start in range(0, len(port_list), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < len(port_list):
                with col:
                    if st.button(port_list[idx], key=f"risk_port_{port_list[idx]}", use_container_width=True):
                        go_to("kredi_risk_portfoy", portfoy=port_list[idx])
                        st.rerun()

    st.markdown("---")

    # Yapilandirma dagilimi
    st.subheader("Yapilandirma Durumu")
    yapil = df.copy()
    yapil["yapil_label"] = yapil["yapilandirma"].map({0: "Normal", 1: "Yapilandirma"})
    yapil_grp = yapil.groupby("yapil_label").agg(
        anapara_risk=("anapara_risk", "sum"), adet=("krd_id", "count")
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
        fig = px.pie(yapil_grp, values="adet", names="yapil_label",
                     title="Adet Dagilimi", hole=0.4,
                     color_discrete_map={"Normal": "#636EFA", "Yapilandirma": "#EF553B"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def page_kredi_risk_portfoy():
    portfoy = st.session_state.drill_portfoy
    sube = st.session_state.drill_sube
    region = st.session_state.drill_region
    st.title(f"Kredi Risk - {portfoy}")
    back_button(target="kredi_risk_sube", label=f"{sube} Subesine Don", sube=sube)

    df = risk_full[(risk_full["portfoy"] == portfoy) & (risk_full["sube"] == sube)].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kredi Adedi", f"{len(df):,}")
    col2.metric("Anapara Risk", f"{df['anapara_risk'].sum()/1e6:,.1f} M")
    col3.metric("Faiz Risk", f"{df['faiz_risk'].sum()/1e6:,.1f} M")
    col4.metric("Yapilandirma", f"{df[df['yapilandirma']==1].shape[0]:,}")

    st.markdown("---")

    # Programa gore risk
    st.subheader("Programa Gore Risk")
    prog = df.groupby("krd_program").agg(
        anapara_risk=("anapara_risk", "sum"), faiz_risk=("faiz_risk", "sum"), adet=("krd_id", "count")
    ).reset_index()
    prog["anapara_risk_m"] = (prog["anapara_risk"] / 1e6).round(1)
    prog["faiz_risk_m"] = (prog["faiz_risk"] / 1e6).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Anapara", x=prog["krd_program"], y=prog["anapara_risk_m"], marker_color="#EF553B"))
        fig.add_trace(go.Bar(name="Faiz", x=prog["krd_program"], y=prog["faiz_risk_m"], marker_color="#636EFA"))
        fig.update_layout(barmode="group", height=400, yaxis_title="Milyon TL")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(prog, values="adet", names="krd_program",
                     title="Program Adet Dagilimi", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Ceyreklik risk trendi
    st.subheader("Ceyreklik Risk Trendi")
    ceyrek_grp = df.groupby("ceyrek").agg(
        anapara_risk=("anapara_risk", "sum"), faiz_risk=("faiz_risk", "sum")
    ).reset_index()
    ceyrek_grp["anapara_risk_m"] = (ceyrek_grp["anapara_risk"] / 1e6).round(1)
    ceyrek_grp["faiz_risk_m"] = (ceyrek_grp["faiz_risk"] / 1e6).round(1)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Anapara", x=ceyrek_grp["ceyrek"], y=ceyrek_grp["anapara_risk_m"], marker_color="#EF553B"))
    fig.add_trace(go.Bar(name="Faiz", x=ceyrek_grp["ceyrek"], y=ceyrek_grp["faiz_risk_m"], marker_color="#636EFA"))
    fig.update_layout(barmode="stack", height=400, xaxis_tickangle=-45, yaxis_title="Milyon TL")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Kredi detay tablosu
    st.subheader("Kredi Risk Detaylari")
    st.dataframe(
        df[["krd_id", "mno", "anapara_risk", "faiz_risk", "pb", "valor", "vade",
            "krd_program", "odeme_sablonu", "yapilandirma"]].sort_values("anapara_risk", ascending=False),
        use_container_width=True, height=400
    )


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

    st.markdown("---")

    # 3. Stress Testing - Kredi Riski Bazli
    st.subheader("Stress Testing - Kur Senaryosu (Kredi Riski Bazli)")
    st.caption("Musterilerin nakit akislari TL cinsindendir. Yabanci para kredilerde kur artisi, "
               "TL cinsinden kredi riskini arttirir. TL krediler kurdan etkilenmez.")

    # Son ay kurlari
    usd_kur = son_kurlar[son_kurlar["pb1"] == "USD"]["rate"].values[0]
    eur_kur = son_kurlar[son_kurlar["pb1"] == "EUR"]["rate"].values[0]
    cny_kur = son_kurlar[son_kurlar["pb1"] == "CNY"]["rate"].values[0]

    kur_artis = st.slider("Kur Artis Orani (%)", min_value=0, max_value=100, value=20, step=5,
                           key="stress_slider")
    kur_carpan = 1 + kur_artis / 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("USD/TRY", f"{usd_kur:,.2f} -> {usd_kur*kur_carpan:,.2f}")
    k2.metric("EUR/TRY", f"{eur_kur:,.2f} -> {eur_kur*kur_carpan:,.2f}")
    k3.metric("CNY/TRY", f"{cny_kur:,.2f} -> {cny_kur*kur_carpan:,.2f}")
    k4.metric("Carpan", f"x{kur_carpan:.2f}", delta=f"+%{kur_artis}")

    # Kredi bazinda stress hesapla
    # risk tablosundaki her krediyi para birimiyle eslestir
    stress_kr = risk[["krd_id", "mno", "anapara_risk", "pb"]].copy()

    # Kur senaryosu: yabanci para kredilerin TL riski kur oraniyla artar, TL krediler sabit kalir
    stress_kr["stress_risk"] = stress_kr.apply(
        lambda r: r["anapara_risk"] * kur_carpan if r["pb"] != "TRY" else r["anapara_risk"], axis=1)

    # Musteri bazinda topla
    cust_stress = stress_kr.groupby("mno").agg(
        mevcut_risk=("anapara_risk", "sum"),
        senaryo_risk=("stress_risk", "sum"),
        yp_kredi_adet=("pb", lambda x: (x != "TRY").sum()),
        toplam_kredi=("krd_id", "count"),
    ).reset_index()

    # Teminat eslestir
    cust_teminat = collateral.groupby("cust_id")["tutar"].sum().reset_index().rename(
        columns={"cust_id": "mno", "tutar": "teminat_tutari"})
    cust_stress = cust_stress.merge(cust_teminat, on="mno", how="left").fillna({"teminat_tutari": 0})
    cust_stress = cust_stress.merge(customer[["mno", "unvan", "sube", "region"]], on="mno", how="left")

    cust_stress["risk_artis"] = cust_stress["senaryo_risk"] - cust_stress["mevcut_risk"]
    cust_stress["teminat_acigi"] = cust_stress["senaryo_risk"] > cust_stress["teminat_tutari"]
    cust_stress["acik_tutar"] = (cust_stress["senaryo_risk"] - cust_stress["teminat_tutari"]).clip(lower=0)

    # Mevcut durumda da acikta olanlar vs senaryoda yeni aciga dusenler
    cust_stress["mevcut_acik"] = cust_stress["mevcut_risk"] > cust_stress["teminat_tutari"]

    acikli = cust_stress[cust_stress["teminat_acigi"]]
    yeni_acikli = cust_stress[cust_stress["teminat_acigi"] & ~cust_stress["mevcut_acik"]]
    toplam_musteri = len(cust_stress)
    acikli_sayi = len(acikli)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Musteri", f"{toplam_musteri:,}")
    k2.metric("Teminat Acigi (Senaryo)", f"{acikli_sayi:,}")
    k3.metric("Yeni Aciga Dusen", f"{len(yeni_acikli):,}",
              delta=f"+{len(yeni_acikli)}" if len(yeni_acikli) > 0 else "0")
    k4.metric("Risk Artisi", f"{cust_stress['risk_artis'].sum()/1e6:,.1f} M TL")

    # Kur artisina gore etkilenen musteri sayisi grafigi
    st.markdown("")
    artis_araligi = list(range(0, 105, 5))
    sayilar = []
    for a in artis_araligi:
        c = 1 + a / 100
        senaryo = stress_kr.apply(
            lambda r: r["anapara_risk"] * c if r["pb"] != "TRY" else r["anapara_risk"], axis=1)
        senaryo_cust = pd.DataFrame({"mno": stress_kr["mno"], "sr": senaryo}).groupby("mno")["sr"].sum()
        tem = cust_teminat.set_index("mno")["teminat_tutari"]
        karsilastir = senaryo_cust.to_frame("sr").join(tem, how="left").fillna(0)
        cnt = int((karsilastir["sr"] > karsilastir["teminat_tutari"]).sum())
        sayilar.append({"kur_artis": a, "musteri": cnt})
    artis_df = pd.DataFrame(sayilar)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=artis_df["kur_artis"], y=artis_df["musteri"],
        mode="lines+markers+text", text=artis_df["musteri"],
        textposition="top center",
        line=dict(color="#EF553B", width=3),
        fill="tozeroy", fillcolor="rgba(239,85,59,0.1)",
    ))
    fig.add_vline(x=kur_artis, line_dash="dash", line_color="blue",
                  annotation_text=f"Secili: %{kur_artis}")
    fig.update_layout(height=450, xaxis_title="Kur Artis Orani (%)",
                      yaxis_title="Teminat Acigi Olan Musteri Sayisi",
                      title="Kur Artisina Gore Teminat Acigi")
    st.plotly_chart(fig, use_container_width=True)

    # Mevcut vs Senaryo risk karsilastirmasi
    c1, c2 = st.columns(2)
    with c1:
        bolge_stress = cust_stress.groupby("region").agg(
            mevcut=("mevcut_risk", "sum"), senaryo=("senaryo_risk", "sum")
        ).reset_index()
        bolge_stress["mevcut_m"] = (bolge_stress["mevcut"] / 1e6).round(1)
        bolge_stress["senaryo_m"] = (bolge_stress["senaryo"] / 1e6).round(1)
        bolge_melt = bolge_stress.melt(id_vars="region", value_vars=["mevcut_m", "senaryo_m"],
                                        var_name="tip", value_name="tutar")
        bolge_melt["tip"] = bolge_melt["tip"].map({"mevcut_m": "Mevcut Risk", "senaryo_m": "Senaryo Risk"})
        fig = px.bar(bolge_melt, x="region", y="tutar", color="tip", barmode="group",
                     labels={"region": "Bolge", "tutar": "Risk (M TL)", "tip": ""},
                     color_discrete_map={"Mevcut Risk": "#636EFA", "Senaryo Risk": "#EF553B"},
                     title="Bolge Bazinda Mevcut vs Senaryo Risk")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if len(acikli) > 0:
            bolge_acik = acikli.groupby("region").agg(
                musteri_say=("mno", "count"), acik_toplam=("acik_tutar", "sum")
            ).reset_index()
            bolge_acik["acik_m"] = (bolge_acik["acik_toplam"] / 1e6).round(1)
            fig = px.bar(bolge_acik.sort_values("acik_m", ascending=True),
                         x="acik_m", y="region", orientation="h",
                         labels={"region": "Bolge", "acik_m": "Acik Tutar (M TL)"},
                         color="region", text="acik_m",
                         title="Bolge Bazinda Teminat Acigi")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Detay tablosu
    with st.expander("Teminat Acigi Detay Tablosu"):
        tablo = cust_stress.copy()
        tablo["mevcut_m"] = (tablo["mevcut_risk"] / 1e6).round(2)
        tablo["senaryo_m"] = (tablo["senaryo_risk"] / 1e6).round(2)
        tablo["artis_m"] = (tablo["risk_artis"] / 1e6).round(2)
        tablo["teminat_m"] = (tablo["teminat_tutari"] / 1e6).round(2)
        tablo["acik_m"] = (tablo["acik_tutar"] / 1e6).round(2)
        tablo["durum"] = tablo["teminat_acigi"].map({True: "TEMINAT ACIGI", False: "Yeterli"})
        tablo["yp_oran"] = ((tablo["yp_kredi_adet"] / tablo["toplam_kredi"]) * 100).round(1)
        tablo = tablo.sort_values("acik_tutar", ascending=False)

        def highlight_acik(row):
            if row["Durum"] == "TEMINAT ACIGI":
                return ["background-color: #ffcccc; color: #cc0000; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            tablo[["mno", "unvan", "sube", "region", "mevcut_m", "senaryo_m",
                   "artis_m", "teminat_m", "acik_m", "yp_oran", "durum"]]
            .rename(columns={
                "mno": "MNO", "unvan": "Unvan", "sube": "Sube", "region": "Bolge",
                "mevcut_m": "Mevcut Risk (M)", "senaryo_m": "Senaryo Risk (M)",
                "artis_m": "Risk Artisi (M)", "teminat_m": "Teminat (M)",
                "acik_m": "Acik (M)", "yp_oran": "YP Oran (%)", "durum": "Durum"
            })
            .style.apply(highlight_acik, axis=1),
            use_container_width=True, height=500
        )


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
# MUSTERI SAYFASI
# ============================================================
def page_musteri():
    st.title("Musteri Analizi ve Siniflandirma")
    back_button()

    # --- Temel musteri verisini zenginlestir ---
    cust = customer.copy()

    # Kredi hacmi ve kullandirim sikligi
    cust_kredi = credit.groupby("mno").agg(
        kredi_adet=("krd_id", "count"),
        toplam_hacim=("lcy_tutar", "sum"),
        toplam_usd=("usd_tutar", "sum"),
    ).reset_index()

    # Odeme performansi: odenen taksit / toplam taksit
    kredi_taksit = credit[["krd_id", "mno", "odeme_sablonu"]].copy()
    SABLON_TAKSIT = {
        "720_gun_4_taksit": 4, "360_gun_2_taksit": 2,
        "5_yil_9_taksit": 9, "7_yil_9_taksit": 9,
        "360_gun_tek_taksit": 1,
    }
    kredi_taksit["toplam_taksit"] = kredi_taksit["odeme_sablonu"].map(SABLON_TAKSIT)
    odenen_taksit = payment.groupby("krd_id")["taksit_tutar"].count().reset_index()
    odenen_taksit.columns = ["krd_id", "odenen_taksit"]
    kredi_taksit = kredi_taksit.merge(odenen_taksit, on="krd_id", how="left")
    kredi_taksit["odenen_taksit"] = kredi_taksit["odenen_taksit"].fillna(0)
    cust_perf = kredi_taksit.groupby("mno").agg(
        toplam_taksit=("toplam_taksit", "sum"),
        odenen_taksit=("odenen_taksit", "sum"),
    ).reset_index()
    cust_perf["odeme_orani"] = (cust_perf["odenen_taksit"] / cust_perf["toplam_taksit"].replace(0, np.nan) * 100).round(1)

    # Risk
    cust_risk = risk.groupby("mno")["anapara_risk"].sum().reset_index().rename(
        columns={"anapara_risk": "toplam_risk"})

    # Birlestir
    cust = cust.merge(cust_kredi, on="mno", how="left")
    cust = cust.merge(cust_perf[["mno", "odeme_orani"]], on="mno", how="left")
    cust = cust.merge(cust_risk, on="mno", how="left")
    cust = cust.merge(pd_scores, on="mno", how="left")
    cust["kredi_adet"] = cust["kredi_adet"].fillna(0).astype(int)
    cust["toplam_hacim"] = cust["toplam_hacim"].fillna(0)
    cust["toplam_risk"] = cust["toplam_risk"].fillna(0)
    cust["odeme_orani"] = cust["odeme_orani"].fillna(0)
    cust["pd_skor"] = cust["pd_skor"].fillna(50)

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Musteri", f"{len(cust):,}")
    col2.metric("KOBi", f"{cust['kobi_flg'].sum():,}")
    col3.metric("Kurumsal", f"{(cust['kobi_flg'] == 0).sum():,}")
    col4.metric("Ort. PD Skoru", f"%{cust['pd_skor'].mean():.1f}")

    st.markdown("---")

    # ==========================================
    # 1. TEMEL SINIFLANDIRMA
    # ==========================================
    st.subheader("Temel Musteri Siniflandirmasi")

    # Kobi / Kurumsal
    cust["segment_tip"] = cust["kobi_flg"].map({1: "KOBi", 0: "Kurumsal"})

    # Bolge
    c1, c2, c3 = st.columns(3)
    with c1:
        seg_tip = cust["segment_tip"].value_counts().reset_index()
        seg_tip.columns = ["Segment", "Adet"]
        fig = px.pie(seg_tip, values="Adet", names="Segment", title="KOBi / Kurumsal",
                     hole=0.4, color="Segment",
                     color_discrete_map={"KOBi": "#00CC96", "Kurumsal": "#AB63FA"})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bolge_grp = cust.groupby("region")["mno"].count().reset_index()
        bolge_grp.columns = ["Bolge", "Adet"]
        fig = px.pie(bolge_grp, values="Adet", names="Bolge", title="Bolge Dagilimi", hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        cust["firma_yasi"] = 2026 - cust["kurulus_year"]
        yas_bins = pd.cut(cust["firma_yasi"], bins=[0, 5, 15, 30, 100],
                          labels=["0-5 Yil", "6-15 Yil", "16-30 Yil", "30+ Yil"])
        yas_grp = yas_bins.value_counts().reset_index()
        yas_grp.columns = ["Grup", "Adet"]
        fig = px.pie(yas_grp, values="Adet", names="Grup", title="Firma Yasi Dagilimi", hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 2. KREDI HACMI SINIFLANDIRMASI
    # ==========================================
    st.subheader("Kredi Hacmine Gore Siniflandirma")

    hacim_q = cust[cust["toplam_hacim"] > 0]["toplam_hacim"].quantile([0.33, 0.66])
    def hacim_sinif(val):
        if val <= 0:
            return "Kredisiz"
        elif val <= hacim_q.iloc[0]:
            return "Dusuk Hacim"
        elif val <= hacim_q.iloc[1]:
            return "Orta Hacim"
        else:
            return "Yuksek Hacim"
    cust["hacim_sinif"] = cust["toplam_hacim"].apply(hacim_sinif)

    c1, c2 = st.columns(2)
    with c1:
        hs = cust["hacim_sinif"].value_counts().reset_index()
        hs.columns = ["Sinif", "Adet"]
        fig = px.bar(hs, x="Sinif", y="Adet", color="Sinif", text="Adet",
                     color_discrete_map={"Kredisiz": "#ABB2B9", "Dusuk Hacim": "#636EFA",
                                         "Orta Hacim": "#FFA15A", "Yuksek Hacim": "#00CC96"},
                     title="Hacim Sinifi Dagilimi")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hacim_bolge = cust[cust["toplam_hacim"] > 0].groupby(["region", "hacim_sinif"])["mno"].count().reset_index()
        hacim_bolge.columns = ["Bolge", "Sinif", "Adet"]
        fig = px.bar(hacim_bolge, x="Bolge", y="Adet", color="Sinif", barmode="stack",
                     color_discrete_map={"Dusuk Hacim": "#636EFA", "Orta Hacim": "#FFA15A",
                                         "Yuksek Hacim": "#00CC96"},
                     title="Bolge Bazinda Hacim Sinifi")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 3. KULLANDIRIM SIKLIGI SINIFLANDIRMASI
    # ==========================================
    st.subheader("Kredi Kullandirim Sikligina Gore Siniflandirma")

    def siklik_sinif(adet):
        if adet == 0:
            return "Kredisiz"
        elif adet <= 10:
            return "Dusuk Siklik"
        elif adet <= 30:
            return "Orta Siklik"
        else:
            return "Yuksek Siklik"
    cust["siklik_sinif"] = cust["kredi_adet"].apply(siklik_sinif)

    c1, c2 = st.columns(2)
    with c1:
        ss = cust["siklik_sinif"].value_counts().reset_index()
        ss.columns = ["Sinif", "Adet"]
        fig = px.pie(ss, values="Adet", names="Sinif", title="Siklik Sinifi Dagilimi",
                     hole=0.4, color="Sinif",
                     color_discrete_map={"Kredisiz": "#ABB2B9", "Dusuk Siklik": "#636EFA",
                                         "Orta Siklik": "#FFA15A", "Yuksek Siklik": "#00CC96"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(cust[cust["kredi_adet"] > 0], x="kredi_adet", nbins=20,
                           labels={"kredi_adet": "Kredi Adedi", "count": "Musteri Sayisi"},
                           color_discrete_sequence=["#636EFA"], title="Kredi Adedi Dagilimi")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 4. ODEME PERFORMANSI SINIFLANDIRMASI
    # ==========================================
    st.subheader("Odeme Performansina Gore Siniflandirma")

    def perf_sinif(oran):
        if oran >= 80:
            return "Cok Iyi"
        elif oran >= 50:
            return "Iyi"
        elif oran >= 20:
            return "Orta"
        else:
            return "Zayif"
    cust["perf_sinif"] = cust["odeme_orani"].apply(perf_sinif)

    c1, c2 = st.columns(2)
    with c1:
        ps = cust["perf_sinif"].value_counts().reset_index()
        ps.columns = ["Sinif", "Adet"]
        fig = px.bar(ps, x="Sinif", y="Adet", color="Sinif", text="Adet",
                     color_discrete_map={"Cok Iyi": "#00CC96", "Iyi": "#636EFA",
                                         "Orta": "#FFA15A", "Zayif": "#EF553B"},
                     title="Odeme Performansi Dagilimi")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(cust[cust["odeme_orani"] > 0], x="odeme_orani", nbins=20,
                           labels={"odeme_orani": "Odeme Orani (%)", "count": "Musteri Sayisi"},
                           color_discrete_sequence=["#00CC96"], title="Odeme Orani Dagilimi")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 5. PD - TEMERR\u00dcT RISKI SINIFLANDIRMASI
    # ==========================================
    st.subheader("Tahmini Temerr\u00fct Ihtimali (PD Skoru)")
    st.caption("Lojistik Regresyon modeli: kobi_flg, kurulus_year, region, lcy_tutar feature'lari ile "
               "0-100 arasi risk skoru. %70 uzeri kirmizi ile isaretlenir.")

    def pd_sinif(skor):
        if skor < 20:
            return "Dusuk Risk"
        elif skor < 50:
            return "Orta Risk"
        elif skor < 70:
            return "Yuksek Risk"
        else:
            return "Kritik Risk"
    cust["pd_sinif"] = cust["pd_skor"].apply(pd_sinif)

    # PD KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dusuk Risk (<20)", f"{(cust['pd_skor'] < 20).sum():,}")
    k2.metric("Orta Risk (20-50)", f"{((cust['pd_skor'] >= 20) & (cust['pd_skor'] < 50)).sum():,}")
    k3.metric("Yuksek Risk (50-70)", f"{((cust['pd_skor'] >= 50) & (cust['pd_skor'] < 70)).sum():,}")
    k4.metric("Kritik Risk (>=70)", f"{(cust['pd_skor'] >= 70).sum():,}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(cust, x="pd_skor", nbins=25,
                           labels={"pd_skor": "PD Skoru (%)", "count": "Musteri Sayisi"},
                           color_discrete_sequence=["#636EFA"], title="PD Skor Dagilimi")
        fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Tehlike Esigi (%70)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pd_grp = cust["pd_sinif"].value_counts().reset_index()
        pd_grp.columns = ["Grup", "Adet"]
        fig = px.pie(pd_grp, values="Adet", names="Grup", title="PD Risk Grubu Dagilimi",
                     hole=0.4, color="Grup",
                     color_discrete_map={"Dusuk Risk": "#00CC96", "Orta Risk": "#636EFA",
                                         "Yuksek Risk": "#FFA15A", "Kritik Risk": "#EF553B"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================================
    # 6. DETAY TABLOSU
    # ==========================================
    st.subheader("Musteri Detay Tablosu")

    tablo = cust[["mno", "unvan", "segment_tip", "sube", "region", "kurulus_year",
                   "kredi_adet", "toplam_hacim", "odeme_orani", "toplam_risk", "pd_skor",
                   "hacim_sinif", "siklik_sinif", "perf_sinif", "pd_sinif"]].copy()
    tablo["toplam_hacim_m"] = (tablo["toplam_hacim"] / 1e6).round(2)
    tablo["toplam_risk_m"] = (tablo["toplam_risk"] / 1e6).round(2)
    tablo = tablo.sort_values("pd_skor", ascending=False)

    def highlight_pd(row):
        if row["PD (%)"] >= 70:
            return ["background-color: #ffcccc; color: #cc0000; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        tablo[["mno", "unvan", "segment_tip", "sube", "region", "kredi_adet",
               "toplam_hacim_m", "odeme_orani", "toplam_risk_m", "pd_skor",
               "hacim_sinif", "siklik_sinif", "perf_sinif", "pd_sinif"]]
        .rename(columns={
            "mno": "MNO", "unvan": "Unvan", "segment_tip": "Tip", "sube": "Sube",
            "region": "Bolge", "kredi_adet": "Kredi Adet",
            "toplam_hacim_m": "Hacim (M TL)", "odeme_orani": "Odeme (%)",
            "toplam_risk_m": "Risk (M TL)", "pd_skor": "PD (%)",
            "hacim_sinif": "Hacim Sinifi", "siklik_sinif": "Siklik Sinifi",
            "perf_sinif": "Performans", "pd_sinif": "PD Sinifi"
        })
        .style.apply(highlight_pd, axis=1),
        use_container_width=True, height=600
    )


# ============================================================
# ROUTING
# ============================================================
pages = {
    "ana_sayfa": page_ana_sayfa,
    "kredi": page_kredi,
    "kredi_kull_region": page_kredi_kull_region,
    "kredi_kull_sube": page_kredi_kull_sube,
    "kredi_kull_portfoy": page_kredi_kull_portfoy,
    "kredi_risk_region": page_kredi_risk_region,
    "kredi_risk_sube": page_kredi_risk_sube,
    "kredi_risk_portfoy": page_kredi_risk_portfoy,
    "teminat": page_teminat,
    "hazine": page_hazine,
    "limit": page_limit,
    "musteri": page_musteri,
}

pages[st.session_state.page]()
