import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SABITLER
# ============================================================
SUBELER = ["Istanbul", "Bursa", "Ankara", "Izmir", "Konya", "Erzurum", "Trabzon", "Manisa"]
REGION_MAP = {
    "Istanbul": "Marmara", "Bursa": "Marmara",
    "Izmir": "Ege", "Manisa": "Ege",
    "Ankara": "Anadolu", "Konya": "Anadolu", "Erzurum": "Anadolu", "Trabzon": "Anadolu",
}

PORTFOY_ISIMLERI = [
    "Ahmet Yilmaz", "Mehmet Kaya", "Ayse Demir", "Fatma Celik", "Ali Sahin",
    "Hasan Yildiz", "Huseyin Ozturk", "Zeynep Arslan", "Elif Dogan", "Murat Kilic",
    "Emre Aksoy", "Burak Korkmaz", "Selin Ozkan", "Derya Acar", "Canan Polat",
    "Oguz Tan",
]

SUBE_PORTFOY_MAP = {
    "Istanbul":  ["Ahmet Yilmaz", "Mehmet Kaya"],
    "Bursa":     ["Ayse Demir", "Fatma Celik"],
    "Ankara":    ["Ali Sahin", "Hasan Yildiz"],
    "Izmir":     ["Huseyin Ozturk", "Zeynep Arslan"],
    "Konya":     ["Elif Dogan", "Murat Kilic"],
    "Erzurum":   ["Emre Aksoy", "Burak Korkmaz"],
    "Trabzon":   ["Selin Ozkan", "Derya Acar"],
    "Manisa":    ["Canan Polat", "Oguz Tan"],
}

PARA_BIRIMLERI = ["USD", "EUR", "TRY", "CNY"]

ODEME_SABLONLARI = [
    {"label": "720_gun_4_taksit",   "gun": 720,  "taksit": 4},
    {"label": "360_gun_2_taksit",   "gun": 360,  "taksit": 2},
    {"label": "5_yil_9_taksit",     "gun": 1825, "taksit": 9},
    {"label": "7_yil_9_taksit",     "gun": 2555, "taksit": 9},
    {"label": "360_gun_tek_taksit", "gun": 360,  "taksit": 1},
]

PROGRAMLAR = ["reeskont", "isletme_kredi", "yatirim_kredi"]

TEMINAT_TURLERI = {
    100: "Banka Teminat Mektubu",
    200: "Ipotek",
    300: "Kefalet",
}
TEMINAT_IDS = list(TEMINAT_TURLERI.keys())

NUM_CUSTOMERS = 200
NUM_CREDITS = 5000
NUM_COLLATERALS = 3000

VALOR_START = datetime(2023, 12, 15)
VALOR_END = datetime(2026, 3, 1)

# ============================================================
# 1. dm.dim_customer
# ============================================================
mnos = sorted(random.sample(range(4200, 42001), NUM_CUSTOMERS))
unvanlar = list({fake.company() for _ in range(NUM_CUSTOMERS * 3)})[:NUM_CUSTOMERS]

_subeler_list = [random.choice(SUBELER) for _ in range(NUM_CUSTOMERS)]
_portfoy_list = [random.choice(SUBE_PORTFOY_MAP[s]) for s in _subeler_list]

dim_customer = pd.DataFrame({
    "mno": mnos,
    "unvan": unvanlar[:NUM_CUSTOMERS],
    "kobi_flg": [random.choice([0, 1]) for _ in range(NUM_CUSTOMERS)],
    "kurulus_year": [random.randint(1980, 2020) for _ in range(NUM_CUSTOMERS)],
    "sube": _subeler_list,
    "portfoy": _portfoy_list,
})
dim_customer["region"] = dim_customer["sube"].map(REGION_MAP)

print(f"dim_customer: {len(dim_customer)} satir")

# ============================================================
# 2. dm.dim_currency  (aylik kurlar, 2023-12 ~ 2026-03)
# ============================================================
base_rates = {"USD": 29.0, "EUR": 31.5, "TRY": 1.0, "CNY": 4.0}
months = pd.date_range("2023-12-01", "2026-03-01", freq="MS")

currency_rows = []
for m in months:
    month_offset = (m.year - 2023) * 12 + m.month - 12
    for pb in PARA_BIRIMLERI:
        drift = 1 + month_offset * random.uniform(0.005, 0.015)
        noise = random.uniform(0.97, 1.03)
        rate = round(base_rates[pb] * drift * noise, 4) if pb != "TRY" else 1.0
        currency_rows.append({"pb1": pb, "pb2": "TRY", "rate": rate, "ay": m.strftime("%Y-%m")})

dim_currency = pd.DataFrame(currency_rows)
print(f"dim_currency: {len(dim_currency)} satir")

# ============================================================
# 3. dm.dim_credit  (+ dw.credit_risk, dw.commitment)
# ============================================================
def random_valor():
    delta = (VALOR_END - VALOR_START).days
    return VALOR_START + timedelta(days=random.randint(0, delta))

def pick_program_and_sablon():
    prog = random.choice(PROGRAMLAR)
    if prog == "reeskont":
        sablon = ODEME_SABLONLARI[4]  # 360 gun tek taksit
    elif prog == "isletme_kredi":
        sablon = random.choice([s for s in ODEME_SABLONLARI if s["gun"] <= 720])
    else:  # yatirim_kredi
        sablon = random.choice([s for s in ODEME_SABLONLARI if s["gun"] >= 720])
    return prog, sablon

krd_ids = sorted(random.sample(range(120000, 550000), NUM_CREDITS))
credit_mnos = [random.choice(mnos) for _ in range(NUM_CREDITS)]

credit_rows = []
risk_rows = []
commitment_rows = []
payment_rows = []

# ref hesaplamasi icin: musteri bazinda valor sirasina gore
temp_credits = []
for i in range(NUM_CREDITS):
    valor = random_valor()
    prog, sablon = pick_program_and_sablon()
    pb = random.choice(PARA_BIRIMLERI)
    lcy_tutar = round(random.uniform(10000, 5000000), 2)

    # USD tutarini hesapla
    month_key = valor.strftime("%Y-%m")
    rate_row = dim_currency[(dim_currency["pb1"] == pb) & (dim_currency["ay"] == month_key)]
    if len(rate_row) > 0 and pb != "TRY":
        usd_rate = dim_currency[(dim_currency["pb1"] == "USD") & (dim_currency["ay"] == month_key)]["rate"].values[0]
        pb_rate = rate_row["rate"].values[0]
        usd_tutar = round(lcy_tutar * pb_rate / usd_rate, 2)
    elif pb == "TRY":
        usd_rate = dim_currency[(dim_currency["pb1"] == "USD") & (dim_currency["ay"] == month_key)]["rate"].values[0]
        usd_tutar = round(lcy_tutar / usd_rate, 2)
    else:
        usd_tutar = lcy_tutar

    vade = valor + timedelta(days=sablon["gun"])
    teminat_tur_id = random.choice(TEMINAT_IDS)
    teminat_tutar = round(lcy_tutar * random.uniform(0.5, 1.5), 2)
    faiz_oran = round(random.uniform(0.0, 1.0), 4)
    yapilandirma = 0  # asagida 200 tanesi rastgele secilecek

    temp_credits.append({
        "krd_id": krd_ids[i], "mno": credit_mnos[i], "valor": valor,
        "prog": prog, "sablon": sablon, "pb": pb,
        "lcy_tutar": lcy_tutar, "usd_tutar": usd_tutar, "vade": vade,
        "teminat_tur_id": teminat_tur_id, "teminat": teminat_tutar,
        "faiz_oran": faiz_oran, "yapilandirma": yapilandirma,
    })

# yapilandirma: rastgele 200 krediyi yapilandirmali yap
yapilandirma_indices = random.sample(range(NUM_CREDITS), 200)
for idx in yapilandirma_indices:
    temp_credits[idx]["yapilandirma"] = 1

# REF: musteri bazinda valor sirasina gore sequence
temp_credits.sort(key=lambda x: (x["mno"], x["valor"]))
ref_counter = {}
for c in temp_credits:
    m = c["mno"]
    ref_counter[m] = ref_counter.get(m, 0) + 1
    c["krd_ref"] = ref_counter[m]

for c in temp_credits:
    credit_rows.append({
        "krd_id": c["krd_id"], "mno": c["mno"], "krd_ref": c["krd_ref"],
        "lcy_tutar": c["lcy_tutar"], "usd_tutar": c["usd_tutar"],
        "para_birimi": c["pb"], "valor": c["valor"].strftime("%Y-%m-%d"),
        "vade": c["vade"].strftime("%Y-%m-%d"),
        "odeme_sablonu": c["sablon"]["label"], "krd_program": c["prog"],
        "yapilandirma": c["yapilandirma"],
        "teminat_tur_id": c["teminat_tur_id"],
        "teminat": TEMINAT_TURLERI[c["teminat_tur_id"]],
        "teminat_tutar": c["teminat"],
        "faiz_oran": c["faiz_oran"],
    })

    # credit_risk: bugune kadar odenmis taksitlere gore kalan risk
    taksit_sayisi = c["sablon"]["taksit"]
    gun = c["sablon"]["gun"]
    taksit_araligi = gun / taksit_sayisi
    taksit_tarihleri = [c["valor"] + timedelta(days=int(taksit_araligi * (t + 1))) for t in range(taksit_sayisi)]
    bugun = VALOR_END  # referans tarih

    taksit_tutar = round(c["lcy_tutar"] / taksit_sayisi, 2)

    if c["yapilandirma"] == 0:
        # odenmis taksitleri payment tablosuna yaz
        for t in taksit_tarihleri:
            if t <= bugun:
                payment_rows.append({
                    "krd_id": c["krd_id"], "cust_id": c["mno"],
                    "taksit_tutar": taksit_tutar,
                    "valor": t.strftime("%Y-%m-%d"),
                })
        odenen = sum(1 for t in taksit_tarihleri if t <= bugun)
        kalan_oran = max(0.0, 1.0 - odenen / taksit_sayisi)
    else:
        kalan_oran = 1.0  # yapilandirma: odemeler yapilmamis, risk sabit

    anapara_risk = round(c["lcy_tutar"] * kalan_oran, 2)
    faiz_risk = round(anapara_risk * c["faiz_oran"] * max(0.05, kalan_oran * 0.5), 2)
    valor_dt_usd = round(c["usd_tutar"] * kalan_oran, 2)

    risk_rows.append({
        "krd_id": c["krd_id"], "mno": c["mno"],
        "valor": c["valor"].strftime("%Y-%m-%d"), "pb": c["pb"],
        "vade": c["vade"].strftime("%Y-%m-%d"),
        "lcy_tutar": c["lcy_tutar"], "usd_tutar": c["usd_tutar"],
        "valor_dt_usd_tutar": valor_dt_usd,
        "anapara_risk": anapara_risk, "faiz_risk": faiz_risk,
    })

    # commitment
    commitment_tutar = round(c["lcy_tutar"] * random.uniform(0.0, 0.4), 2)
    commitment_rows.append({
        "krd_id": c["krd_id"], "mno": c["mno"],
        "pb": c["pb"], "commitment_tutar": commitment_tutar,
    })

dim_credit = pd.DataFrame(credit_rows)
credit_risk = pd.DataFrame(risk_rows)
commitment = pd.DataFrame(commitment_rows)
dim_payment = pd.DataFrame(payment_rows)

print(f"dim_credit: {len(dim_credit)} satir")
print(f"dim_payment: {len(dim_payment)} satir")
print(f"credit_risk: {len(credit_risk)} satir")
print(f"commitment: {len(commitment)} satir")

# ============================================================
# 4. dw.cust_limit
# ============================================================
limit_rows = []
limit_id = 90000
for mno in mnos:
    cust_credits = [c for c in temp_credits if c["mno"] == mno]
    total_lcy = sum(c["lcy_tutar"] for c in cust_credits) if cust_credits else random.uniform(50000, 2000000)
    sube = dim_customer[dim_customer["mno"] == mno]["sube"].values[0]
    region = dim_customer[dim_customer["mno"] == mno]["region"].values[0]

    genel_limit = round(total_lcy * random.uniform(1.2, 2.0), 2)
    bir_grup = round(genel_limit * random.uniform(0.5, 0.8), 2)
    iki_grup = round(genel_limit * random.uniform(0.3, 0.6), 2)
    uc_grup = round(genel_limit * random.uniform(0.1, 0.3), 2)

    start_dt = VALOR_START - timedelta(days=random.randint(30, 365))
    end_dt = start_dt + timedelta(days=random.choice([365, 730, 1095]))

    limit_rows.append({
        "limit_id": limit_id, "mno": mno,
        "genel_limit": genel_limit, "bir_grup_limit": bir_grup,
        "iki_grup_limit": iki_grup, "uc_grup_limit": uc_grup,
        "start_dt": start_dt.strftime("%Y-%m-%d"),
        "end_dt": end_dt.strftime("%Y-%m-%d"),
        "sube": sube, "region": region,
    })
    limit_id += 1

cust_limit = pd.DataFrame(limit_rows)
print(f"cust_limit: {len(cust_limit)} satir")

# ============================================================
# 5. dw.collateral
# ============================================================
collateral_rows = []
collateral_ids = sorted(random.sample(range(600000, 999999), NUM_COLLATERALS))

for i in range(NUM_COLLATERALS):
    cust_mno = random.choice(mnos)
    pb = random.choice(PARA_BIRIMLERI)
    tutar = round(random.uniform(20000, 8000000), 2)
    valor = random_valor()
    vade = valor + timedelta(days=random.choice([360, 720, 1095, 1825]))
    teminat_tur_id = random.choice(TEMINAT_IDS)
    finans_kurulusu = random.choice(["Ziraat", "Halkbank", "Vakifbank", "Isbank", "Garanti", "YKB", "Akbank"])

    collateral_rows.append({
        "collateral_id": collateral_ids[i], "cust_id": cust_mno,
        "collateral_pb": pb, "tutar": tutar,
        "finans_kurulusu": finans_kurulusu,
        "valor": valor.strftime("%Y-%m-%d"),
        "vade": vade.strftime("%Y-%m-%d"),
        "teminat_tur_id": teminat_tur_id,
    })

collateral = pd.DataFrame(collateral_rows)
print(f"collateral: {len(collateral)} satir")

# ============================================================
# 6. dw.collateral_usage (kredi-teminat iliskisi)
# ============================================================
usage_rows = []
relation_types = ["DIRECT", "CROSS", "POOL"]

for c in temp_credits:
    cust_collaterals = collateral[collateral["cust_id"] == c["mno"]]
    if len(cust_collaterals) == 0:
        continue
    num_links = random.randint(1, min(3, len(cust_collaterals)))
    linked = cust_collaterals.sample(n=num_links)
    for _, col_row in linked.iterrows():
        usage_rows.append({
            "credit_id": c["krd_id"],
            "collateral_id": col_row["collateral_id"],
            "relation_tp": random.choice(relation_types),
        })

collateral_usage = pd.DataFrame(usage_rows)
print(f"collateral_usage: {len(collateral_usage)} satir")

# ============================================================
# CSV EXPORT
# ============================================================
dim_customer.to_csv(os.path.join(OUTPUT_DIR, "dm_dim_customer.csv"), index=False, sep=";")
dim_currency.to_csv(os.path.join(OUTPUT_DIR, "dm_dim_currency.csv"), index=False, sep=";")
dim_credit.to_csv(os.path.join(OUTPUT_DIR, "dm_dim_credit.csv"), index=False, sep=";")
dim_payment.to_csv(os.path.join(OUTPUT_DIR, "dm_dim_payment.csv"), index=False, sep=";")
cust_limit.to_csv(os.path.join(OUTPUT_DIR, "dw_cust_limit.csv"), index=False, sep=";")
credit_risk.to_csv(os.path.join(OUTPUT_DIR, "dw_credit_risk.csv"), index=False, sep=";")
commitment.to_csv(os.path.join(OUTPUT_DIR, "dw_commitment.csv"), index=False, sep=";")
collateral.to_csv(os.path.join(OUTPUT_DIR, "dw_collateral.csv"), index=False, sep=";")
collateral_usage.to_csv(os.path.join(OUTPUT_DIR, "dw_collateral_usage.csv"), index=False, sep=";")

print(f"\nTum tablolar '{OUTPUT_DIR}' klasorune CSV olarak yazildi.")
