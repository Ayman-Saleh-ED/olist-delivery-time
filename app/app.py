from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
FEATURES = ["customer_state", "seller_state", "category", "price", "freight", "weight",
            "volume", "n_items", "n_sellers", "distance_km", "same_state",
            "limit_days", "month", "weekday", "hour"]
NAMES = {
    "bed_bath_table": "Bed & Bath Set", "health_beauty": "Health & Beauty Kit",
    "sports_leisure": "Sports & Leisure Item", "furniture_decor": "Home Decor Piece",
    "computers_accessories": "Computer Accessory", "housewares": "Housewares Item",
    "watches_gifts": "Watch / Gift", "telephony": "Phone Accessory",
    "auto": "Car Accessory", "toys": "Toy",
}

st.set_page_config(page_title="Olist Store", page_icon="🛒")
st.markdown("""
<style>
  .block-container { max-width: 640px; padding-top: 2rem; }
  .brand { font-size: 1.5rem; font-weight: 800; color: #1c629b; letter-spacing: -.02em; }
  .card { border: 1px solid #dde6f0; border-radius: 14px; padding: 1.1rem 1.3rem; margin: .6rem 0 1.1rem; background: white; }
  .card small { color: #64748b; }
  .eta { border-radius: 16px; padding: 1.6rem; text-align: center; background: #edf7ff; border: 1px solid #a9d2ee; margin: 1rem 0; }
  .eta .big { font-size: 2.4rem; font-weight: 850; color: #163f6e; line-height: 1.2; }
  .eta .sub { color: #526780; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(HERE / "models" / "hgb_total.joblib")


@st.cache_data
def load_catalog():
    products = pd.read_csv(HERE / "products.csv")
    products["name"] = products["category"].map(NAMES)
    addresses = pd.read_csv(HERE / "addresses.csv")
    return products, addresses


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(a)))


def predict_days(product, address) -> float:
    now = datetime.now()
    row = {
        "customer_state": address["state"], "seller_state": product["seller_state"],
        "category": product["category"], "price": product["price"], "freight": product["freight"],
        "weight": product["weight"], "volume": product["volume"], "n_items": 1, "n_sellers": 1,
        "distance_km": haversine_km(product["s_lat"], product["s_lng"], address["lat"], address["lng"]),
        "same_state": int(address["state"] == product["seller_state"]),
        "limit_days": product["limit_days"], "month": now.month, "weekday": now.weekday(), "hour": now.hour,
    }
    return max(1.0, float(load_model().predict(pd.DataFrame([row], columns=FEATURES))[0]))


products, addresses = load_catalog()
st.session_state.setdefault("page", "shop")

st.markdown('<div class="brand">🛒 Olist Store</div>', unsafe_allow_html=True)

if st.session_state.page == "shop":
    st.subheader("1. Choose your product")
    prod_labels = (products["name"] + "  -  R$ " + products["price"].map("{:,.2f}".format)).tolist()
    product = products.iloc[prod_labels.index(st.selectbox("Product", prod_labels))]
    st.markdown(f'<div class="card"><b>{product["name"]}</b><br>'
                f'<small>Sold by a seller in {product["seller_city"].title()}, {product["seller_state"]}</small></div>',
                unsafe_allow_html=True)

    st.subheader("2. Delivery address")
    addr_labels = (addresses["city"] + ", " + addresses["state"]).tolist()
    address = addresses.iloc[addr_labels.index(st.selectbox("Deliver to", addr_labels))]

    shipping = "Free" if product["freight"] == 0 else f"R$ {product['freight']:,.2f}"
    st.markdown(f'<div class="card">Item: R$ {product["price"]:,.2f}<br>Shipping: {shipping}<br>'
                f'<b>Total: R$ {product["price"] + product["freight"]:,.2f}</b></div>', unsafe_allow_html=True)

    if st.button("Order Now", type="primary", use_container_width=True):
        st.session_state.update(page="done", product=product.to_dict(), address=address.to_dict())
        st.rerun()
else:
    product, address = st.session_state.product, st.session_state.address
    days = round(predict_days(product, address))
    arrival = datetime.now() + timedelta(days=days)
    st.success("Thank you! Your order is confirmed.")
    st.markdown(f'<div class="eta"><div class="sub">Estimated delivery</div>'
                f'<div class="big">{arrival.strftime("%a, %d %b %Y")}</div>'
                f'<div class="sub">in about {days} days</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><b>{product["name"]}</b><br>'
                f'<small>From {product["seller_city"].title()}, {product["seller_state"]} '
                f'to {address["city"]}, {address["state"]}</small></div>', unsafe_allow_html=True)
    if st.button("Place another order", use_container_width=True):
        st.session_state.page = "shop"
        st.rerun()
