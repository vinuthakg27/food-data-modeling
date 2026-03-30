import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FoodBill AI - Predict Your Delivery Bill",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

[data-testid="stAppViewContainer"] {
    background-image:
        linear-gradient(135deg, rgba(15,10,5,0.82) 0%, rgba(30,12,0,0.75) 50%, rgba(10,18,10,0.85) 100%),
        url('https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=1800&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: rgba(10,6,2,0.92) !important; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #f0ebe3;
}

.hero { text-align: center; padding: 2.8rem 1rem 1.6rem; }
.hero-emoji { font-size: 3.8rem; line-height: 1; }
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 900;
    background: linear-gradient(90deg, #ff9b4a 0%, #ffcc70 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.3rem 0 0.6rem;
}
.hero p { color: #c8b99a; font-size: 1.05rem; font-weight: 300; letter-spacing: 0.5px; }

.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,160,60,0.18);
    border-radius: 20px;
    padding: 1.8rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #ff9b4a;
    margin-bottom: 1rem;
}

label, .stSlider label, .stSelectbox label, .stNumberInput label, .stRadio label {
    color: #d4c5ae !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

input[type="number"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,160,60,0.25) !important;
    border-radius: 10px !important;
    color: #f0ebe3 !important;
}

[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,160,60,0.25) !important;
    border-radius: 10px !important;
    color: #f0ebe3 !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff6b35 0%, #ff9b4a 50%, #ffcc70 100%) !important;
    color: #1a0a00 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 6px 28px rgba(255,107,53,0.4) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 36px rgba(255,107,53,0.55) !important;
}

.result-card {
    background: linear-gradient(135deg, rgba(255,107,53,0.15) 0%, rgba(255,204,112,0.1) 100%);
    border: 1px solid rgba(255,155,74,0.4);
    border-radius: 20px;
    padding: 2rem 1.6rem;
    text-align: center;
    margin-top: 1.2rem;
    box-shadow: 0 8px 40px rgba(255,107,53,0.15);
}
.result-label { font-size: 0.8rem; letter-spacing: 3px; text-transform: uppercase; color: #ff9b4a; margin-bottom: 0.5rem; }
.result-amount {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 4.2rem);
    font-weight: 900;
    background: linear-gradient(90deg, #ff9b4a, #ffcc70);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

.metric-tile {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,160,60,0.15);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-tile .mt-val { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: #ffcc70; }
.metric-tile .mt-label { font-size: 0.75rem; color: #a89880; letter-spacing: 1.5px; text-transform: uppercase; }

.breakdown-row {
    display: flex;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,160,60,0.1);
    font-size: 0.95rem;
}
.breakdown-row:last-child { border-bottom: none; font-weight: 600; color: #ff9b4a; }

[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
[data-baseweb="tab"] { background: transparent !important; border-radius: 9px !important; color: #a89880 !important; font-weight: 500 !important; }
[aria-selected="true"][data-baseweb="tab"] { background: rgba(255,155,74,0.18) !important; color: #ff9b4a !important; }

hr { border-color: rgba(255,160,60,0.12) !important; }
.footer { text-align: center; padding: 1.5rem 0 0.5rem; color: #6b5d4e; font-size: 0.8rem; letter-spacing: 1px; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-emoji">🍜</div>
    <h1>FoodBill AI</h1>
    <p>Predict your delivery bill with machine learning · Instant · Accurate</p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        st.error("⚠️ model.pkl not found. Place it in the same folder as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

expected_features = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None

tab1, tab2, tab3, tab4 = st.tabs(["🔮  Manual Predict", "🤖  Chat Predictor", "📊  Features", "⚙️  Model Info"])

# ── TAB 1: Manual Predict ─────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📦 Order Details</div>', unsafe_allow_html=True)
        num_items = st.number_input("Number of Items", min_value=1, max_value=50, value=3, step=1)
        avg_item_price = st.number_input("Avg Item Price (₹)", min_value=0.0, max_value=500.0, value=150.0, step=10.0)
        discount_percent = st.slider("Discount %", min_value=0.0, max_value=50.0, value=0.0, step=1.0, format="%.0f%%")
        delivery_distance = st.number_input("Delivery Distance (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        delivery_rating = st.slider("Delivery Rating ⭐", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">👤 Customer</div>', unsafe_allow_html=True)
        customer_age = st.number_input("Age", min_value=15, max_value=80, value=28, step=1)
        customer_gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        weekend = st.radio("Day", ["Weekday", "Weekend"], horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🍽️ Restaurant & Cuisine</div>', unsafe_allow_html=True)
        restaurant_name = st.selectbox("Restaurant", ["McDonald's", "KFC", "Domino's", "Pizza Hut", "Subway"])
        cuisine_type = st.selectbox("Cuisine Type", ["Chinese", "Healthy", "Indian", "Italian", "Japanese"])
        meal_time = st.selectbox("Meal Time", ["Breakfast", "Lunch", "Dinner"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("✨  Predict My Bill", use_container_width=True)

        if predict_clicked:
            try:
                weekend_val = 1 if weekend == "Weekend" else 0
                gender_val  = 1 if customer_gender == "Male" else 0

                feature_dict = {
                    'num_items': num_items,
                    'avg_item_price': avg_item_price,
                    'delivery_distance_km': delivery_distance,
                    'delivery_rating': delivery_rating,
                    'customer_age': customer_age,
                    'discount_percent': discount_percent,
                    'customer_gender': gender_val,
                    'weekend': weekend_val,
                }
                for cuisine in ["chinese", "healthy", "indian", "italian", "japanese"]:
                    feature_dict[f'cuisine_type_{cuisine}'] = 1 if cuisine_type.lower() == cuisine else 0
                for meal in ["breakfast", "lunch", "dinner"]:
                    feature_dict[f'meal_time_{meal}'] = 1 if meal_time.lower() == meal else 0
                for rest in ["domino's", "kfc", "mcdonald's", "pizza hut", "subway"]:
                    feature_dict[f'restaurant_name_{rest}'] = 1 if restaurant_name.lower() == rest else 0

                input_df = pd.DataFrame([feature_dict])
                if expected_features:
                    for feat in expected_features:
                        if feat not in input_df.columns:
                            input_df[feat] = 0
                    input_df = input_df[expected_features]

                prediction = max(0, model.predict(input_df)[0])

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Estimated Bill</div>
                    <div class="result-amount">₹{prediction:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                subtotal = avg_item_price * num_items
                discount_amt = subtotal * (discount_percent / 100)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-tile"><div class="mt-val">₹{subtotal:,.0f}</div><div class="mt-label">Subtotal</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-tile"><div class="mt-val">-₹{discount_amt:,.0f}</div><div class="mt-label">Discount</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-tile"><div class="mt-val">{delivery_rating}⭐</div><div class="mt-label">Rating</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">💸 Breakdown</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="breakdown-row"><span>Subtotal ({num_items} items × ₹{avg_item_price:.0f})</span><span>₹{subtotal:,.2f}</span></div>
                <div class="breakdown-row"><span>Discount ({discount_percent:.0f}%)</span><span>-₹{discount_amt:,.2f}</span></div>
                <div class="breakdown-row"><span>🎯 Predicted Total</span><span>₹{prediction:,.2f}</span></div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                with st.expander("Debug info"):
                    st.write(str(e))
                    if expected_features:
                        st.write("Expected features:", expected_features)

# ── TAB 2: Chat Predictor ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label" style="padding-top:0.5rem;">🤖 Chat with AI to predict your bill</div>', unsafe_allow_html=True)

    # Chat steps — every step has clickable chip buttons + optional custom text input
    CHAT_STEPS = [
        {
            "key": "restaurant_name",
            "question": "🏪 Which restaurant are you ordering from?",
            "options": ["McDonald's", "KFC", "Domino's", "Pizza Hut", "Subway"],
            "hint": "Not listed? Type the restaurant name below",
            "custom": True
        },
        {
            "key": "cuisine_type",
            "question": "🍽️ What type of cuisine?",
            "options": ["Chinese", "Healthy", "Indian", "Italian", "Japanese"],
            "hint": "Not listed? Type the cuisine below",
            "custom": True
        },
        {
            "key": "meal_time",
            "question": "⏰ What meal time is this for?",
            "options": ["Breakfast", "Lunch", "Dinner"],
            "hint": "",
            "custom": False
        },
        {
            "key": "num_items",
            "question": "🛒 How many items are you ordering?",
            "options": ["1", "2", "3", "4", "5", "6"],
            "hint": "Not listed? Type a number below",
            "custom": True
        },
        {
            "key": "avg_item_price",
            "question": "💰 What is the average price per item in Rs?",
            "options": ["100", "150", "200", "300", "400", "500"],
            "hint": "Not listed? Type the price below",
            "custom": True
        },
        {
            "key": "discount_percent",
            "question": "🏷️ Any discount? Select the percentage (pick 0 if none)",
            "options": ["0", "5", "10", "15", "20", "30"],
            "hint": "Not listed? Type the discount % below",
            "custom": True
        },
        {
            "key": "delivery_distance_km",
            "question": "📍 Delivery distance in km?",
            "options": ["1", "2", "3", "5", "7", "10"],
            "hint": "Not listed? Type the distance below",
            "custom": True
        },
        {
            "key": "delivery_rating",
            "question": "⭐ Delivery rating (1.0 to 5.0)?",
            "options": ["3.0", "3.5", "4.0", "4.5", "5.0"],
            "hint": "Not listed? Type the rating below",
            "custom": True
        },
        {
            "key": "customer_age",
            "question": "👤 What is the customer age?",
            "options": ["18", "22", "25", "28", "35", "45"],
            "hint": "Not listed? Type the age below",
            "custom": True
        },
        {
            "key": "customer_gender",
            "question": "🧑 Customer gender?",
            "options": ["Male", "Female"],
            "hint": "",
            "custom": False
        },
        {
            "key": "weekend",
            "question": "📅 Is this a weekend order?",
            "options": ["Yes", "No"],
            "hint": "",
            "custom": False
        },
    ]

    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hey! I am your FoodBill AI assistant. I will ask you a few quick questions — just click an option or type your own answer. Let's predict your bill!"}
        ]
    if "chat_step" not in st.session_state:
        st.session_state.chat_step = 0
    if "chat_data" not in st.session_state:
        st.session_state.chat_data = {}
    if "chat_done" not in st.session_state:
        st.session_state.chat_done = False

    def run_model_prediction(data):
        try:
            restaurant_name_c = data.get("restaurant_name", "McDonald's")
            cuisine_type_c    = data.get("cuisine_type", "Indian")
            meal_time_c       = data.get("meal_time", "Lunch")
            weekend_val       = 1 if data.get("weekend", "No") == "Yes" else 0
            gender_val        = 1 if data.get("customer_gender", "Male") == "Male" else 0

            feature_dict = {
                "num_items":            float(data.get("num_items", 1)),
                "avg_item_price":       float(data.get("avg_item_price", 100)),
                "delivery_distance_km": float(data.get("delivery_distance_km", 3)),
                "delivery_rating":      float(data.get("delivery_rating", 4)),
                "customer_age":         float(data.get("customer_age", 25)),
                "discount_percent":     float(data.get("discount_percent", 0)),
                "customer_gender":      gender_val,
                "weekend":              weekend_val,
            }
            for cuisine in ["chinese", "healthy", "indian", "italian", "japanese"]:
                feature_dict[f"cuisine_type_{cuisine}"] = 1 if cuisine_type_c.lower() == cuisine else 0
            for meal in ["breakfast", "lunch", "dinner"]:
                feature_dict[f"meal_time_{meal}"] = 1 if meal_time_c.lower() == meal else 0
            for rest in ["domino's", "kfc", "mcdonald's", "pizza hut", "subway"]:
                feature_dict[f"restaurant_name_{rest}"] = 1 if restaurant_name_c.lower() == rest else 0

            input_df = pd.DataFrame([feature_dict])
            if expected_features:
                for feat in expected_features:
                    if feat not in input_df.columns:
                        input_df[feat] = 0
                input_df = input_df[expected_features]

            return max(0, model.predict(input_df)[0])
        except Exception:
            return None

    def reset_chat():
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hey! I am your FoodBill AI assistant. I will ask you a few quick questions — just click an option or type your own answer. Let's predict your bill!"}
        ]
        st.session_state.chat_step = 0
        st.session_state.chat_data = {}
        st.session_state.chat_done = False

    def pick_option(value, key):
        st.session_state.chat_messages.append({"role": "user", "content": value})
        st.session_state.chat_data[key] = value
        st.session_state.chat_step += 1
        st.rerun()

    # ── Render chat history ──
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Active question ──
    if not st.session_state.chat_done:
        step = st.session_state.chat_step

        if step < len(CHAT_STEPS):
            current = CHAT_STEPS[step]

            # Show the question bubble if not already shown
            last_msg = st.session_state.chat_messages[-1] if st.session_state.chat_messages else {}
            if last_msg.get("content") != current["question"]:
                st.session_state.chat_messages.append({"role": "assistant", "content": current["question"]})
                with st.chat_message("assistant"):
                    st.markdown(current["question"])

            # ── Chip buttons ──
            num_opts = len(current["options"])
            cols = st.columns(num_opts)
            for i, opt in enumerate(current["options"]):
                if cols[i].button(opt, key=f"chip_{step}_{i}", use_container_width=True):
                    pick_option(opt, current["key"])

            # ── Optional custom text input below chips ──
            if current["custom"]:
                if current["hint"]:
                    st.caption(current["hint"])
                user_typed = st.chat_input("Type a custom value and press Enter...")
                if user_typed:
                    pick_option(user_typed, current["key"])

        else:
            # ── All answered — predict ──
            st.session_state.chat_done = True
            prediction = run_model_prediction(st.session_state.chat_data)

            if prediction is not None:
                data = st.session_state.chat_data
                subtotal     = float(data.get("num_items", 1)) * float(data.get("avg_item_price", 0))
                discount_amt = subtotal * float(data.get("discount_percent", 0)) / 100

                result_msg = (
                    "🎉 **Here is your predicted bill!**\n\n"
                    "| Detail | Value |\n|---|---|\n"
                    f"| Restaurant | {data.get('restaurant_name', '-')} |\n"
                    f"| Cuisine | {data.get('cuisine_type', '-')} |\n"
                    f"| Meal Time | {data.get('meal_time', '-')} |\n"
                    f"| Items | {data.get('num_items', '-')} x Rs {data.get('avg_item_price', '-')} |\n"
                    f"| Discount | {data.get('discount_percent', 0)}% -> -Rs {discount_amt:,.2f} |\n\n"
                    f"## Predicted Total: Rs {prediction:,.2f}\n\n"
                    "*Powered by the same trained model.pkl as the manual predictor!*"
                )
                st.session_state.chat_messages.append({"role": "assistant", "content": result_msg})
                with st.chat_message("assistant"):
                    st.markdown(result_msg)
            else:
                st.error("Prediction failed. Please try again.")

    # ── Reset button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Start Over", key="chat_reset"):
        reset_chat()
        st.rerun()

# ── TAB 3: Features ───────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📐 Numerical Features</div>', unsafe_allow_html=True)
    num_info = pd.DataFrame({
        "Feature":     ["num_items","avg_item_price","delivery_distance_km","delivery_rating","customer_age","discount_percent"],
        "Description": ["Items ordered","Avg price per item (₹)","Delivery distance (km)","Delivery rating (1–5 ⭐)","Customer age (years)","Discount applied (%)"],
        "Range":       ["1–50","₹0–500","0–50 km","1.0–5.0","15–80 yrs","0–50%"],
    })
    st.dataframe(num_info, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🏷️ Categorical Features</div>', unsafe_allow_html=True)
    cat_info = pd.DataFrame({
        "Feature":  ["customer_gender","cuisine_type","meal_time","restaurant_name","weekend"],
        "Options":  ["Male, Female","Chinese, Healthy, Indian, Italian, Japanese","Breakfast, Lunch, Dinner","McDonald's, KFC, Domino's, Pizza Hut, Subway","Weekday, Weekend"],
    })
    st.dataframe(cat_info, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 4: Model Info ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🤖 Model Details</div>', unsafe_allow_html=True)
    n_feat = model.n_features_in_ if hasattr(model, 'n_features_in_') else "?"
    st.markdown(f"""
| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Input features | {n_feat} |
| Target variable | `total_bill` (₹) |
| Framework | scikit-learn |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📈 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
The model was trained on **200 food delivery orders** and learns the relationship
between order characteristics (items, prices, distance, cuisine, etc.) and the final bill.

Key steps in the pipeline:
- **Label encoding** for `customer_gender` and `weekend`
- **One-hot encoding** for `cuisine_type`, `meal_time`, and `restaurant_name`
- **Linear Regression** fit on the cleaned dataset
- Predictions are clipped at ₹0 to avoid negative bills
    """)
    if expected_features:
        with st.expander("View all model features"):
            st.code("\n".join(expected_features))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">🍜 FoodBill AI · Powered by Streamlit & scikit-learn</div>', unsafe_allow_html=True)