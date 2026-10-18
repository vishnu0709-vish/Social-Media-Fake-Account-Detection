import streamlit as st
import joblib
import pandas as pd
import re
import matplotlib.pyplot as plt
import gender_guesser.detector as gender

# ===============================
# 🔧 Load trained model
# ===============================
model_data = joblib.load("model/random_forest_model.pkl")
model = model_data["model"]
feature_columns = model_data["feature_columns"]

# ===============================
# 🧠 Feature extraction (same as training)
# ===============================
g_detector = gender.Detector(case_sensitive=False)

def extract_username_features_fast(usernames):
    if isinstance(usernames, str):
        usernames = [usernames]

    features = {
        'length': [],
        'num_digits': [],
        'num_upper': [],
        'num_underscore': [],
        'num_special': [],
        'num_vowels': [],
        'digit_ratio': [],
        'vowel_ratio': [],
        'gender_code': [],
        'spam_keyword_count': []
    }

    spam_keywords = [
        'free', 'gift', 'win', 'cash', 'offer', 'promo', 'deal',
        'bot', 'follow', 'airdrop', 'prize', 'claim', 'reward', 'crypto'
    ]

    gender_dict = {
        'female': -2, 'mostly_female': -1, 'unknown': 0,
        'andy': 0, 'mostly_male': 1, 'male': 2
    }

    for name in usernames:
        name = str(name).strip()
        length = len(name)
        num_digits = sum(c.isdigit() for c in name)
        num_upper = sum(c.isupper() for c in name)
        num_underscore = name.count('_')
        num_special = len(re.findall(r"[^a-zA-Z0-9_]", name))
        num_vowels = len(re.findall(r"[aeiouAEIOU]", name))
        digit_ratio = num_digits / length if length > 0 else 0
        vowel_ratio = num_vowels / length if length > 0 else 0

        first_part = re.split(r'[_\W]+', name)[0]
        gender_type = g_detector.get_gender(first_part)
        gender_code = gender_dict.get(gender_type, 0)

        spam_count = sum(1 for word in spam_keywords if word in name.lower())

        features['length'].append(length)
        features['num_digits'].append(num_digits)
        features['num_upper'].append(num_upper)
        features['num_underscore'].append(num_underscore)
        features['num_special'].append(num_special)
        features['num_vowels'].append(num_vowels)
        features['digit_ratio'].append(digit_ratio)
        features['vowel_ratio'].append(vowel_ratio)
        features['gender_code'].append(gender_code)
        features['spam_keyword_count'].append(spam_count)

    df_features = pd.DataFrame(features)
    df_features = df_features.reindex(columns=feature_columns, fill_value=0)
    return df_features


# ===============================
# 🎯 Prediction function
# ===============================
def predict_username(username):
    features = extract_username_features_fast(username)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0][int(pred)]
    return pred, proba, features


# ===============================
# 🎨 Streamlit UI
# ===============================
st.set_page_config(page_title="Social Media Fake Account Detector", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Social Media Fake Account Detector")
st.write("Detect fake or spam accounts using linguistic patterns in usernames.")

tab1, tab2 = st.tabs(["🔹 Single Username", "📂 Batch Prediction"])

# ===============================
# 🧍 Single Username Prediction
# ===============================
with tab1:
    username = st.text_input("Enter Username", placeholder="e.g. xXx_freegift99_xXx")

    if st.button("Predict"):
        if not username.strip():
            st.warning("Please enter a valid username!")
        else:
            pred, conf, features = predict_username(username)
            conf_percent = round(conf * 100, 2)

            if pred == 1:
                st.success(f"✅ Genuine Account Detected! \nConfidence: {conf_percent}%")
            else:
                st.error(f"⚠️ Fake/Spam Account Detected! \nConfidence: {conf_percent}%")

            with st.expander("🔍 View Extracted Username Features"):
                st.dataframe(features)

            # Plot spam features visually
            st.subheader("📊 Feature Importance for Username")
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots()
            ax.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax.invert_yaxis()
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.title("Feature Importance (Random Forest)")
            st.pyplot(fig)

# ===============================
# 📂 Batch Prediction
# ===============================
with tab2:
    uploaded_file = st.file_uploader("Upload a CSV file with a 'name' column", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        if "name" not in df_batch.columns:
            st.error("The uploaded file must contain a 'name' column.")
        else:
            st.write("✅ File uploaded successfully!")
            df_features = extract_username_features_fast(df_batch["name"])
            df_features = df_features.reindex(columns=feature_columns, fill_value=0)
            preds = model.predict(df_features)
            probas = model.predict_proba(df_features)

            df_batch["Prediction"] = ["Fake" if p == 0 else "Genuine" for p in preds]
            df_batch["Confidence"] = [round(max(prob)*100, 2) for prob in probas]

            st.write("📊 Batch Results:")
            st.dataframe(df_batch[["name", "Prediction", "Confidence"]])

            csv = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "batch_predictions.csv", "text/csv")

st.markdown("---")
st.caption("Developed with ❤️ using Machine Learning")
