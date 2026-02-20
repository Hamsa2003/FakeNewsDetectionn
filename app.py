import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.markdown("<h1 style='text-align: center;'>📰 Fake News Detection App</h1>", unsafe_allow_html=True)

st.markdown("### 📊 Model Accuracy: 98.41%")

st.write("Enter a news article below and check whether it is Real or Fake.")

input_text = st.text_area("✍ Enter News Text Here", height=200)

if st.button("🔎 Predict Now"):

    if input_text.strip() == "":
        st.warning("⚠ Please enter some text.")

    else:
        vector_input = vectorizer.transform([input_text])
        prediction = model.predict(vector_input)
        probability = model.predict_proba(vector_input)

        import pandas as pd

        fake_prob = probability[0][0] * 100
        real_prob = probability[0][1] * 100

        if prediction[0] == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        st.markdown("### 📊 Prediction Confidence")

        chart_data = pd.DataFrame({
            "Fake": [fake_prob],
            "Real": [real_prob]
        })

        st.bar_chart(chart_data)