import streamlit as st
import pandas as pd
from database import load_data
from model import train_model

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Student Analytics System",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 AI-Powered Student Analytics System")
st.markdown("### Student Performance Prediction using Machine Learning")

# Sidebar
st.sidebar.header("📌 About Project")

st.sidebar.info("""
This project predicts student marks using:
- Attendance
- Study Hours
- Machine Learning

Technologies Used:
- Python
- Pandas
- Streamlit
- Scikit-learn
- GitHub
""")

# Load Dataset from CSV
df = load_data()

# Dataset Section
st.subheader("📊 Student Dataset")

st.dataframe(df)

# Dataset Statistics
st.subheader("📈 Dataset Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Marks", round(df["Marks"].mean(), 2))

with col2:
    st.metric("Highest Marks", df["Marks"].max())

with col3:
    st.metric("Average Attendance", round(df["Attendance"].mean(), 2))

# Charts
st.subheader("📉 Marks Analysis")

st.bar_chart(df.set_index("Student")["Marks"])

st.subheader("📉 Attendance Analysis")

st.line_chart(df.set_index("Student")["Attendance"])

# Scatter Chart
st.subheader("🔍 Relationship Between Attendance and Marks")

chart_data = pd.DataFrame({
    "Attendance": df["Attendance"],
    "Marks": df["Marks"]
})

st.scatter_chart(chart_data)

# Machine Learning Section
X = df[["Attendance", "StudyHours"]]
y = df["Marks"]

# Train Model using model.py
model = train_model(X, y)

# Prediction Section
st.subheader("🤖 Predict Student Performance")

attendance = st.slider(
    "Attendance Percentage",
    0,
    100,
    80
)

hours = st.slider(
    "Study Hours Per Day",
    0,
    10,
    4
)

# Prediction
prediction = model.predict([[attendance, hours]])

# Prediction Result
st.success(f"🎯 Predicted Marks: {prediction[0]:.2f}")

# Performance Category
if prediction[0] >= 85:
    st.info("🏆 Performance: Excellent")

elif prediction[0] >= 70:
    st.info("👍 Performance: Good")

else:
    st.warning("⚠️ Performance: Needs Improvement")

# AI Suggestions
st.subheader("💡 AI Suggestions")

if attendance < 75:
    st.warning(
        "Increase attendance to improve academic performance."
    )

if hours < 3:
    st.warning(
        "Increase daily study hours for better marks."
    )

if attendance >= 85 and hours >= 5:
    st.success(
        "Excellent academic consistency detected!"
    )

# Footer
st.markdown("---")
st.markdown("### 🚀 Developed by Shaik Sohel")
st.markdown("M.Tech | Data Science & AI Enthusiast")