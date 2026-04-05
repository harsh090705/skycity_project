# 🍽️ SkyCity Auckland: Profit Intelligence Dashboard

A predictive and prescriptive analytics suite for multi-channel restaurant operations. This project uses **XGBoost** and **SHAP** to model how profit responds to channel mix (UberEats, DoorDash, Self-Delivery) and cost sensitivities.

## 🚀 Key Features
- **Predictive Modeling**: High-accuracy profit forecasting using Gradient Boosting.
- **Profit Sensitivity**: Real-time simulation of commission rate and delivery cost impacts.
- **Channel Optimization**: Grid-search algorithm to find the most profitable order distribution.
- **Explainable AI**: SHAP integration to visualize which features (like COGS or Subregion) drive the bottom line.

## 🛠️ Project Structure
- `data/`: Raw and engineered restaurant datasets.
- `notebooks/`: Exploratory Data Analysis (EDA).
- `models/`: Trained model binaries and performance reports.
- `app.py`: High-end Streamlit dashboard with custom CSS.

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/skycity-profit-intelligence.git](https://github.com/YOUR_USERNAME/skycity-profit-intelligence.git)
2.Install dependencies:
    pip install -r requirements.txt
3.Run the dashboard:
    streamlit run app.py