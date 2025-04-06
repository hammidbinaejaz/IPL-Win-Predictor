import streamlit as st
import pickle
import pandas as pd
import os

# 📌 Ensure correct loading of the model
file_path = os.path.join(os.path.dirname(__file__), 'pipe.pkl')

# Check if pipe.pkl exists
if not os.path.exists(file_path):
    st.error(f"🚨 Model file 'pipe.pkl' not found at: {file_path}")
    st.stop()

# Load the model
with open(file_path, 'rb') as f:
    pipe = pickle.load(f)

# 🎯 Team & City Options
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# 🏏 Streamlit UI
st.title('🏏 IPL Win Predictor')

# 📌 Team Selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

# 📌 City Selection
selected_city = st.selectbox('Select Host City', sorted(cities))

# 📌 Target Score
target = st.number_input('Target Score', min_value=1, step=1)

# 📌 Match Details
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('Wickets Lost', min_value=0, max_value=10, step=1)

# 🎯 Prediction Button
if st.button('🔮 Predict Win Probability'):
    if overs == 0:
        crr = 0  # Avoid division by zero
    else:
        crr = score / overs

    runs_left = max(target - score, 0)  # Ensuring non-negative runs left
    balls_left = max(120 - int(overs * 6), 0)  # Prevent negative values
    remaining_wickets = 10 - wickets_out
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0  # Prevent ZeroDivisionError

    # 🏏 Creating Input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # 🔥 Predict Probability
    try:
        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0] * 100  # Convert to percentage
        win_prob = result[0][1] * 100

        # 🎯 Display Prediction
        st.subheader(f"🏆 {batting_team} Winning Probability: **{round(win_prob, 1)}%**")
        st.subheader(f"⚾ {bowling_team} Winning Probability: **{round(loss_prob, 1)}%**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
