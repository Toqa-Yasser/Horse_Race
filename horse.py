import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO


@st.cache_data
def load_data():
        return pd.read_pickle(BytesIO(requests.get("https://github.com/Toqa-Yasser/Horse_Race/blob/main/horseRace.pkl").content))
if 'df' not in st.session_state :
    df = load_data()
if 'modelt' not in st.session_state:
    model = joblib.load(BytesIO(requests.get('https://github.com/Toqa-Yasser/Horse_Race/blob/main/pipeline.h5').content))
# Function for "WON OR LOSE" Section
def page1():
    st.title("Predict: WON OR LOSE 🏇 ")
    st.write("Predict whether the horse will win or lose based on its features.")
    
    def getinput():
        horse_age = st.selectbox('Horse Age : ', options=[2, 3, 4], index=0)
        horse_country = st.selectbox('Horse Country : ', ['AUS', 'NZ', 'Other'])
        horsegear = st.selectbox('Horse Gear :', ['With', 'Without'])
        
        horse_gear = 'other'if horsegear == 'With' else '--'


        
        actual_weight = st.number_input("Actual Weight")
        draw = st.slider('Draw :', min_value=1, max_value=15, step=1)
        
        trainer_id = st.selectbox('Trainer ID :',df['trainer_id'].unique())
        win_rate_trainer = st.number_input('Win Rate of Trainer', min_value=0.0, max_value=1.0, value=df.groupby(['trainer_id'])['win_rate_trainer'].max().loc[trainer_id])
        
        jockey_id = st.selectbox('Jockey ID :',df['jockey_id'].unique())
        win_rate_jockey = st.number_input('Win Rate of Jockey', min_value=0.0, max_value=1.0, value=df.groupby(['jockey_id'])['win_rate_jockey'].max().loc[jockey_id])
        
        venue = st.selectbox('Venue?', ['ST', 'HV'])
        config = st.selectbox('Config?', ['A', 'A+3', 'B', 'B+2', 'C', 'C+3'])
        year = st.slider('Year', min_value=1997, max_value=2005, value=1997, step=1)
        month = st.slider('Month', min_value=1, max_value=12, value=1, step=1)
        day = st.selectbox("Select a Day of the Week:", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        

        numberofraces = st.number_input('Number Of Races Of Horse', min_value=0, value=1)
        total_wins = st.number_input('Number Of Won Races Of Horse', min_value=0,max_value=numberofraces , value=0)
        
        win_rate= total_wins / numberofraces


        horse_id = st.selectbox('Horse ID or guest?', ['Guest'] + df['horse_id'].unique().tolist())
        if horse_id == 'Guest':
            avg_finish_position = st.number_input('Average Finish Position', min_value=0.0, value=0.0)
        else:
            avg_finish_position = st.number_input('Average Finish Position', min_value=1.0, value=df.groupby('horse_id')['result'].mean().loc[horse_id])

        if horse_id == 'Guest':
            finish_time_trend = st.number_input('Finish Time Trend', min_value=-100.0, max_value=100.0, value=0.0)
        else:
            finish_time_trend = st.number_input('Finish Time Trend', min_value=-100.0, max_value=100.0, value=df.groupby('horse_id')['finish_time_trend'].max().loc[horse_id])

        return pd.DataFrame(data=[
            [horse_age, horse_country, horse_gear, actual_weight, draw, trainer_id, jockey_id, venue, config,
             year, month, day, total_wins, win_rate_jockey, win_rate_trainer, win_rate,
             avg_finish_position, finish_time_trend]
        ], columns=['horse_age', 'horse_country', 'horse_gear', 'actual_weight', 'draw', 'trainer_id', 'jockey_id',
                    'venue', 'config', 'year', 'month', 'day', 'total_wins', 'win_rate_jockey', 'win_rate_trainer',
                    'win_rate', 'avg_finish_position', 'finish_time_trend'])

    test = getinput()
    if st.button('Predict'):
        if model.predict(test)[0] :
            st.success(f'The horse is predicted to: WIN')
        else :
            st.error(f'The horse is predicted to: LOSE')

# Function for "ANALYSIS" Section
def page2():
    st.title("📊 Analysis")
    st.write("Detailed analysis and insights about the races and horses.")
    

    # App Title
    st.title("Comprehensive Horse Racing Data Analysis")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio(
        "Select a Section",
        ["Horse ID Lookup","Overview", "Univariate Analysis", "Bivariate Analysis", "Heatmaps", "Conclusions"]
    )
    
    # Overview Section
    if options == "Overview":
        st.header("Dataset Overview")
        st.write("### Dataset Shape:", df.shape)
        st.dataframe(df.head(10))
        st.write("### Columns in Dataset:")
        st.write(df.columns.tolist())


    elif options == "Horse ID Lookup":
        st.header("Lookup Data by Horse ID")
        horse_ids = df['horse_id'].unique()  # Get unique Horse IDs
        selected_horse_id = st.selectbox("Select a Horse ID:", options=horse_ids)

        # Filter data for the selected Horse ID
        horse_data = df[df['horse_id'] == selected_horse_id]
        
        if horse_data.empty:
            st.warning("No data available for the selected Horse ID.")
        else:
            st.subheader(f"Details for Horse ID: {selected_horse_id}")
            st.dataframe(horse_data)
            st.write("### Summary Statistics")
            st.write(horse_data.describe())

    
    # Univariate Analysis Section
    elif options == "Univariate Analysis":
        st.header("Univariate Analysis")
    
        # Horse Age Distribution
        st.subheader("Horse Age Distribution")
        fig_age = px.histogram(df, x='horse_age', title="Distribution of Horse Age", color_discrete_sequence=["skyblue"])
        st.plotly_chart(fig_age)


        st.subheader("Horse Rating Distribution")
        fig_rating = px.histogram(df, x=df['horse_rating'], title="Horse Rating Distribution")
        st.plotly_chart(fig_rating)
        st.write("**Conclusion**: Most of the horses have a rating of 60, indicating eligibility for competition in this race.")
    
        # Horse Gear
        st.subheader("Horse Gear Usage")
        fig_gear = px.histogram(df, x=df['horse_gear'], title="Horse Gear Usage")
        st.plotly_chart(fig_gear)
        st.write("**Conclusion**: Most horses competed without any specialized gear.")
    
        # Declared Weight
        st.subheader("Declared Weight Distribution")
        fig_declared_weight = px.box(df, x=df['declared_weight'], title="Declared Weight Distribution")
        st.plotly_chart(fig_declared_weight)
        st.write("**Conclusion**: The total weight the horse is expected to carry ranges from 940 lbs to 1300 lbs.")
    
        # Actual Weight
        st.subheader("Actual Weight Distribution")
        fig_actual_weight = px.box(df, x=df['actual_weight'], title="Actual Weight Distribution")
        st.plotly_chart(fig_actual_weight)
        st.write("**Conclusion**: The actual weight the horse carried ranges from 110 lbs to 133 lbs.")
    
        # Weight Difference
        st.subheader("Weight Difference Distribution")
        fig_weight_diff = px.box(df, x=df['weight_difference'], title="Weight Difference Distribution")
        st.plotly_chart(fig_weight_diff)
        st.write("""
        **Conclusion**: A large weight difference between declared and actual weights could indicate:
        - Health Issues or Fatigue: Unexpected weight changes may signal stress or poor conditioning.
        - Inconsistent Performance: Weight fluctuations might affect stamina or strength during a race.
        """)

        # Finish Time
        st.subheader("Finish Time Distribution")
        fig_finish_time = px.histogram(df, x='finish_time', title="Finish Time Distribution")
        st.plotly_chart(fig_finish_time)
        st.write("**Conclusion**: The average time for racers to finish is approximately 85 seconds.")
    
        # Win Odds
        st.subheader("Win Odds Distribution")
        fig_win_odds = px.histogram(df, x='win_odds', title="Win Odds Distribution")
        st.plotly_chart(fig_win_odds)
        st.write("""
        **Conclusion**: 10% of horses have win odds of 99 (the max), while 80% have odds ranging between 1 and 40.
        """)
    
        # Races by Year
        st.subheader("Races by Year")
        fig_year = px.histogram(df, x=df['year'].astype(str), title="Races by Year", height=600)
        st.plotly_chart(fig_year)
        st.write("""
        **Conclusion**: The years with the highest number of races are 2002, 2004, and 1999. The years with the lowest are 2005 and 1997.
        """)
    
        # Races by Month
        st.subheader("Races by Month")
        fig_month = px.histogram(df, x=df['month'].astype(str), title="Races by Month", height=600).update_layout(
            xaxis={'categoryarray': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
        st.plotly_chart(fig_month)
        st.write("""
        **Conclusion**: The month with the highest number of races is August. The months with the lowest are April and May.
        """)

        # Races by Day
        st.subheader("Races by Day of the Week")
        fig_day = px.histogram(df, x=df['day'], title="Races by Day of the Week").update_layout(
            xaxis={'categoryarray': ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Thursday', 'Friday']})
        st.plotly_chart(fig_day)
        st.write("**Conclusion**: Most races occur on Sunday or Wednesday.")
    
        
# Bivariate Analysis Section
    elif options == "Bivariate Analysis":
        st.header("Bivariate Analysis")
    
        # Horse Age vs Wins
        st.subheader("Horse Age vs Wins")
        fig_age_wins = px.histogram(df, x='horse_age', y='won', histfunc='sum', title="Horse Age vs Wins")
        st.plotly_chart(fig_age_wins)
    
        # Country vs Wins
        st.subheader("Horse Country vs Wins")
        fig_country_wins = px.histogram(df, x='horse_country', y='won', histfunc='sum', title="Country of Horses vs Wins")
        st.plotly_chart(fig_country_wins)
    
        # Draw vs Wins
        st.subheader("Draw vs Wins")
        fig_draw_wins = px.histogram(df, x='draw', y='won', title="Effect of Draw Position on Wins")
        st.plotly_chart(fig_draw_wins)
    
        # Horse Type vs Wins
        st.subheader("Horse Type vs Wins")
        fig_type_wins = px.histogram(df, x='horse_type', y='won', histfunc='sum', title="Horse Type vs Wins")
        st.plotly_chart(fig_type_wins)

        # Prize Distribution
        st.subheader("Prize Distribution by Race Class")
        fig_prize = px.box(df, x='race_class', y='prize', title="Race Class vs Prize Amount")
        st.plotly_chart(fig_prize)
    
        # Section Times and Winning Probability
        st.subheader("Section Times vs Wins")
        for i in range(1, 7):
            fig_section_time = px.histogram(df, x=f'time{i}runs', y='won', title=f'Section {i} Time vs Winning Probability')
            st.plotly_chart(fig_section_time)

        # Finish Time vs Win Odds
        st.subheader("Finish Time vs Win Odds")
        fig_finish_odds = px.scatter(df, x='finish_time', y='win_odds', title="Finish Time vs Win Odds")
        st.plotly_chart(fig_finish_odds)
    
        # Weight vs Wins
        st.subheader("Actual Weight vs Wins")
        fig_weight_wins = px.histogram(df, x='actual_weight', y='won', histfunc='sum', title="Actual Weight vs Wins")
        st.plotly_chart(fig_weight_wins)
    
        # Winning Probability by Section Positions
        st.subheader("Winning Probability by Section Positions")
        for i in range(1, 7):
            fig_section_pos = px.histogram(df, x=f'position_sec{i}', y='won', title=f'Section {i} Position vs Winning Probability')
            st.plotly_chart(fig_section_pos)
    
        # Race Distance vs Wins
        st.subheader("Race Distance vs Wins")
        fig_distance = px.histogram(df, x='distance', y='won', title="Race Distance vs Winning Probability")
        st.plotly_chart(fig_distance)
    
    # Heatmaps Section
    elif options == "Heatmaps":
        st.header("Correlation Heatmaps")
        
        st.subheader("Performance Metrics Heatmap")
        plt.figure(figsize=(14, 12))
        sns.heatmap(df[['finish_time', 'time1runs', 'time2runs', 'time3runs', 'position_sec1', 'position_sec2', 'position_sec3']].corr(),
                    annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)
    
        st.subheader("Horse Characteristics Heatmap")
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[['horse_age', 'horse_rating', 'actual_weight', 'draw', 'distance', 'prize']].corr(),
                    annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)
    
        st.subheader("Betting and Outcomes Heatmap")
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[['win_odds', 'place_odds', 'won', 'result']].corr(),
                    annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)
    
    # Conclusions Section
    elif options == "Conclusions":
        st.header("Key Conclusions")
        st.markdown("""
        ### Insights from the Analysis
        - **Horse Age**: Most winners are 3 years old, suggesting younger horses perform better.
        - **Country Bias**: Horses from Australia and New Zealand dominate the data and win more often.
        - **Horse Type**: Geldings win most frequently, likely due to their higher representation in the dataset.
        - **Finish Time**: Weak correlation with winning; no specific time guarantees a win.
        - **Draw Position**: Lower draw numbers significantly improve the chances of winning.
        - **Betting Insights**: Win odds alone do not strongly predict race outcomes.
        - **Race Distance**: Shorter distances tend to favor winning horses.
        - **Weight Impact**: Horses with higher actual weights have a slight advantage.
        - Most races are divided into three sections.
        - The minimum time taken in these races is **55.16 seconds**, achieved by HorseId 1792 with JockeyId 64.
        - The maximum time is **163.58 seconds**.
        - Most horses have a rating of 60, making them eligible for competition.
        - Most horses competed without any specialized gear.
        - The declared weight ranges from **940 lbs to 1300 lbs**, while the actual weight ranges from **110 lbs to 133 lbs**.
        - A large weight difference could indicate health issues, fatigue, or inconsistent performance.
        - The average finish time for races is approximately **85 seconds**.
        - The years with the highest number of races are **2002, 2004, and 1999**, and the lowest are **2005 and 1997**.
        - Most races occur in **August**, while the least occur in **April and May**.
        - Most races are held on **Sunday or Wednesday**.
        """)
    

# Sidebar with Persistent Navigation
st.sidebar.title("Horse Racing App 🏇 ")
st.sidebar.write("---")
page = st.sidebar.radio("Navigate", options=["Main Page", "Predict: WON OR LOSE", "Analysis"])

# Main Page Display Logic
if page == "Predict: WON OR LOSE":
    page1()
elif page == "Analysis":
    page2()
else:
    st.markdown(
        """
        <div style="text-align: center;">
            <h1> Welcome to the Horse Racing Predictor App!</h1>
            <h4>Select a section from the <b>sidebar</b> to begin. 👈 </h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image('race and horse.jpg')
