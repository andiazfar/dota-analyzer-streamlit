import streamlit as st
import pandas as pd

import requests
import json
import os

#%matplotlib inline
import matplotlib.pyplot as plt

# https://github.com/aliavni/awesome-data-explorer/blob/master/app.py

# Header of the Web Application!
st.write("""
# Simple DOTA2 Visualization and Machine Learning App
There will be multiple parts to it!

1. Visualization of data on the best hero for the player.
2. A machine learning model that predicts the outcome of their matches based on their match history
""")

st.sidebar.title("Hello!")
st.sidebar.subheader("""
This is a simple page for me to experiment on Data Science and it's deployment.
Do visit my GitHub at link below!
""")

link = 'https://github.com/andiazfar'
st.sidebar.markdown(link, unsafe_allow_html=True)


player_id = st.text_input("Please input the player ID: ", "e.g. 128683292")
# player_id = 128683292 # Default value



# First we import all files 1 by 1 from the OpenDota API
# https://stackoverflow.com/questions/42518864/convert-json-data-from-request-into-pandas-dataframe

if st.button('Don\'t forget to press enter up top!'):
    st.write('Running!')
    # import_data_from_API() # This saves a txt file, but we can skip that!!

    r_heroes = requests.get("https://api.opendota.com/api/heroes")
    j_heroes = r_heroes.json()
    hero_table = pd.DataFrame.from_dict(j_heroes)
    hero_table = hero_table.rename(columns={'id':'hero_id'})
    hero_table['hero_id'] = hero_table['hero_id'].astype(int)
    hero_table

    r_p_cmd = "https://api.opendota.com/api/players/"+str(player_id)+"/heroes"
    r_player_winrate = requests.get(r_p_cmd)
    j_player_winrate = r_player_winrate.json()
    player_table = pd.DataFrame.from_dict(j_player_winrate)
    player_table['hero_id'] = player_table['hero_id'].astype(int)
    player_table

    table = pd.merge(hero_table, player_table, on='hero_id')
    table_shown = table.set_index("localized_name")
    table_shown = table_shown[['games','win']]
    st.header("This is the player's hero stats!")
    st.write(table_shown)

    sort_options ='win'
    option_top = 10

    topn = table.sort_values(by=sort_options, ascending=False).set_index('localized_name').head(option_top)
    topn = topn.drop(columns=["name","last_played"])
    topn["win_rate"] = (topn.win/topn.games)*100
    topn=topn.sort_values(by='win_rate', ascending=False)
    st.header("This is the player's top 10 heroes!")
    topn_shown = topn[['games','win','win_rate']]
    st.write(topn_shown)


    # ++++++++++++++++++++++++ #
    # First: Graph on Win Rate #
    # ++++++++++++++++++++++++ #

    color_list = ["#347cbf"]*option_top
    # Change the color to green, of the highest current features
    feature = topn.win_rate.tolist()
    max_val = feature.index(max(feature))

    color_list[max_val] = "#24a627"

    st.header("Some plots on the player's with their best heroes!")
    st.write("""
    
    """)
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    # fig, axs = plt.subplots(2)
    ax.bar(topn.index, topn.win_rate, color=color_list)
    ax.set_title("Win rate percentage by best heroes (%)")
    ax.set_xlabel("Top " +str(option_top) +" Heroes")
    plt.xticks(rotation=90)
    ax.set_ylabel("Win Rate").set_rotation(0)
    plt.ylabel("Win Rate",labelpad=60)
    st.pyplot(fig)


    st.write("""
    
    """)
    # +++++++++++++++++++++++++++++ #
    # Second: Graph on Absolute Win #
    # +++++++++++++++++++++++++++++ #

    color_list = ["#347cbf"]*option_top
    # Change the color to green, of the highest current features
    feature = topn.win.tolist()
    max_val = feature.index(max(feature))

    color_list[max_val] = "#24a627"

    fig2 = plt.figure(figsize=(20, 10))
    ax2 = fig2.add_subplot()
    ax2.bar(topn.index, topn.win, color=color_list)
    ax2.set_title("Absolute win count by Top 10 heroes")
    ax2.set_xlabel("Top " +str(option_top) +" Heroes")
    plt.xticks(rotation=90)
    ax2.set_ylabel("Win Rate").set_rotation(0)
    plt.ylabel("Win Rate",labelpad=60)
    st.pyplot(fig2)

    # +++++++++++++++++++++++++++++++++++++++++++++++ #
    # Third: Scatter Plot on Percentage and Hero Type #
    # +++++++++++++++++++++++++++++++++++++++++++++++ #

    scatter_table = table
    scatter_table["win_rate"] = (scatter_table.win/scatter_table.games)*100
    scatter_table = scatter_table[['hero_id','primary_attr','win_rate']]

    scatter_table_str = scatter_table.loc[scatter_table['primary_attr'] == "str"]
    scatter_table_agi = scatter_table.loc[scatter_table['primary_attr'] == "agi"]
    scatter_table_int = scatter_table.loc[scatter_table['primary_attr'] == "int"]

    fig3 = plt.figure(figsize=(20, 10))
    # plt.rcParams.update({'font.size': 22})
    ax3 = fig3.add_subplot()
    ax3.scatter(x=scatter_table_str.hero_id ,y=scatter_table_str.win_rate, marker=".", s=300, label="Str Heroes")
    ax3.scatter(x=scatter_table_agi.hero_id ,y=scatter_table_agi.win_rate, marker="x", s=200, label="Agi Heroes")
    ax3.scatter(x=scatter_table_int.hero_id ,y=scatter_table_int.win_rate, marker="^", s=200, label="Int Heroes")
    # ax3.legend(title_fontsize=100)
    ax3.set_title("Distribution of Win Rates of Heroes by Primary Attribute")
    ax3.set_xlabel("Hero IDs")
    ax3.set_ylabel("Win Rate (%)")
    ax3.legend(prop=dict(size=18), loc='lower right')
    st.pyplot(fig3)

    st.write("Note to self: Add a line later to see on average which attribute is better for the player")

    # +++++++++++++++++++++ #
    # Machine Learning Part #
    # +++++++++++++++++++++ #

    st.title("""
    Next, we'll use 3 regular machine learning techniques to predict whether the player will win or lose the match.
    """)
    st.write("""
    It will take use the data of the player's recent matches, and we'll pick some features that it will
    take into account when predicting the win rate
    """)

    # +++++++++++++++++++++ #
    # Machine Learning Part #
    # +++++++++++++++++++++ #

    # Grab data into variable
    m_cmd = "https://api.opendota.com/api/players/"+str(player_id)+"/matches"
    r_match = requests.get(m_cmd)
    j_match = r_match.json()

    # Take their most recent matches
    match_table = pd.DataFrame.from_dict(j_match)
    st.write("""
    Let's see what are the available data that comes with the API
    """)
    st.subheader("Number of data points: " + str(len(match_table)))
    st.subheader("Features available")
    st.write(match_table.columns.tolist())

    st.write("""
    To make the model simpler and "calculatable", we have to make some assumptions.
    
    After delving into the data, we can see that we do not need all of the datas from the API, particularly:
    - leaver_status
        - Most of the time when there's a leaver, the game is not scored.
    - match_id
        - This is randomly assigned by the DOTA2 client. Does not mean anything in terms of player's skill.
    - version
        - Version of the game.
    - start_time
        - "Start time of the match in seconds elapsed since 1970", according to the API. Funny if you ask me, but we wont be needing this.
    """)

    st.subheader("""
    Assumptions we're making:
    """)

    st.write("""
    1. The game does not change too drastically over time, so the changes between version is ignored.
    2. Player's skill is consistent at any given time, where they are not affected by external factors.
    3. Players plays all roles in the game at the same level.
    4. Player's performance is affected by:
        1. Game Mode - A certain game mode is more serious than others, such as Ranked All Pick, making player focus more.
        2. Lobby type - Same as above, where players might focus more on certain lobbies. E.g.: Tournament Lobby
        3. Party Size - If player plays in a party, they tend to communicate easier in the game, which could affect the outcome of the game.
    5. Scores in terms of kills, deaths, and assists are not weighted, and seen purely as numbers.
    """)

    # ++++++++++++++++++++++++++++ #
    # Machine Learning Calculation #
    # ++++++++++++++++++++++++++++ #

    match_table = match_table.drop(columns=['leaver_status','match_id','version','start_time'])
    match_table['party_size'] = match_table['party_size'].fillna(1)
    match_table.loc[match_table['player_slot'] < 128, 'player_slot'] = 0
    match_table.loc[match_table['player_slot'] > 127, 'player_slot'] = 1
    match_table.loc[match_table['radiant_win'] == False, 'radiant_win'] = 0
    match_table.loc[match_table['radiant_win'] == True, 'radiant_win'] = 1
    match_table['status'] = 0
    match_table.loc[match_table['player_slot'] == match_table['radiant_win'], 'status'] = 1
    match_table['skill'] = match_table['skill'].fillna(0)
    match_table = match_table.drop(columns=["player_slot","radiant_win"])

    # +++ #
    # KNN #
    # +++ #

    from sklearn.model_selection import train_test_split as split
    X = match_table.iloc[:,:-1]
    y = match_table.iloc[:,-1]
    X_train,X_test,y_train,y_test = split(X,y,random_state=0)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(X_train_s, y_train)

    knn_train_score = model.score(X_train_s, y_train)
    knn_test_score = model.score(X_test_s, y_test)
    
    # +++++++++++++++++ #
    # Linear Regression #
    # +++++++++++++++++ #
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    from sklearn.linear_model import LinearRegression
    model2 = LinearRegression()
    model2.fit(X_train_s, y_train)

    lr_train_score = model2.score(X_train_s, y_train)
    lr_test_score = model2.score(X_test_s, y_test)
        
    # +++++++++++++ #
    # Random Forest #
    # +++++++++++++ #
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=200, max_features = 5,  max_depth=None, min_samples_split = 5)
    
    forest.fit(X_train, y_train)
    
    rf_train_score = forest.score(X_train, y_train)
    rf_test_score = forest.score(X_test, y_test)

    # ++++++++++++++++++++++++++++++ #
    # Machine Learning Result Report #
    # ++++++++++++++++++++++++++++++ #

    st.title("""
    Results of Machine Learning Algorithm
    """)

    st.subheader("""
    Notes on each algorithm:
    """)

    st.write("""
    ### KNN
    - Uses MinMaxScaler to transform data
    - Number of neighbors set to: 5
    - Weights option: Uniform

    ### Logistic Regression
    - Uses StandardScaler to transform data
    - Default value for parameters
    
    ### Random Forest
    - Parameters used:
        - Number of trees: 200
        - Minimum sample split = 5

    ### SVM
    - I would put SVM over here, but it takes too long to process live. Can't really pickle the model because eveyrone's data is different.
    - To try: Take a smaller subset of data and pass over SVM
    """)

    ml_report_data = {'Score': ["Train data", "Test data"], 'KNN': [knn_train_score, knn_test_score], 'Linear Regression':[lr_train_score, lr_test_score], 'Random Forest':[rf_train_score, rf_test_score]}
    ml_report = pd.DataFrame(data=ml_report_data)
    ml_report_show = ml_report.set_index(['Score'])
    ml_report_show

    st.subheader("""
    
    """)

    st.write("""
    
    """)



else:
    st.write('')