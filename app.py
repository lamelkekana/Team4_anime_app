import streamlit as st
import pandas as pd
import numpy as np
import requests
import dill
import pickle
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_svd():
    with open("svd_best_model.pkl", "rb") as mf:
        svd_model = pickle.load(mf)

    return svd_model


# Load the pickled files (cached version)
svd_model = load_svd()

@st.cache_data
def load_train_data():
    return pd.read_csv("train.csv")

@st.cache_data
def load_anime_data():
    return pd.read_csv("df_anime_cleaned.csv")

@st.cache_data
def load_tfidf_data():
    return pd.read_csv("tfidf_df_reduced.csv")

# Load data
df_train = load_train_data()
df_anime = load_anime_data()
tfidf_df = load_tfidf_data()


def get_recommendations_content(input_anime, df_anime=df_anime, reduced_tfidf_dfsvd=tfidf_df, top_n=10):
    # Check if the input anime exists in the dataset
    if input_anime not in df_anime['name'].values:
        return None

    # Get the index of the input anime
    title_index = df_anime[df_anime['name'] == input_anime].index[0]

    # Extract the vector representation of the input anime
    input_svd_vector = reduced_tfidf_dfsvd.iloc[title_index, :-1].to_numpy().reshape(1, -1)

    # Calculate cosine similarities between the input and all other animes
    similarities = cosine_similarity(input_svd_vector, reduced_tfidf_dfsvd.iloc[:, :-1]).flatten()

    # Get indices of the top N most similar animes, excluding the input itself
    similar_indices = similarities.argsort()[::-1][1:top_n + 1]

    # Return the names of the most similar animes
    return list(df_anime.iloc[similar_indices]['name'])


# Get unique anime IDs
anime_ids = df_anime['anime_id'].unique()


# Collaborative filtering recommendations

# Get top N popular anime (based on average rating or number of interactions)
top_popular_anime = df_anime.sort_values(by='ave_rating',ascending=False).head(10)
top_popular_anime = top_popular_anime['anime_id'].tolist()


@st.cache_data
def get_recommendations_collaborative(user_id, n=10):
    # Check if user exists in the training data,if not return highest rated animes(cold start feature)
    if user_id not in df_train['user_id'].unique():
        popular_anime = df_anime[df_anime['anime_id'].isin(top_popular_anime)][['name','ave_rating']]
        # Sort by ave_rating in descending order
        popular_anime = popular_anime.sort_values(by='ave_rating', ascending=False)
        return list(zip(popular_anime['name'], popular_anime['ave_rating']))

    # Get anime already rated by user
    rated_anime = set(df_train[df_train['user_id'] == user_id]['anime_id'])

    # Predict ratings for unseen anime
    predictions = [svd_model.predict(user_id, anime_id) for anime_id in anime_ids if anime_id not in rated_anime]

    # Sort by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations (anime_id and predicted rating)
    top_n = [(pred.iid, pred.est) for pred in predictions[:n]]

    # Fetch anime names and pair them with predicted ratings
    recommended_anime = [(df_anime[df_anime['anime_id'] == anime_id]['name'].values[0], rating) for anime_id, rating in top_n]

    return recommended_anime

def get_hybrid_recommendations(user_id, anime_name, n=10, weight_content=0.5, weight_collab=0.5):
    # get content based filtered animes
    content_recs = get_recommendations_content(anime_name)
    
    if isinstance(content_recs, str):  # If anime not found
        content_recs = []
    
    # get collaborative based filtered animes
    collab_recs = get_recommendations_collaborative(user_id, n)

    # Combine recommendations
    combined_recs = {}
    
    # Assign weights and normalize scores
    for i, anime in enumerate(content_recs):
        combined_recs[anime] = combined_recs.get(anime, 0) + (weight_content * (n - i))

    for i, anime in enumerate(collab_recs):
        combined_recs[anime] = combined_recs.get(anime, 0) + (weight_collab * (n - i))

    # Sort by weighted score and return top N
    final_recommendations = sorted(combined_recs, key=combined_recs.get, reverse=True)[:n]

    return final_recommendations


# Main function of the application
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(" ", ["Model application", "About"])
    # Page content according to the sidebar
    st.markdown(
        """
        <style>
            /* Set background color to black */
            .stApp {
                background-color: black !important;
            }

            /* Set text color to white */
            .stMarkdown, h1, h2, h3, h4, h5, h6, p, label {
                color: white !important;
            }

            /* Style buttons */
            .stButton>button {
                background-color: white !important;
                color: black !important;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
                border: 2px solid white;
            }

            /* Style input boxes */
            .stTextInput>div>div>input {
                background-color: #222 !important;
                color: white !important;
            }

            /* Style sidebar */
            [data-testid="stSidebar"] {
                background-color: #222 !important;
            }
        """, 
        unsafe_allow_html=True
    )
    # Add Empty Space to Push Contributors to the Bottom
    for _ in range(20):  # Adjust the range to push text further down
        st.sidebar.write("")
    # Add Contributor Names Below the Sidebar Menu
    st.sidebar.markdown("---")  # Horizontal Line for Separation

    if page == "About":
        st.title("Recommendation Techniques")
    
        st.subheader("Content-Based Filtering")
        st.write(
        """
        This method recommends anime based on their characteristics by analysing 
        the features of previously liked anime and suggesting similar ones. 
        It evaluates similarity based on attributes such as genre, type, author and sypnosis,
        using a similarity matrix to quantify the relationships between anime. 
        The system then recommends content with the highest similarity scores, 
        typically favouring those with shared characteristics, such as genre or author.
        """
        )
        st.write("> Example: If a user enjoys action and adventure anime, the system suggests other anime with similar genres.")

        st.subheader("Collaborative Filtering")
        st.write(
        """
        This technique recommends anime based on user interactions and preferences, 
        assuming that individuals with similar tastes will enjoy the same content. 
        It leverages the Principle of User Similarity, identifying patterns across users and categories. 
        A utility matrix captures user ratings, and the system recommends anime with high similarity scores, 
        indicating a strong likelihood of user preference.
        """
        )
        st.write("> Example: If two users highly rate the same anime, and one watches a new anime, the system suggests it to the other.")
        st.write("- **Model Used : Singular Value Decomposition (SVD)**")
        st.write(
        """
        >> Singular Value Decomposition (SVD) is a technique that simplifies complex data by breaking a 
        big matrix into smaller pieces. In anime recommendations, it helps the system figure out relationships 
        between users and the anime they like, making it easier to predict what someone might enjoy watching next
.
        """
        )

        st.subheader("Hybrid Filtering")
        st.write(
        """
        A combination of content-based and collaborative filtering to improve recommendation accuracy,  
        also known as a hybrid content-collaborative filtering system. 
        This approach overcomes the limitations of each method when used alone.
        """
        )
        st.write("> Example: The system considers both a user’s previous interactions (collaborative filtering) and anime attributes (content-based filtering) to generate recommendations.")
        st.markdown(
        """
        <style>
            .naruto-image {
                position: fixed;
                bottom: 10px;
                right: 30px;
                width: 600px; /* Adjust size as needed */
                opacity: 0.25;
            }
        </style>
        <img src="https://wallpapers.com/images/high/naruto-rasengan-power-up-hidp1eykd6lix9vh.png" class="naruto-image">
        """, 
        unsafe_allow_html=True
    )

    

    # Streamlit UI
    elif page == "Model application":
        st.title("🎥 Anime Recommender System")
        # st.image("images/20 Best Anime Characters-wallpaper.jpg", use_container_width=200)
        st.write("Find the best anime based on your preferences!")
        st.markdown(
        """
        <style>

            /* Add Luffy image at the bottom right */
            .luffy-image {
                position: fixed;
                bottom: 10px;
                right: 10px;
                width: 600px; /* Adjust size as needed */
                opacity: 0.3; /* Slight transparency for blending */
            }
            .image-with-opacity {
            opacity: 0.6; /* Adjust opacity value here */
            }
            
        </style>
        <img src="https://cdn.pixabay.com/photo/2021/10/04/18/21/goku-6680776_1280.jpg" class="luffy-image">
        <img src="https://static1.thegamerimages.com/wordpress/wp-content/uploads/2024/12/updated-by-umair-malik-on-june-12-2024-with-new-cards-introduced-every-month-the-meta-is-constantly-evolving-and-new-cards-find-their-way-into-multiple-decks-especially-these-ones-we-ve-updat-2024-12-29t190027-962.jpg" class="image-with-opacity" width="500">
        """, 
        unsafe_allow_html=True
    )
        

        # Recommendation type selection
        recommendation_type = st.radio("Select Recommendation Type", ["Content-based", "Collaborative-based", "Hybrid"])


        # Extract the unique anime names from the dataset
        anime_list = df_anime["name"].unique()
        unique_userID = df_train['user_id']

        if recommendation_type == "Content-based":
            # User Input (Search method for the anime movie/show)
            st.subheader ("Search or select your favourite anime")
            # User Input
            selected_anime = st.selectbox("🔍 Select an Anime:", anime_list)

            # Apply custom CSS
            st.markdown("""
            <style>
                div.stButton > button {
                    background-color: black !important;
                    color: white !important;
                    font-weight: bold;
                }
            </style>
            """, unsafe_allow_html=True)
            
            
            if st.button("Get Recommendations 🎬"):
                recommendations = get_recommendations_content(selected_anime)
                
                if recommendations is None:
                    st.write("❌ Anime not found.")
                else:
                    st.write("### 🎯 Recommended Anime for You:")
                    for i, anime in enumerate(recommendations, 1):
                        st.write(f"{i}. {anime} 🎥")

        elif recommendation_type == 'Collaborative-based':
            # User Input (Search method for the anime movie/show)
            st.subheader ("Search UserID")
            selected_user = st.number_input("🔍 Enter userID:", step=1, format="%d")
            
            # Apply custom CSS
            st.markdown("""
            <style>
                div.stButton > button {
                    background-color: black !important;
                    color: white !important;
                    font-weight: bold;
                }
            </style>
            """, unsafe_allow_html=True)

            if st.button("Get Recommendations 🎬"):
                collab_recommendations = get_recommendations_collaborative(selected_user)

                if collab_recommendations == f"User ID {selected_user} not found. Please enter a valid user ID.":
                    st.write("❌ User not found or no recommendations available.")
                    st.stop()  # Stop execution and allow the user to try again
                else:
                    st.write("### 🎯 Recommended Anime for You:")
                    for i, (anime_name, rating) in enumerate(collab_recommendations, 1):
                        st.write(f"{i}. {anime_name} 🎥")

        elif recommendation_type == 'Hybrid':
            # User Input (Search method for the anime movie/show)
            st.subheader ("Search or select your favourite anime")
            # User Input
            selected_anime = st.selectbox("🔍 Select an Anime:", anime_list)
            st.subheader ("Search UserID")
            selected_user = st.number_input("🔍 Enter userID:", step=1, format="%d")
            # Apply custom CSS
            st.markdown("""
            <style>
                div.stButton > button {
                    background-color: black !important;
                    color: white !important;
                    font-weight: bold;
                }
            </style>
            """, unsafe_allow_html=True)

            if st.button("Get Recommendations 🎬"):
                hyb_recommendations = get_hybrid_recommendations(int(selected_user), selected_anime)
                
                if hyb_recommendations is None:
                    st.write("❌ Anime not found.")
                else:
                    st.write("### 🎯 Recommended Anime for You:")
                    for i, anime in enumerate(hyb_recommendations, 1):
                        # Extract the first element if anime is a tuple
                        anime_name = anime[0] if isinstance(anime, tuple) else anime
                        st.write(f"{i}. {anime_name} 🎥")


if __name__ == "__main__":
    main()