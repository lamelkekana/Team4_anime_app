## Anime Recommendation System

### 📌 Overview

This Streamlit application provides anime recommendations using three different filtering techniques:

**Content-Based Filtering** - Recommends anime based on similarity in attributes such as genre, synopsis, and other metadata.

**Collaborative Filtering** - Utilizes user-item interactions to generate recommendations based on past behavior.

**Hybrid Filtering** - Combines content-based and collaborative filtering to improve recommendation accuracy.

### ⭐ Features

Search for an anime and get recommendations.

User-based collaborative filtering for personalized suggestions.

Hybrid approach to enhance recommendation quality.

Interactive UI built with Streamlit for seamless experience.

### 🛠️ Dependencies

Python 3

Streamlit

Pandas

Numpy

Scikit-learn

Surprise (for collaborative filtering)

### ⚙️ Installation

Clone the repository:
```
git clone https://github.com/lamelkekana/Team4_Streamlit_Anime_Recommender_System.git
cd <cloned-directory>
```

**Create the new evironment - you only need to do this once**

```
 # create the conda environment
conda create --name <env>
```

**This is how you activate the virtual environment in a terminal and install the project dependencies**

```
# activate the virtual environment
conda activate <env>
# install the pip package
conda install pip
# install the requirements for this project, requirements.txt is provided in the the repo
pip install -r requirements.txt ```
```

Alternatively, You can use Python’s built-in venv module to create a virtual environment

**Create the new evironment**

```
# create enironment
python -m venv <env_name>
# activate environment
obesity_env\Scripts\activate

```
**If pip is not installed in your environment, you can install it by running:**

```
python -m ensurepip --upgrade
```
**Install the Project Dependencies**
```
pip install -r requirements.txt

```

**Usage**

Run the Streamlit app:

```
# Make sure you are in the repo directory and that the newly created environment is activated
streamlit run app.py
```

### 📝 Dataset

The application utilizes a dataset containing:

user_id: Unique identifier for users.

anime_id: Unique identifier for anime.

rating: User ratings for different anime.

anime metadata: Information like genre, type and other attributes.


### 🎯 Recommendation Techniques

1. **Content-Based Filtering**

Uses TF-IDF and cosine similarity to recommend anime similar to a given one.

2. **Collaborative Filtering**

Implements user-based collaborative filtering using matrix factorization(SVD).

3. **Hybrid Filtering**

Combines both techniques for more accurate and diverse recommendations.


### Contributors

- Lindelwe Mathonsi
- Lesego Malatsi
- Lamel Kekana
- Vrishti Singh
- Laudicia Ramasenya




