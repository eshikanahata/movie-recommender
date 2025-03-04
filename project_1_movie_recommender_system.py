"""
project_1_movie_recommender_system.ipynb

"""

import numpy as np
import pandas as pd

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# need to merge both dataframes by movie id
movies = pd.merge(movies, credits, left_on='id', right_on='movie_id')

# picking which columns to keep/discard
# since the recommendation system is content based we need to see which columns will aid in generating tags for each movie

movies = movies[["genres", "movie_id", "keywords", "title_x", "overview", "cast", "crew"]]

# need to convert this into a dataframe containing 3 columns: movie id, title, tags
# we will make tags by combining remaining columns

movies.dropna(inplace=True) # there are only 3 rows with missing values in overview column

import ast
def convert(column):
  result = []
  for i in ast.literal_eval(column):
    result.append(i["name"])
  return result

movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)

def convert2(column):
  result = []
  for i in ast.literal_eval(column)[0:3]:
    result.append(i["character"])
  return result

movies['cast']=movies['cast'].apply(convert2)

def convert3(column):
  result = []
  for i in ast.literal_eval(column):
    if i["job"]=="Director":
      result.append(i["name"])
      break
  return result

movies['crew']=movies['crew'].apply(convert3)

movies["overview"]=movies["overview"].apply(lambda x: x.split())

# we need to remove space within the same entries as they will get split into seperate tags otherwise (which we dont want)
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

# create a new column for all tags
movies["tags"] = movies["genres"] + movies["keywords"] + movies["overview"] + movies["cast"] + movies["crew"]

movies_final = movies[["movie_id", "title_x","tags"]]

movies_final['tags']=movies_final['tags'].apply(lambda x: " ".join(x))

movies_final['tags']=movies_final['tags'].apply(lambda x: x.lower())

# need to apply stemming to club similar words together
import nltk
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

def stem(text):
  result = []
  for i in text.split():
    result.append(ps.stem(i))
  return " ".join(result)

movies_final["tags"]=movies_final["tags"].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = "english")

vectors = cv.fit_transform(movies_final["tags"]).toarray()

# using cosine similarity to measure proximity of tags
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors).astype(np.float16)

def recommend(movie):
  index = movies_final[movies_final["title_x"]==movie].index[0]
  movies_list = sorted(list(enumerate(similarity[index])),reverse=True, key=lambda x: x[1])[1:6]
  for i in movies_list:
    print(movies_final.iloc[i[0]].title_x)



# next steps: include relevant numerical columns which were dropped in the beginning and use numerical to categorical feature transformations
# to see how model performs with the help of this new information
# try a different word encoding technique and see how recommendation system improves

