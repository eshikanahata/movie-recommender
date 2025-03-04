import streamlit as st
import pickle as pk
import pandas as pd

def recommend(movie):
  index = movies[movies["title_x"]==movie].index[0]
  movies_list = sorted(list(enumerate(similarity[index])),reverse=True, key=lambda x: x[1])[1:6]
  recommended_movies= []
  for i in movies_list:
    recommended_movies.append(movies.iloc[i[0]].title_x)
    
  return recommended_movies
  

movies_dict = pk.load(open("movie_dict.pkl", "rb"))  
movies= pd.DataFrame(movies_dict)
similarity  =  pk.load(open("similarity.pkl", "rb")) 

st.title("Movie Recommender System")

selected_movie_name = st.selectbox("Select a movie", movies["title_x"].values)

if st.button("Recommend"):
    recommendations  = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)

