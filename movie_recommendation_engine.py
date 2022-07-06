import time

import pandas as pd
from scipy.sparse import csr_matrix
import joblib
import streamlit as st
import json, requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import keyboard


#importing data
ratings = pd.read_csv('Movie_Recommendation_Engine_Data/ratings.csv')
movies = pd.read_csv('C:\Python Projects\Movie_Recommendation_Engine_Data\movies.csv')

data = ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating')
data.fillna(0, inplace=True)

#number of users who voted
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
#groupby - (unique values)[for the given column].agg('count'-count number of ratings)
#number of movies that were voted
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

#threshold-> number of users who voted a movie should  be more than 10
data = data.loc[no_user_voted[no_user_voted > 10].index,:]
#threshold-> number of votes by a particular user should be more than 50
data = data.loc[:,no_movies_voted[no_movies_voted > 50].index]

#removing sparsity
csr_data = csr_matrix(data.values)
data.reset_index(inplace = True)

#loading K Nearest Neighbours model
knn = joblib.load('movie_recommendation_engine.sav')

#getting recommendations
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 6 #10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = data[data['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = data.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"


def movie_details(recommended_movies):

        list_of_ids = []
        list_of_images = []
        list_of_titles = []
        list_of_ratings = []
        list_of_years = []
        list_of_trailer_desc = []
        list_of_trailer_link = []

        for movie_name in recommended_movies['Title']:

            url_movie = f'https://imdb-api.com/API/SearchMovie/k_oey762go/{movie_name}'
            res = requests.get(url=url_movie)
            soup = BeautifulSoup(res.content, 'html.parser')
            data = json.loads(soup.text)
            best_match = data['results'][0]

            #"""movie_id, movie_poster, movie_title"""

            movie_id = best_match['id']
            movie_image_url = best_match['image']
            movie_title = best_match['title']

            url_rating = f'https://imdb-api.com/API/Ratings/k_oey762go/{movie_id}'
            url_trailer = f'https://imdb-api.com/API/Trailer/k_oey762go/{movie_id}'

            res_rating = requests.get(url=url_rating)
            res_trailer = requests.get(url=url_trailer)

            soup_rating = BeautifulSoup(res_rating.content, 'html.parser')
            soup_trailer = BeautifulSoup(res_trailer.content, 'html.parser')

            data_rating = json.loads(soup_rating.text)
            data_trailer = json.loads(soup_trailer.text)

            #"""movie_rating, release_year, trailer_description, trailer_link"""

            movie_rating = data_rating['imDb']
            release_year = data_rating['year']

            trailer_desc = data_trailer['videoDescription']
            trailer_link = data_trailer['link']

            list_of_ids.append(movie_id)
            list_of_images.append(movie_image_url)
            list_of_titles.append(movie_title)
            list_of_ratings.append(movie_rating)
            list_of_years.append(release_year)
            list_of_trailer_desc.append(trailer_desc)
            list_of_trailer_link.append(trailer_link)

        movie_dict = {'id':list_of_ids, 'title':list_of_titles, 'image_url':list_of_images, 'rating':list_of_ratings, 'release_year':list_of_years, 'trailer_desc':list_of_trailer_desc, 'trailer_link':list_of_trailer_link}
        new_movie_dataframe = pd.DataFrame(movie_dict)

        return new_movie_dataframe



#Web app

#title
st.title("Movie Recommendation System")
st.write('')

col1, col2 = st.columns([10, 1])
with col1:
#text input
    title = st.text_input('Enter the movie name:', placeholder='Movie Name')

st.write('')

if title == '':
    pass
else:
    recommended_movies = get_movie_recommendation(title)

    if isinstance(recommended_movies, pd.DataFrame):
        recommended_movie_details_df = movie_details(recommended_movies)

        no_of_rows = 2
        no_of_columns = 3

        for rows in range(no_of_rows):

            list_of_columns = st.columns(no_of_columns)
            column_iter = iter(list_of_columns)

            for index in range(no_of_columns*rows, no_of_columns*(rows+1)):

                id = recommended_movie_details_df['id'][index]
                image_url = recommended_movie_details_df['image_url'][index]
                title = recommended_movie_details_df['title'][index]
                movie_rating = recommended_movie_details_df['rating'][index]
                release_year = recommended_movie_details_df['release_year'][index]
                trailer_desc = recommended_movie_details_df['trailer_desc'][index]
                trailer_link = recommended_movie_details_df['trailer_link'][index]

                with next(column_iter):
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))

                    st.image(image, caption=title, use_column_width='always', width=5)
                    st.write(f'Release Year : {release_year}')
                    st.write(f'Movie Rating : {movie_rating}')
                    st.write(f'Trailer link : {trailer_link}')
                    expander = st.expander('Trailer Description')
                    expander.write(f'{trailer_desc}')

    else:
        st.write(f'{recommended_movies}')

