from typing import Tuple, Text

import pandas as pd
import numpy as np


URL_MOVIES = 'https://raw.githubusercontent.com/dotnet/mbmlbook/main/src/5.%20Making%20Recommendations/Data/MovieLensForEducation/movies.csv'
URL_RATINGS = 'https://raw.githubusercontent.com/dotnet/mbmlbook/main/src/5.%20Making%20Recommendations/Data/MovieLensForEducation/ratings.csv'


def split_movielens_ratings(df_ratings: pd.DataFrame,
                            train_frac: float = 0.7,
                            random_state: int = 20220926,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    #  "For each person we will use 70% of their likes/dislikes to train on and leave 30% to use for validation"
    idx_train = df_ratings.groupby('user_id').sample(frac=train_frac, replace=False, random_state=random_state).index
    idx_valid = np.setdiff1d(df_ratings.index, idx_train)
    MOVIES_TO_REMOVE = np.setdiff1d(df_ratings.loc[idx_valid, 'movie_id'], df_ratings.loc[idx_train, 'movie_id'])

    df_valid = df_ratings.loc[(idx_valid)]
    idx_valid = df_valid[~df_valid.movie_id.isin(MOVIES_TO_REMOVE)].index

    df_train = df_ratings.loc[idx_train].reset_index().rename(columns={'index': 'orig_index'})
    df_valid = df_ratings.loc[idx_valid].reset_index().rename(columns={'index': 'orig_index'})
    return df_train, df_valid

def load_movielens_dataset(url_movies: Text = URL_MOVIES,
                           url_ratings: Text = URL_RATINGS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_movies = pd.read_csv(url_movies, sep=';', header=None)
    df_movies.columns = ['movie_id', 'movie_name', 'movie_genre']

    df_ratings = pd.read_csv(url_ratings, header=None)
    df_ratings.columns = ['user_id', 'movie_id', 'movie_rating']
    assert not df_ratings.duplicated(['user_id','movie_id']).any()

    idx_user, users = pd.factorize(df_ratings.user_id.values)
    idx_movie, movies = pd.factorize(df_ratings.movie_id.values)
    df_ratings.loc[:, 'idx_user'] = idx_user
    df_ratings.loc[:, 'idx_movie'] = idx_movie

    df_summary = (df_ratings.groupby('movie_id').
                  agg({'movie_id': len, 'movie_rating': np.mean}).
                  rename(columns={'movie_id': 'n_ratings', 'movie_rating': 'avg_rating'}).
                  reset_index()
                 )
    df_movies = df_movies.merge(df_summary, on='movie_id')
    return df_movies, df_ratings, users, movies
