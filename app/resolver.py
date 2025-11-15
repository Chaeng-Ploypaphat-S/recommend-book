import pandas as pd

item_fname = 'data/movies_final.csv'

try:
    MOVIES_DF_CACHE = pd.read_csv(item_fname)
    MOVIES_DF_CACHE = MOVIES_DF_CACHE.fillna('')
except FileNotFoundError:
    print(f"Error: File not found at {item_fname}")
    MOVIES_DF_CACHE = pd.DataFrame()

def random_items():
    return MOVIES_DF_CACHE.sample(n=10).to_dict("records")

def random_genres_items(genre: str):
    genre_df = MOVIES_DF_CACHE[MOVIES_DF_CACHE['genres'].apply(lambda x: genre.lower() in x.lower())]
    return genre_df.sample(n=5).to_dict("records")