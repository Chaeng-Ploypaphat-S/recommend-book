import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import pickle

saved_model_fname = "model/finalized_model.sav"
data_fname = "data/ratings.csv"
item_fname = "data/movies_final.csv"
weight = 10

try:
    RATINGS_DF_CACHE = pd.read_csv(data_fname)
except FileNotFoundError:
    print(f"Error: File not found at {data_fname}")
    RATINGS_DF_CACHE = pd.DataFrame()
    
    
try:
    MOVIES_DF_CACHE = pd.read_csv(item_fname)
except FileNotFoundError:
    print(f"Error: File not found at {data_fname}")
    MOVIES_DF_CACHE = pd.DataFrame()

def model_train():
    ratings_df = RATINGS_DF_CACHE
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    
    # create a sparse matrix of all the users/repos
    rating_matrix = coo_matrix(
        (
            ratings_df["rating"].astype(np.float32),
            (
                ratings_df["movieId"].cat.codes.copy(),
                ratings_df["userId"].cat.codes.copy()
            )
        )
    )
    als_model = AlternatingLeastSquares(
        factors=50, regularization=0.01, dtype=np.float64, iterations=50
    )
    als_model.fit(weight * rating_matrix)
    pickle.dump(als_model, open(saved_model_fname, "wb"))
    return als_model

def calculate_item_based(item_id, items):
    loaded_model = pickle.load(open(saved_model_fname, 'rb'))
    recs = loaded_model.similar_items(itemid=int(item_id), N=11)
    return [str(items[r]) for r in recs[0]]

import numpy as np # <-- You may need to add this import at the top of your recommender.py file

def item_based_recommendation(item_id):
    ratings_df = RATINGS_DF_CACHE
    ratings_df["userId"] = ratings_df["userId"].astype("category")
    ratings_df["movieId"] = ratings_df["movieId"].astype("category")
    movies_df = MOVIES_DF_CACHE
    items = dict(enumerate(ratings_df["movieId"].cat.categories))
    
    try:
        parsed_id = ratings_df["movieId"].cat.categories.get_loc(int(item_id))
        result = calculate_item_based(parsed_id, items)
    except KeyError:
        result = []
        
    result = [int(x) for x in result if x != item_id]

    record_df = movies_df[movies_df["movieId"].isin(result)]
    cleaned_df = record_df.replace([float('inf'), float('-inf')], np.nan)
    final_df = cleaned_df.replace([np.nan], [None])
    
    return final_df.to_dict("records")

if __name__ == "__main__":
    model = model_train()