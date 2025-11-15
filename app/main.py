from typing import List, Optional
from fastapi import FastAPI, Query
from resolver import random_items, random_genres_items

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Test Home API"}

@app.get("/all/")
async def all_movies():
    return {"message": random_items()}

@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    return {"message": random_genres_items(genre)}

@app.get("/user-based/")
async def user_based(params: Optional[List[str]] = Query(None)):
    return {"message": "Test user-based request"}

@app.get("/item-based/{item_id}")
async def item_based(item_id: str):
    return {"message": f"item based: {item_id}"}