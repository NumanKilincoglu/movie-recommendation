from django.http import JsonResponse
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from pathlib import Path 

def load_movies():
    file_path = Path(settings.BASE_DIR) / 'movie.json'
    with open(file_path, 'r') as file:
        return json.load(file)

def create_features(movie, title_weight=1):
    title_features = (movie['title'].lower() + ' ') * title_weight

    director = ' '.join(person['name'].lower() for person in movie['director'])
    actors = ' '.join(actor['name'].lower() for actor in movie['actors'])
    writers = ' '.join(writer['name'].lower() for writer in movie['writer'])
    genres = ' '.join(genre['name'].lower() for genre in movie['genres'])

    other_features = f"{movie['description'].lower()} {director} {actors} {writers} {genres}"

    features = title_features + other_features

    return features

def recommend_movies(request, title):
    movies = load_movies()
    titles = [movie['title'].lower() for movie in movies]

    features = [create_features(movie, title_weight=3) for movie in movies]

    title = title.lower()

    if title not in titles:
        return JsonResponse({'error': 'Movie not found'}, status=404)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(features)

    index = titles.index(title)
    cosine_sim = cosine_similarity(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    indices = cosine_sim.argsort()[-11:-1][::-1]

    recommended_movies = [{'title': titles[i].title(), 'similarity_score': format(cosine_sim[i], '.5f')} for i in indices]
    return JsonResponse({'recommended_movies': recommended_movies})