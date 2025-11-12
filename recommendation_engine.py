import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pickle

class RestaurantRecommender:
    def __init__(self, cleaned_data_path, encoded_data_path, encoder_path):
        self.cleaned_data = pd.read_csv(cleaned_data_path)
        self.encoded_data = pd.read_csv(encoded_data_path)
        
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        
        # Use NearestNeighbors for efficient similarity search
        self.nn_model = self._build_nearest_neighbors()
        
    def _build_nearest_neighbors(self, n_neighbors=50):
        """Build Nearest Neighbors model for efficient similarity search"""
        print("Building Nearest Neighbors model...")
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        nn_model.fit(self.encoded_data)
        return nn_model
    
    def recommend_by_similarity(self, restaurant_index, top_n=5):
        """Recommend restaurants using Nearest Neighbors"""
        if restaurant_index >= len(self.encoded_data):
            return np.array([]), np.array([])
        
        # Reshape for single sample
        query_point = self.encoded_data.iloc[restaurant_index:restaurant_index+1]
        
        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(query_point, n_neighbors=top_n+1)
        
        # Exclude the query point itself
        similar_indices = indices[0][1:top_n+1]
        similarity_scores = 1 - distances[0][1:top_n+1]  # Convert distance to similarity
        
        return similar_indices, similarity_scores
    
    def recommend_by_features(self, city=None, cuisine=None, min_rating=0, max_cost=float('inf'), top_n=5):
        """Recommend restaurants based on user preferences"""
        filtered_data = self.cleaned_data.copy()
        
        # Apply filters
        if city and city != 'All':
            filtered_data = filtered_data[filtered_data['city'].str.lower() == city.lower()]
        if cuisine and cuisine != 'All':
            filtered_data = filtered_data[filtered_data['cuisine'].str.lower().str.contains(cuisine.lower(), na=False)]
        
        filtered_data = filtered_data[
            (filtered_data['rating'] >= min_rating) & 
            (filtered_data['cost'] <= max_cost)
        ]
        
        # If no specific filters or no results, use all data but apply rating and cost filters
        if len(filtered_data) == 0:
            filtered_data = self.cleaned_data[
                (self.cleaned_data['rating'] >= min_rating) & 
                (self.cleaned_data['cost'] <= max_cost)
            ]
        
        # Sort by rating and cost
        recommendations = filtered_data.sort_values(
            ['rating', 'cost'], 
            ascending=[False, True]
        ).head(top_n)
        
        return recommendations
    
    def find_similar_restaurants(self, restaurant_name, top_n=5):
        """Find similar restaurants by name"""
        # Find restaurant index
        restaurant_indices = self.cleaned_data[
            self.cleaned_data['name'].str.contains(restaurant_name, case=False, na=False)
        ].index
        
        if len(restaurant_indices) == 0:
            return pd.DataFrame()
        
        restaurant_index = restaurant_indices[0]
        similar_indices, similarity_scores = self.recommend_by_similarity(restaurant_index, top_n)
        
        if len(similar_indices) == 0:
            return pd.DataFrame()
        
        similar_restaurants = self.cleaned_data.iloc[similar_indices].copy()
        similar_restaurants['similarity_score'] = similarity_scores
        
        return similar_restaurants
    
    def get_restaurant_suggestions(self, query):
        """Get restaurant name suggestions for autocomplete"""
        suggestions = self.cleaned_data[
            self.cleaned_data['name'].str.contains(query, case=False, na=False)
        ]['name'].unique().tolist()
        return suggestions[:10]  # Return top 10 suggestions

def test_recommendation_engine():
    """Test the recommendation engine"""
    print("Loading recommendation engine...")
    recommender = RestaurantRecommender(
        'cleaned_data.csv',
        'encoded_data.csv', 
        'encoder.pkl'
    )
    
    # Test feature-based recommendation
    print("\nTesting feature-based recommendation:")
    recommendations = recommender.recommend_by_features(
        city='Abohar',
        cuisine='North Indian',
        min_rating=4.0,
        max_cost=300,
        top_n=5
    )
    
    print("Top recommendations:")
    for idx, row in recommendations.iterrows():
        print(f"- {row['name']} | Rating: {row['rating']} | Cost: ₹{row['cost']} | Cuisine: {row['cuisine']}")
    
    # Test similarity-based recommendation
    print("\nTesting similarity-based recommendation:")
    if len(recommender.cleaned_data) > 0:
        similar_restaurants = recommender.find_similar_restaurants(
            recommender.cleaned_data.iloc[0]['name'], 
            top_n=3
        )
        if len(similar_restaurants) > 0:
            print("Similar restaurants:")
            for idx, row in similar_restaurants.iterrows():
                print(f"- {row['name']} | Similarity: {row['similarity_score']:.2f}")

if __name__ == "__main__":
    test_recommendation_engine()