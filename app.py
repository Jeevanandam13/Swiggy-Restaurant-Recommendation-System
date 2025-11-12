import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback

# Page configuration
st.set_page_config(
    page_title="Swiggy Restaurant Recommender",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B00;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .stButton button {
        background-color: #FF6B00;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Debug outputs to surface import/runtime issues when the page appears blank
try:
    st.write("Debug: Streamlit app initialized")
    import sys as _sys
    st.write(f"Python: {_sys.version.split()[0]}")
    try:
        st.write(f"Streamlit: {st.__version__}")
    except Exception:
        pass
except Exception:
    # If Streamlit can't render (e.g., happens before page load), print to console
    import traceback as _traceback
    print("[app.py] Debug write failed:")
    _traceback.print_exc()

def safe_load_recommender():
    """Safely load the recommendation engine with detailed error handling"""
    try:
        # Import inside function to catch import errors
        from recommendation_engine import RestaurantRecommender
        
        # Check if required files exist
        required_files = ['cleaned_data.csv', 'encoded_data.csv', 'encoder.pkl']
        missing_files = []
        for file in required_files:
            try:
                with open(file, 'r') as f:
                    pass
            except:
                missing_files.append(file)
        
        if missing_files:
            st.error(f"Missing required files: {', '.join(missing_files)}")
            st.info("Please run the preprocessing scripts first:")
            st.code("""
python data_cleaning.py
python data_preprocessing.py
            """)
            return None, None
        
        # Load recommender
        recommender = RestaurantRecommender(
            'cleaned_data.csv',
            'encoded_data.csv', 
            'encoder.pkl'
        )
        
        # Load cleaned data
        cleaned_data = pd.read_csv('cleaned_data.csv')
        
        return recommender, cleaned_data
        
    except Exception as e:
        st.error(f"Error loading recommendation system: {str(e)}")
        st.code(f"Detailed error:\n{traceback.format_exc()}")
        return None, None

def display_recommendations(recommendations):
    """Display recommendations in a nice format"""
    if recommendations is None or len(recommendations) == 0:
        st.warning("No recommendations to display")
        return
    
    for idx, row in recommendations.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Safely get values with defaults
                name = row.get('name', 'Unknown Restaurant')
                city = row.get('city', 'Unknown City')
                cuisine = row.get('cuisine', 'Unknown Cuisine')
                address = row.get('address', 'Address not available')
                
                st.markdown(f"### {name}")
                st.write(f"**📍 {city}**")
                st.write(f"**🍽️ Cuisine:** {cuisine}")
                st.write(f"**📝 Address:** {address}")
                
            with col2:
                # Safely get numerical values
                rating = row.get('rating', 0)
                cost = row.get('cost', 0)
                
                rating_color = "green" if rating >= 4.0 else "orange" if rating >= 3.0 else "red"
                st.markdown(f"""
                <div style='text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>
                    <h3 style='color: {rating_color}; margin: 0;'>{rating} ⭐</h3>
                    <p style='margin: 5px 0 0 0;'>₹{cost}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add similarity score if available
            if 'similarity_score' in row:
                st.write(f"**Similarity Score:** {row['similarity_score']:.2f}")
            
            st.markdown("---")

def main():
    try:
        # Header
        st.markdown('<h1 class="main-header">🍔 Swiggy Restaurant Recommender</h1>', unsafe_allow_html=True)
        
        # Debug info
        if st.sidebar.checkbox("Show Debug Info"):
            st.sidebar.write("### Debug Information")
            try:
                import sklearn
                st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
            except:
                st.sidebar.write("scikit-learn: Not available")
            
            try:
                st.sidebar.write(f"pandas: {pd.__version__}")
            except:
                st.sidebar.write("pandas: Not available")
        
        # Load data and recommender
        with st.spinner('Loading recommendation engine...'):
            recommender, cleaned_data = safe_load_recommender()
        
        if recommender is None or cleaned_data is None:
            st.error("""
            **System Setup Required**
            
            Please run the following commands in your terminal to set up the system:
            ```bash
            python data_cleaning.py
            python data_preprocessing.py
            ```
            Then refresh this page.
            """)
            return
        
        # Display basic info
        st.success(f"✅ System loaded successfully! Found {len(cleaned_data)} restaurants.")
        
        # Sidebar for filters
        st.sidebar.title("🔍 Search Filters")
        
        # City filter
        try:
            cities = ['All'] + sorted(cleaned_data['city'].dropna().unique().tolist())
            selected_city = st.sidebar.selectbox("Select City", cities)
        except Exception as e:
            st.sidebar.error(f"Error loading cities: {e}")
            cities = ['All']
            selected_city = 'All'
        
        # Cuisine filter
        try:
            all_cuisines = ['All']
            cuisine_list = []
            for cuisines in cleaned_data['cuisine'].dropna():
                if isinstance(cuisines, str):
                    cuisine_list.extend([c.strip() for c in cuisines.split(',')])
            all_cuisines.extend(sorted(list(set(cuisine_list))))
            selected_cuisine = st.sidebar.selectbox("Select Cuisine", all_cuisines)
        except Exception as e:
            st.sidebar.error(f"Error loading cuisines: {e}")
            all_cuisines = ['All']
            selected_cuisine = 'All'
        
        # Rating filter
        min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.0, 0.1)
        
        # Cost filter
        try:
            max_cost_value = int(cleaned_data['cost'].max()) if not cleaned_data['cost'].isna().all() else 1000
            max_cost = st.sidebar.slider("Maximum Cost (₹)", 0, max(1000, max_cost_value), 500, 50)
        except:
            max_cost = 500
        
        # Number of recommendations
        top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 10)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h2 class="sub-header">🎯 Get Recommendations</h2>', unsafe_allow_html=True)
            
            # Recommendation method selection
            method = st.radio(
                "Choose recommendation method:",
                ["Feature-based Filtering", "Find Similar Restaurants"],
                horizontal=True
            )
            
            if method == "Feature-based Filtering":
                if st.button("Get Recommendations", type="primary", use_container_width=True):
                    with st.spinner('Finding the best restaurants for you...'):
                        try:
                            recommendations = recommender.recommend_by_features(
                                city=None if selected_city == 'All' else selected_city,
                                cuisine=None if selected_cuisine == 'All' else selected_cuisine,
                                min_rating=min_rating,
                                max_cost=max_cost,
                                top_n=top_n
                            )
                            
                            if len(recommendations) > 0:
                                st.success(f"Found {len(recommendations)} recommendations!")
                                display_recommendations(recommendations)
                            else:
                                st.warning("No restaurants found matching your criteria. Try adjusting your filters.")
                        except Exception as e:
                            st.error(f"Error getting recommendations: {e}")
                            st.code(traceback.format_exc())
            
            else:  # Find Similar Restaurants
                try:
                    restaurant_names = ['Select a restaurant'] + sorted(cleaned_data['name'].dropna().unique().tolist())
                    selected_restaurant = st.selectbox("Select a restaurant to find similar ones:", restaurant_names)
                    
                    if selected_restaurant != 'Select a restaurant' and st.button("Find Similar", type="primary", use_container_width=True):
                        with st.spinner(f'Finding restaurants similar to {selected_restaurant}...'):
                            try:
                                similar_restaurants = recommender.find_similar_restaurants(selected_restaurant, top_n)
                                
                                if len(similar_restaurants) > 0:
                                    st.success(f"Found {len(similar_restaurants)} similar restaurants!")
                                    display_recommendations(similar_restaurants)
                                else:
                                    st.warning("No similar restaurants found. Try selecting a different restaurant.")
                            except Exception as e:
                                st.error(f"Error finding similar restaurants: {e}")
                except Exception as e:
                    st.error(f"Error loading restaurant list: {e}")
        
        with col2:
            st.markdown('<h2 class="sub-header">📊 Quick Stats</h2>', unsafe_allow_html=True)
            
            try:
                total_restaurants = len(cleaned_data)
                avg_rating = cleaned_data['rating'].mean()
                avg_cost = cleaned_data['cost'].mean()
                
                st.metric("Total Restaurants", total_restaurants)
                st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
                st.metric("Average Cost", f"₹{avg_cost:.2f}")
                
                # Top rated restaurants
                st.markdown("---")
                st.subheader("🏆 Top Rated")
                top_rated = cleaned_data.nlargest(3, 'rating')[['name', 'rating', 'city']].dropna()
                for idx, row in top_rated.iterrows():
                    st.write(f"**{row['name']}**")
                    st.write(f"Rating: {row['rating']} ⭐ | {row['city']}")
                    st.write("---")
            except Exception as e:
                st.error(f"Error displaying stats: {e}")
    
    except Exception as e:
        st.error(f"Unexpected error in main application: {e}")
        st.code(f"Full error traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as _e:
        # Surface any uncaught exceptions inside Streamlit UI and console
        try:
            st.error(f"Uncaught exception: {_e}")
            st.code(traceback.format_exc())
        except Exception:
            import traceback as _traceback
            print("Uncaught exception in main():")
            _traceback.print_exc()
        # Re-raise so logs/terminals still show the failure
        raise