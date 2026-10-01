import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)

# 1. Load & Clean Data
df = pd.read_csv('tedx_dataset.csv')
df = df.dropna(subset=['title', 'details']).reset_index(drop=True)
df['combined_text'] = df['title'] + " " + df['details']

# Keep 'title' this time so we can search for it!
df_clean = df[['title', 'main_speaker', 'combined_text']].copy()

# 2. NLP Preprocessing (Aggressive Filtering)
stop_words = set(stopwords.words('english'))
# Added the conversational noise words spotted in your WordCloud output
custom_noise = ['ted', 'talk', 'tedx', 'talks', 'speaker', 'video', 'will', 
                'make', 'say', 'one', 'even', 'way', 'need', 'really', 'much', 
                'dont', 'thats', 'time', 'people', 'world', 'life', 'new']
stop_words.update(custom_noise)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

print("Cleaning text corpus...")
df_clean['processed_text'] = df_clean['combined_text'].apply(clean_text)

# 3. TF-IDF Vectorization
print("Vectorizing text (converting words to numbers)...")
# Limit to 5000 features to keep the matrix manageable and memory-efficient
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df_clean['processed_text'])

# 4. Compute Cosine Similarity
print("Computing similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 5. The Recommendation Function
# Create a reverse mapping of indices and talk titles
indices = pd.Series(df_clean.index, index=df_clean['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Check if the title exists in our dataset
    if title not in indices:
        return f"Error: '{title}' not found in the dataset."
        
    # Get the index of the talk that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all talks with that talk
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the talks based on the similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top 5 most similar talks (ignoring the first one, which is itself)
    sim_scores = sim_scores[1:6]

    # Get the talk indices
    talk_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar talks
    return df_clean['title'].iloc[talk_indices]

# 6. Test the Engine
print("\n--- Recommendation Test ---")
# Pick a random title from your dataset to test. 
# We'll use the first title in the dataframe as a guaranteed test case.
sample_title = df_clean['title'].iloc[0] 
print(f"If you liked: '{sample_title}'")
print("\nYou might also like:")
print(get_recommendations(sample_title))