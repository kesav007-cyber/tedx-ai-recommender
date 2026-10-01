from flask import Flask, request, render_template
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- 1. Engine Initialization ---
print("Initializing Recommendation Engine...")
df = pd.read_csv('tedx_dataset.csv').dropna(subset=['title', 'details']).reset_index(drop=True)

df['year'] = df['posted'].astype(str).str.extract(r'(\d{4})')
df['combined_text'] = df['title'] + " " + df['details']

custom_noise = ['ted', 'talk', 'tedx', 'talks', 'speaker', 'video', 'will', 
                'make', 'say', 'one', 'even', 'way', 'need', 'really', 'much', 
                'dont', 'thats', 'time', 'people', 'world', 'life', 'new']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join([w for w in text.split() if w not in custom_noise])

df['processed_text'] = df['combined_text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['processed_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Extract vocabulary feature names for AI tagging
feature_names = tfidf.get_feature_names_out()

indices = pd.Series(df.index, index=df['title']).drop_duplicates()
all_titles = sorted(df['title'].tolist()) 

# --- 2. Recommendation Logic with AI Tagging ---
def get_recommendations(title):
    if title not in indices:
        return []
        
    idx = indices[title]
    
    # Prevents matrix crashes if two talks have the exact same title
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
        
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    talk_indices = [i[0] for i in sim_scores]
    
    recommendations_data = []
    for i in talk_indices:
        talk = df.iloc[i]
        
        # Extract top 3 highest-scoring TF-IDF words for this talk vector
        row_vector = tfidf_matrix[i]
        sorted_items = sorted(zip(row_vector.indices, row_vector.data), key=lambda x: x[1], reverse=True)
        top_keywords = [feature_names[feat_idx] for feat_idx, score in sorted_items[:3]]
        
        recommendations_data.append({
            'title': talk['title'],
            'speaker': talk['main_speaker'],
            'year': talk['year'],
            'snippet': str(talk['details'])[:120] + "...",
            'tags': top_keywords
        })
    return recommendations_data

# --- 3. Web Routing ---
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_title = ""
    
    if request.method == 'POST':
        selected_title = request.form.get('title')
        if selected_title:
            recommendations = get_recommendations(selected_title)
            
    return render_template('index.html', titles=all_titles, recommendations=recommendations, selected_title=selected_title)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)