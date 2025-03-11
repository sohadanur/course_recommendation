from flask import Flask, request, render_template
import pandas as pd
import neattext.functions as ntf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

app = Flask(__name__)

# Load dataset and preprocess
df = pd.read_csv("udemy_course_data.csv")
df['cleaned_title'] = df['course_title'].apply(lambda x: ntf.remove_stopwords(x))
df['cleaned_title'] = df['cleaned_title'].apply(lambda x: ntf.remove_special_characters(x))
df['cleaned_title'] = df['cleaned_title'].apply(lambda x: x.lower())

# Define recommendation function
def recommend_courses(course_title, top_n=5):
    closest_match, score = process.extractOne(course_title, df['course_title'].tolist())
    
    if score < 50:
        return [{"course_title": "Course not found.", "Similarity_Score": "", "price": "", "num_subscribers": "", "url": ""}]
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_title'])
    
    input_vector = tfidf_vectorizer.transform([course_title])
    
    cosine_sim = cosine_similarity(input_vector, tfidf_matrix)
    
    similar_indices = cosine_sim.argsort()[0][-top_n-1:-1][::-1]
    
    recommendations = []
    for idx in similar_indices:
        recommendations.append({
            "course_title": df.iloc[idx]['course_title'],
            "Similarity_Score": cosine_sim[0][idx],
            "price": df.iloc[idx]['price'],
            "num_subscribers": df.iloc[idx]['num_subscribers'],
            "url": df.iloc[idx]['url']
        })
    
    return recommendations

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        course = request.form["course"]
        recommendations = recommend_courses(course)
    
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)