import pandas as pd
import numpy as np
import neattext as nt
import neattext.functions as ntf
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load the dataset
df = pd.read_csv("udemy_course_data.csv")

# Display the first few rows
print(df.head())

# Display dataset columns
print(df.columns)
# Apply text cleaning functions to the 'course_title' column
df['cleaned_title'] = df['course_title'].apply(lambda x: ntf.remove_stopwords(x))
df['cleaned_title'] = df['cleaned_title'].apply(lambda x: ntf.remove_special_characters(x))
df['cleaned_title'] = df['cleaned_title'].apply(lambda x: x.lower())  # Convert to lowercase

# Display cleaned data
print(df[['course_title', 'cleaned_title']].head())
df.drop_duplicates(subset="cleaned_title", keep="first", inplace=True)
print("After removing duplicates:", df.shape)
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(df['cleaned_title'])

# Print feature names
print(vectorizer.get_feature_names_out())
cosine_sim = cosine_similarity(vectors)

# Check similarity matrix shape
print(cosine_sim.shape)
def recommend_courses(course_title, top_n=5):
    if course_title not in df['course_title'].values:
        return "Course not found."

    idx = df[df['course_title'] == course_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return df.iloc[recommended_indices][['course_title']]
print(recommend_courses("How To Maximize Your Profits Trading Options"))
