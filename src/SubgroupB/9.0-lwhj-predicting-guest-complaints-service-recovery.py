
!pip install pandas matplotlib wordcloud scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation

def load_data(filename):
    """
    Load review data from a CSV file, create a combined 'full_text' field, and label complaints.
    Complaints are defined as ratings <= 2.
    """
    df = pd.read_csv(filename)
    df['full_text'] = df['title'].fillna('') + ' ; ' + df['review_text'].fillna('')
    df['label'] = df['rating'].apply(lambda x: 'complaint' if x <= 2 else 'non-complaint')
    return df

def train_complaint_model(X_train, y_train):
    """
    Build and train a text classification pipeline using TF-IDF and SVM.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',         # Remove common English stop words
            ngram_range=(1, 2),           # Use unigrams and bigrams
            min_df=3,                     # Ignore rare terms (appear in <3 docs)
            max_df=0.9                    # Ignore very frequent terms (appear in >90% of docs)
        )),
        ('clf', SVC(kernel='linear', C=1.0, class_weight='balanced'))  # SVM with balanced class weights
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the model using test data and print a classification report.
    """
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def predict_new_reviews(pipeline, reviews):
    """
    Predict labels (complaint or non-complaint) for a list of new reviews.
    """
    predictions = pipeline.predict(reviews)
    print("\nPredictions on New Reviews:")
    for review, label in zip(reviews, predictions):
        print(f"Review: {review} --> Prediction: {label}")

def perform_topic_modeling(df, n_topics=5):
    """
    Perform LDA topic modeling on complaint reviews to identify recurring issues.
    Returns:
        - A DataFrame of top phrases for each topic
        - The fitted TF-IDF vectorizer
    """
    # Filter only complaints (ratings < 3)
    complaints_df = df[df['rating'] < 3].copy()
    complaints_df['full_review'] = complaints_df['title'].fillna('') + ' ' + complaints_df['review_text'].fillna('')
    
    # Vectorize using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf.fit_transform(complaints_df['full_review'])
    feature_names = tfidf.get_feature_names_out()
    
    # Apply LDA to identify latent topics
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    # Extract top terms for each topic
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[-8:][::-1]]
        topics.append({'Topic': f'Topic {idx + 1}', 'Top Phrases': ', '.join(top_features)})

    labeled_df = pd.DataFrame(topics)
    return labeled_df, tfidf

def plot_wordcloud(tfidf):
    """
    Generate a word cloud visualization of the vocabulary extracted by the TF-IDF vectorizer.
    """
    words = ', '.join(tfidf.get_feature_names_out())
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=set(['universal', 'studio'])  # Remove irrelevant branding words
    ).generate(words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Complaint Review Word Cloud")
    plt.show()

if __name__ == "__main__":
    # Load and prepare data
    filename = "/app/data/External/universal_studio_branches.csv.zip"
    df = load_data(filename)

    # Split data into training and test sets (stratified to preserve label balance)
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_text'], df['label'], 
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # Train and evaluate the model
    model = train_complaint_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Perform topic modeling on complaint reviews
    topics_df, tfidf = perform_topic_modeling(df)
    print("\nComplaint Topics Identified:")
    print(topics_df)

    # Generate a word cloud from the TF-IDF vocabulary
    plot_wordcloud(tfidf)
