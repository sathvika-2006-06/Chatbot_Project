import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
data = pd.read_csv("chatbot_dataset.csv")
questions = data['question'].tolist()
answers = data['answer'].tolist()
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
questions_clean = [clean_text(q) for q in questions]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions_clean)
def chatbot_response(user_input):
    user_input = clean_text(user_input)
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    best_match = similarity.argmax()
    score = similarity[0][best_match]
    if score < 0.3:
        return "Sorry, I am not sure how to answer that."
    return answers[best_match]
print("="*50)
print("        SMARTBOT - RULE BASED CHATBOT")
print("="*50)
print("Type 'exit' to end conversation")
print()
while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        print("Bot : Goodbye! Have a great day.")
        break
    response = chatbot_response(user_input)
    print("Bot :", response)