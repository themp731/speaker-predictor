from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

if not os.path.exists('../model/bert_embedder'):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    model.save('../model/bert_embedder')

# -------------------------------
# Load BERT embedder and classifier
# -------------------------------
# Load the classifier model (trained on embeddings + structured features)
clf = joblib.load('../model/bert_classifier.joblib')

# Load the BERT-based sentence embedder
embedder = SentenceTransformer('../model/bert_embedder')

# Start the flask app
app = Flask(__name__)


# Home route that handles both the form display (GET) and form submission (POST).
@app.route('/', methods=["GET", "POST"])
def index():
    top_preds=None # Always define as empty from start
    # Get the guess from user input
    if request.method == 'POST':
        # Get data from the form submission
        message = request.form['message']   # Text message input
        
        # Get individual time components from the form
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        ampm = request.form['ampm']

        # Convert 12-hour to 24-hour format
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0

        # Define the time-of-day bucket
        def get_time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
            
        time_of_day = get_time_of_day(hour)

        # Checkboxes return a key only if selected, so we check if the key is present in form
        has_link = 'has_link' in request.form                 # Returns True if checkbox checked
        has_image = 'has_image' in request.form
        edited = 'edited' in request.form

        # ----- 2. Encode message using BERT -----
        embedding = embedder.encode([message])  # Returns 2D array: [[...]]
        
        # ----- 3. Convert structured features into same shape -----
        structured_features = np.array([[has_link, has_image, edited]])

        # ----- 4. Concatenate embeddings + structured features -----
        full_input = np.hstack((embedding, structured_features))  # Shape: (1, 387)

        # ----- 5. Predict speaker probabilities -----
        probs = clf.predict_proba(full_input)[0]
        classes = clf.classes_

        # Get top 3 predictions
        top_preds = sorted(zip(classes, probs), key=lambda x: -x[1])[:3]
        top_preds = [(speaker, round(prob * 100, 2)) for speaker, prob in top_preds]

    return render_template('index.html', top_preds=top_preds)

# Entry point to start the Flask development server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render uses PORT env variable
    app.run(host='0.0.0.0', port=10000)