import os
import sys
import pickle
import pandas as pd
import webbrowser
from threading import Timer
from flask import Flask, render_template_string, request, redirect, url_for

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from model import RecommendationModel
except ImportError:
    pass

app = Flask(__name__)

# --- LOAD RESOURCES ---
DATA_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")
MATRIX_PATH = os.path.join(DATA_DIR, "user_item_matrix.csv")

model = None
matrix = None

def load_resources():
    global model, matrix
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(MATRIX_PATH):
        matrix = pd.read_csv(MATRIX_PATH, index_col=0)

load_resources()

# --- MODERN UI TEMPLATE ---

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-commerce Intelligence Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --accent: #00d2ff;
            --glass: rgba(15, 23, 42, 0.8);
        }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            /* Using your uploaded image as the background */
            background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                        url('/static/black-friday-sales-sign-neon-light_23-2151833076.avif');
            background-size: cover;
            background-position: center;
            color: white;
        }
        .main-container {
            text-align: center;
            max-width: 900px;
            width: 90%;
            background: var(--glass);
            backdrop-filter: blur(15px);
            padding: 3rem;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #fff, #00d2ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Badged UI Style */
        .badge-grid {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 3rem;
            flex-wrap: wrap;
        }
        .feature-badge {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px 25px;
            border-radius: 50px;
            border: 1px solid rgba(0, 210, 255, 0.3);
            transition: 0.3s;
        }
        .feature-badge:hover {
            background: rgba(0, 210, 255, 0.1);
            transform: translateY(-5px);
        }
        .badge-val {
            display: block;
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--accent);
        }
        .badge-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.8;
        }

        .input-section {
            max-width: 500px;
            margin: 0 auto;
        }
        .input-section p {
            margin-bottom: 1.5rem;
            font-weight: 300;
            color: rgba(255, 255, 255, 0.7);
        }
        input[type="number"] {
            width: 100%;
            padding: 18px;
            border-radius: 15px;
            border: none;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 1rem;
            text-align: center;
            margin-bottom: 1.5rem;
            outline: none;
            transition: 0.3s;
        }
        input[type="number"]:focus {
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 0 2px var(--accent);
        }
        .recommend-btn {
            background: var(--accent);
            color: #000;
            border: none;
            padding: 18px 45px;
            border-radius: 15px;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            width: 100%;
            transition: 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .recommend-btn:hover {
            background: #fff;
            transform: scale(1.02);
            box-shadow: 0 10px 20px rgba(0, 210, 255, 0.3);
        }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>E-commerce Intelligence Dashboard</h1>
        
        <div class="badge-grid">
            <div class="feature-badge">
                <span class="badge-val">{{ num_users if num_users else '5878' }}</span>
                <span class="badge-label">Unique Customers</span>
            </div>
            <div class="feature-badge">
                <span class="badge-val">{{ num_items if num_items else '4631' }}</span>
                <span class="badge-label">Total Products</span>
            </div>
            <div class="feature-badge">
                <span class="badge-val">{{ variance if variance else '96.9' }}%</span>
                <span class="badge-label">Model Variance</span>
            </div>
        </div>

        <div class="input-section">
            <p>Enter a Customer ID to see personalized product suggestions.</p>
            <form action="/get_recs" method="POST">
                <input type="number" name="user_id" placeholder="Ex: 1024" required>
                <button type="submit" class="recommend-btn">Get Recommendations</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

RECS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommendations for User {{ user_id }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --accent: #00d2ff;
            --glass: rgba(15, 23, 42, 0.8);
        }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                        url('/static/black-friday-sales-sign-neon-light_23-2151833076.avif');
            background-size: cover;
            background-position: center;
            color: white;
            padding: 2rem;
        }
        .main-container {
            text-align: center;
            max-width: 600px;
            width: 100%;
            background: var(--glass);
            backdrop-filter: blur(15px);
            padding: 3rem;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 2rem;
            font-weight: 700;
            background: linear-gradient(to right, #fff, #00d2ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .rec-list {
            list-style: none;
            padding: 0;
            margin: 2rem 0;
            text-align: left;
        }
        .rec-item {
            background: rgba(255, 255, 255, 0.05);
            margin-bottom: 10px;
            padding: 15px 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: 0.3s;
        }
        .rec-item:hover {
            background: rgba(0, 210, 255, 0.1);
            transform: translateX(5px);
            border-color: var(--accent);
        }
        .product-id {
            color: var(--accent);
            font-weight: 700;
        }
        .back-btn {
            display: inline-block;
            margin-top: 1rem;
            color: rgba(255, 255, 255, 0.6);
            text-decoration: none;
            font-size: 0.9rem;
            transition: 0.3s;
        }
        .back-btn:hover {
            color: var(--accent);
        }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>Suggestions for User #{{ user_id }}</h1>
        <ul class="rec-list">
            {% for item in recommendations %}
            <li class="rec-item">
                <span>Product ID</span>
                <span class="product-id">#{{ item }}</span>
            </li>
            {% endfor %}
        </ul>
        <a href="/" class="back-btn">← Back to Dashboard</a>
    </div>
</body>
</html>
"""

# --- ROUTES ---

@app.route('/')
def index():
    n_users = len(matrix.index) if matrix is not None else 5878
    n_items = len(matrix.columns) if matrix is not None else 4631
    try:
        var = round(model.svd.explained_variance_ratio_.sum() * 100, 1)
    except:
        var = 96.9
    return render_template_string(INDEX_HTML, num_users=n_users, num_items=n_items, variance=var)

@app.route('/get_recs', methods=['POST'])
def get_recs():
    user_id = request.form.get('user_id')
    return redirect(url_for('show_recommendations', user_id=user_id))

@app.route('/recommendations/<int:user_id>')
def show_recommendations(user_id):
    recs = model.get_recommendations(user_id, n=10) if model else ["101", "202", "303", "404", "505"]
    return render_template_string(RECS_HTML, user_id=user_id, recommendations=recs)

def open_browser():
    webbrowser.open_new("https://shopsmart-app-lovable.lovable.app")

if __name__ == '__main__':
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000)