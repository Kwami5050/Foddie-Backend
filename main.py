from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from datetime import datetime, timedelta
import sqlite3
import secrets
import json
import os
from functools import wraps
from dotenv import load_dotenv
import requests
from groq import Groq
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
YOUTUBE_API_KEY = "AIzaSyAy6x8utPLPPyt_4thV9JKdv3OUHQGLPNI"
PEXELS_API_KEY = "HkCVDhZhSeEIA3UCyGtVr4IPKtsamjfMYIZAivNiMVGv1o2iTqFCwSIt"
G_API_KEY="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"

client = Groq(api_key=G_API_KEY)
# Configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', '3f9b3cd3a4bc4f2181d62e9f4e6e1e96bff932a3ddce781c72c7be8d6e3bb654')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///foodie.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_USER', '0320080067@htu.edu.gh')
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASS', 'xbmhdwfpqodawdzh')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('EMAIL_USER', '0320080067@htu.edu.gh')

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app)

# Load Teachable Machine model
model = tf.keras.models.load_model("keras_model.h5")  # make sure this file exists in your project root

# Load class labels
with open("labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    otp_code = db.Column(db.String(6))
    otp_expiry = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    preferences = db.relationship('UserPreference', backref='user', uselist=False)
    saved_recipes = db.relationship('SavedRecipe', backref='user')

class UserPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    diet_type = db.Column(db.String(50))
    allergies = db.Column(db.String(100))
    cuisine = db.Column(db.String(50))
    cooking_method = db.Column(db.String(50))
    exclusions = db.Column(db.String(100))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class Recipe(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    ingredients = db.Column(db.Text, nullable=False)  # JSON string
    directions = db.Column(db.Text, nullable=False)   # JSON string
    rating = db.Column(db.Float, default=0.0)
    time = db.Column(db.String(20))
    category = db.Column(db.String(50))
    image_path = db.Column(db.String(200))
    video_url = db.Column(db.String(500))
    country = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SavedRecipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.String(50), db.ForeignKey('recipe.id'), nullable=False)
    saved_at = db.Column(db.DateTime, default=datetime.utcnow)

# Utility Functions
def generate_otp():
    return str(secrets.randbelow(1000000)).zfill(6)

def send_otp_email(email, otp):
    try:
        msg = Message(
            'Email Verification - Recipe App',
            recipients=[email],
            html=f'''
            <h2>Welcome to Recipe App!</h2>
            <p>Your verification code is: <strong style="font-size: 24px; color: #007bff;">{otp}</strong></p>
            <p>This code will expire in 10 minutes.</p>
            <p>If you didn't create an account, please ignore this email.</p>
            '''
        )
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

def validate_password(password):
    return len(password) >= 6

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def fetch_pinterest_image(query):
    try:
        search_url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"
        res = requests.get(search_url, timeout=5)

        if res.status_code == 200:
            html = res.text
            start = html.find('src="') + 5
            end = html.find('"', start)
            if start > 4 and end > start:
                return html[start:end]
    except Exception as e:
        print(f"Pinterest fetch error: {e}")
    return None

# Routes
@app.route('/')
def health_check():
    return jsonify({"message": "🚀 Teachable Machine Ingredient Detection API is running!"})

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)[0]
        top_index = int(np.argmax(predictions))
        top_label = labels[top_index]
        confidence = float(predictions[top_index])

        return jsonify({
            "ingredient": top_label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        phone_number = data.get('phone_number', '').strip()
        password = data.get('password', '')
        print(data)
        if not email or not phone_number or not password:
            return jsonify({'error': 'All fields are required'}), 400

        if not validate_password(password):
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 409

        otp = generate_otp()
        otp_expiry = datetime.utcnow() + timedelta(minutes=10)
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(
            email=email,
            phone_number=phone_number,
            password_hash=password_hash,
            otp_code=otp,
            otp_expiry=otp_expiry
        )
        db.session.add(user)
        db.session.commit()

        if send_otp_email(email, otp):
            return jsonify({
                'message': 'Account created successfully. Please check your email for verification code.',
                'user_id': user.id
            }), 201
        else:
            return jsonify({'error': 'Account created but failed to send verification email'}), 500

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        otp = data.get('otp')
        print(data)
        if not user_id or not otp:
            return jsonify({'error': 'User ID and OTP are required'}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        if user.is_verified:
            return jsonify({'error': 'Account already verified'}), 400

        if datetime.utcnow() > user.otp_expiry:
            return jsonify({'error': 'OTP expired'}), 400

        if user.otp_code != otp:
            return jsonify({'error': 'Invalid OTP'}), 400

        user.is_verified = True
        user.otp_code = None
        user.otp_expiry = None
        db.session.commit()

        return jsonify({'message': 'Email verified successfully'}), 200

    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/signin', methods=['POST'])
def signin():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        if not user.is_verified:
            return jsonify({'error': 'Please verify your email before signing in'}), 401

        if not bcrypt.check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid email or password'}), 401

        session['user_id'] = user.id
        session['user_email'] = user.email

        return jsonify({
            'message': 'Signed in successfully',
            'user': {
                'id': user.id,
                'email': user.email,
                'phone_number': user.phone_number
            }
        }), 200

    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/signout', methods=['POST'])
@login_required
def signout():
    session.clear()
    return jsonify({'message': 'Signed out successfully'}), 200

@app.route('/api/preferences', methods=['POST', 'GET'])
def user_preferences():
    if request.method == 'POST':
        try:
            data = request.get_json()
            user_id = data.get('user_id')

            if not user_id:
                return jsonify({'error': 'user_id is required'}), 400

            preferences = UserPreference.query.filter_by(user_id=user_id).first()

            if preferences:
                preferences.diet_type = data.get('diet_type')
                preferences.allergies = data.get('allergies')
                preferences.cuisine = data.get('cuisine')
                preferences.cooking_method = data.get('cooking_method')
                preferences.exclusions = data.get('exclusions')
                preferences.updated_at = datetime.utcnow()
            else:
                preferences = UserPreference(
                    user_id=user_id,
                    diet_type=data.get('diet_type'),
                    allergies=data.get('allergies'),
                    cuisine=data.get('cuisine'),
                    cooking_method=data.get('cooking_method'),
                    exclusions=data.get('exclusions')
                )
                db.session.add(preferences)

            db.session.commit()
            return jsonify({'message': 'Preferences saved successfully'}), 200

        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

    else:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        preferences = UserPreference.query.filter_by(user_id=user_id).first()
        if preferences:
            return jsonify({
                'diet_type': preferences.diet_type,
                'allergies': preferences.allergies,
                'cuisine': preferences.cuisine,
                'cooking_method': preferences.cooking_method,
                'exclusions': preferences.exclusions
            }), 200
        else:
            return jsonify({
                'diet_type': None,
                'allergies': None,
                'cuisine': None,
                'cooking_method': None,
                'exclusions': None
            }), 200

@app.route('/api/recipes', methods=['GET'])
def get_recipes():
    try:
        recipes = Recipe.query.all()
        recipe_list = []
        
        for recipe in recipes:
            recipe_data = {
                'id': recipe.id,
                'title': recipe.title,
                'ingredients': json.loads(recipe.ingredients),
                'directions': json.loads(recipe.directions),
                'rating': recipe.rating,
                'time': recipe.time,
                'category': recipe.category,
                'image': recipe.image_path,
                'videoUrl': recipe.video_url,
                'country': recipe.country
            }
            recipe_list.append(recipe_data)
        
        return jsonify({'recipes': recipe_list}), 200
        
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/recipes/<recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    try:
        recipe = Recipe.query.get(recipe_id)
        if not recipe:
            return jsonify({'error': 'Recipe not found'}), 404
        
        recipe_data = {
            'id': recipe.id,
            'title': recipe.title,
            'ingredients': json.loads(recipe.ingredients),
            'directions': json.loads(recipe.directions),
            'rating': recipe.rating,
            'time': recipe.time,
            'category': recipe.category,
            'image': recipe.image_path,
            'videoUrl': recipe.video_url,
            'country': recipe.country
        }
        
        return jsonify(recipe_data), 200
        
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/recipes', methods=['POST'])
def add_recipe():
    try:
        data = request.get_json()
        
        required_fields = ['id', 'title', 'ingredients', 'directions']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        existing_recipe = Recipe.query.get(data['id'])
        if existing_recipe:
            return jsonify({'error': 'Recipe ID already exists'}), 409
        
        recipe = Recipe(
            id=data['id'],
            title=data['title'],
            ingredients=json.dumps(data['ingredients']),
            directions=json.dumps(data['directions']),
            rating=data.get('rating', 0.0),
            time=data.get('time'),
            category=data.get('category'),
            image_path=data.get('image'),
            video_url=data.get('videoUrl'),
            country=data.get('country')
        )
        
        db.session.add(recipe)
        db.session.commit()
        
        return jsonify({'message': 'Recipe added successfully'}), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/save-recipe', methods=['POST'])
def save_recipe():
    try:
        data = request.get_json()
        recipe_id = data.get('recipe_id')
        user_id = data.get('user_id')
        
        if not recipe_id:
            return jsonify({'error': 'Recipe ID is required'}), 400
        
        recipe = Recipe.query.get(recipe_id)
        if not recipe:
            return jsonify({'error': 'Recipe not found'}), 404
        
        existing_save = SavedRecipe.query.filter_by(
            user_id=user_id, 
            recipe_id=recipe_id
        ).first()
        
        if existing_save:
            return jsonify({'error': 'Recipe already saved'}), 409
        
        saved_recipe = SavedRecipe(user_id=user_id, recipe_id=recipe_id)
        db.session.add(saved_recipe)
        db.session.commit()
        
        return jsonify({'message': 'Recipe saved successfully'}), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/saved-recipes', methods=['GET'])
def get_saved_recipes():
    try:
        user_id = session['user_id']
        
        saved_recipes = db.session.query(Recipe).join(
            SavedRecipe, Recipe.id == SavedRecipe.recipe_id
        ).filter(SavedRecipe.user_id == user_id).all()
        
        recipe_list = []
        for recipe in saved_recipes:
            recipe_data = {
                'id': recipe.id,
                'title': recipe.title,
                'ingredients': json.loads(recipe.ingredients),
                'directions': json.loads(recipe.directions),
                'rating': recipe.rating,
                'time': recipe.time,
                'category': recipe.category,
                'image': recipe.image_path,
                'videoUrl': recipe.video_url,
                'country': recipe.country
            }
            recipe_list.append(recipe_data)
        
        return jsonify({'saved_recipes': recipe_list}), 200
        
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/remove-saved-recipe', methods=['DELETE'])
def remove_saved_recipe():
    try:
        data = request.get_json()
        recipe_id = data.get('recipe_id')
        user_id = data.get('user_id')
        
        if not recipe_id:
            return jsonify({'error': 'Recipe ID is required'}), 400
        
        saved_recipe = SavedRecipe.query.filter_by(
            user_id=user_id, 
            recipe_id=recipe_id
        ).first()
        
        if not saved_recipe:
            return jsonify({'error': 'Saved recipe not found'}), 404
        
        db.session.delete(saved_recipe)
        db.session.commit()
        
        return jsonify({'message': 'Recipe removed from saved list'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/generate-recipes', methods=['POST'])
def generate_recipes():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        custom_text = data.get('custom_text', "")

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        user_pref = UserPreference.query.filter_by(user_id=user_id).first()

        preferences_text = ""
        if user_pref:
            preferences_text = f"""
            Diet type: {user_pref.diet_type or "None"}
            Allergies: {user_pref.allergies or "None"}
            Preferred cuisine: {user_pref.cuisine or "None"}
            Cooking method: {user_pref.cooking_method or "None"}
            Exclusions: {user_pref.exclusions or "None"}
            """

        prompt = f"""
            Generate exactly 5 structured recipes as a JSON array.
            Each recipe must have these exact fields:
            "id" (string), "title" (string), "ingredients" (array of strings), 
            "directions" (array of strings), "rating" (number between 1-5),
            "time" (string like "30 min"), "category" (string), "country" (string).

            User preferences:
            {preferences_text}

            User ingredients/requirements:
            {custom_text}

            Return ONLY the JSON array, no explanations or extra text.
            Example format: [{{"id": "1", "title": "Recipe Name", "ingredients": ["item1", "item2"], "directions": ["step1", "step2"], "rating": 4.5, "time": "30 min", "category": "Main", "country": "Ghana"}}]
            """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a professional chef. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        recipes_json = response.choices[0].message.content
        recipes_json = recipes_json.strip()
        if recipes_json.startswith('```json'):
            recipes_json = recipes_json[7:]
        if recipes_json.endswith('```'):
            recipes_json = recipes_json[:-3]
        recipes_json = recipes_json.strip()

        recipes_data = json.loads(recipes_json)
        enriched_recipes = []

        for i, r in enumerate(recipes_data):
            try:
                recipe_id = r.get('id', f"groq_{i+1}")
                title = r.get('title', f'Generated Recipe {i+1}')
                
                video_url = None
                if YOUTUBE_API_KEY:
                    try:
                        yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q=how+to+make+{title}&key={YOUTUBE_API_KEY}"
                        yt_res = requests.get(yt_url, timeout=5).json()
                        if "items" in yt_res and yt_res["items"]:
                            video_url = f"https://www.youtube.com/watch?v={yt_res['items'][0]['id']['videoId']}"
                    except Exception as e:
                        print(f"YouTube API error: {e}")

                image_url = None
                if PEXELS_API_KEY:
                    try:
                        img_url = f"https://api.pexels.com/v1/search?query={title}&per_page=1"
                        img_res = requests.get(img_url, headers={"Authorization": PEXELS_API_KEY}, timeout=5).json()
                        if img_res.get('photos'):
                            image_url = img_res['photos'][0]['src']['medium']
                    except Exception as e:
                        print(f"Pexels API error: {e}")
                
                if not image_url:
                    image_url = f"https://source.unsplash.com/800x600/?food,{title.replace(' ', '+')}"               
                
                r['image'] = image_url
                r['videoUrl'] = video_url
                enriched_recipes.append(r)
                
            except Exception as recipe_error:
                print(f"Error processing recipe {i}: {recipe_error}")
                continue
            
        db.session.commit()
        return jsonify({'recipes': enriched_recipes}), 200

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return jsonify({'error': f'Invalid JSON response from AI: {str(e)}'}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()
        if Recipe.query.count() == 0:
            sample_recipes = [
                {
                    'id': 23,
                    'title': 'Ghana Meat Pie',
                    'ingredients': [
                        'All-purpose flour',
                        'Baking powder',
                        'Butter or margarine',
                        'Eggs',
                        'Milk',
                        'Corned beef or minced meat',
                        'Onion',
                        'Spices (salt, pepper, garlic powder)',
                        'Optional: diced potato'
                    ],
                    'directions': [
                        'Preheat oven to 180°C (350°F).',
                        'Mix flour, baking powder, and cold butter until crumb-like.',
                        'In a separate bowl, whisk egg(s) and milk, then add to flour mix to form dough.',
                        'Sauté corned beef with onion and spices (and potato if using).',
                        'Roll dough, cut into shapes, fill with the meat mixture, and seal.',
                        'Place on baking tray, brush with beaten egg, and bake until golden (~20–25 min).'
                    ],
                    'rating': 4.9,
                    'time': '60 min',
                    'category': 'Snack',
                    'image': 'https://i.ytimg.com/vi/83JszdEN-Ec/maxresdefault.jpg',
                    'videoUrl': 'https://www.youtube.com/watch?v=QuuRG_Ovq2M',
                    'country': 'Ghana'
                },
                {
                    'id': 24,
                    'title': 'Kelewele',
                    'ingredients': [
                        'Ripe plantains',
                        'Ginger',
                        'Garlic',
                        'Onion',
                        'Chili or cayenne pepper',
                        'Nutmeg',
                        'Salt',
                        'Vegetable oil'
                    ],
                    'directions': [
                        'Peel and cut ripe plantains into bite-sized cubes.',
                        'Blend or finely mince ginger, garlic, onion, pepper, and nutmeg to make a spice paste.',
                        'Toss the plantain cubes in the spice paste until they’re well coated.',
                        'Deep-fry the spiced cubes in hot oil until golden and crispy.',
                        'Drain excess oil and serve hot as a popular Ghanaian snack.'
                    ],
                    'rating': 4.8,
                    'time': '30 min',
                    'category': 'Snack',
                    'image': 'https://i.ytimg.com/vi/ZK-V07D_0Q4/maxresdefault.jpg',
                    'videoUrl': 'https://www.youtube.com/watch?v=c0Bt_1gbko8',
                    'country': 'Ghana'
                },
               
                {
                    'id': 26,
                    'title': 'Okro Stew with Banku',
                    'ingredients': [
                        'Okro (okra, fresh or frozen)',
                        'Palm oil',
                        'Onion',
                        'Tomatoes',
                        'Pepper (e.g., scotch bonnet)',
                        'Salt',
                        'Seasoning cubes',
                        'Smoked fish or meat (optional)',
                        'Banku or fufu (for serving)'
                    ],
                    'directions': [
                        'Slice or chop okro into small pieces.',
                        'Heat palm oil in a pot. Fry chopped onions until fragrant and translucent.',
                        'Add chopped tomatoes and pepper. Cook until the tomato base reduces.',
                        'Stir in okro and mix well.',
                        'Add smoked fish or meat if using, then season with salt and seasoning cubes.',
                        'Simmer until the okro is tender and the stew thickens.',
                        'Serve hot with banku or fufu.'
                    ],
                    'rating': 4.7,
                    'time': '45 min',
                    'category': 'Dinner',
                    'image': 'https://i.ytimg.com/vi/3Xt58Sujaw0/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDu3xO3OWq2OioXq3fSILlWFpdy_A',
                    'videoUrl': 'https://www.youtube.com/watch?v=Z9vw4fRErrE',
                    'country': 'Ghana'
                },
                {
                    'id': 27,
                    'title': 'Gari Foto (Gari Jollof)',
                    'ingredients': [
                        'Gari',
                        'Tomatoes',
                        'Onion',
                        'Pepper',
                        'Seasoning cubes',
                        'Salt',
                        'Cooking oil',
                        'Optional: eggs or protein for added texture'
                    ],
                    'directions': [
                        'Heat oil in a pan, then fry chopped onions until golden.',
                        'Add blended tomatoes and pepper; cook down until the sauce thickens.',
                        'Pour gari into the sauce and stir thoroughly to combine.',
                        'Cook for a few minutes while stirring constantly, until the gari absorbs the sauce and softens.',
                        'Season with salt and seasoning cubes to taste.',
                        'Optionally, stir in cooked eggs or protein for extra flavor.',
                        'Serve hot.'
                    ],
                    'rating': 4.5,
                    'time': '25 min',
                    'category': 'Lunch',
                    'image': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNr5qhLepFEx_humuHpZRWbY6mpQrV5kZwdQ&s',
                    'videoUrl': 'https://www.youtube.com/watch?v=NSoszTf0y1w',
                    'country': 'Ghana'
                }
                
                    
            ]
            for recipe_data in sample_recipes:
                recipe = Recipe(
                    id=recipe_data['id'],
                    title=recipe_data['title'],
                    ingredients=json.dumps(recipe_data['ingredients']),
                    directions=json.dumps(recipe_data['directions']),
                    rating=recipe_data['rating'],
                    time=recipe_data['time'],
                    category=recipe_data['category'],
                    image_path=recipe_data['image'],
                    video_url=recipe_data['videoUrl'],
                    country=recipe_data['country']
                )
                db.session.add(recipe)
            
            db.session.commit()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)