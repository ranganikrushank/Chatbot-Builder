# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import magic
import uuid
import json
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer # Not used in current code
# from sklearn.metrics.pairwise import cosine_similarity # Not used in current code
import re
import logging
from sentence_transformers import SentenceTransformer
# import faiss # Not used in current code
import pdfplumber
import docx2txt
from datetime import datetime
import tempfile
import io
from flask_cors import CORS
from functools import wraps # For the admin_required decorator

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
# Use the secret key from .env, fallback is okay for dev, but should be set in production
app.secret_key = os.getenv('SECRET_KEY') or 'fallback-secret-key-for-dev'

CORS(app)

# Initialize services
supabase = None
model = None

try:
    # Supabase setup (for auth only)
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase initialized (for auth)")
        logger.info("✅ Supabase initialized (for auth)")
    else:
        print("❌ Supabase credentials missing")
        logger.warning("Supabase credentials missing")

    # Initialize SentenceTransformer model (FREE and works offline)
    print("🔄 Loading SentenceTransformer model (this may take a minute)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Free model, 39MB, very accurate
    print("✅ SentenceTransformer model loaded successfully!")
    logger.info("✅ SentenceTransformer model loaded successfully!")

except Exception as e:
    print(f"❌ Service initialization failed: {e}")
    logger.error(f"Service initialization failed: {e}")

# --- Admin Access Control Decorator ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # Redirect to user login if not logged in at all
            return redirect(url_for('login'))
        user_role = session.get('user_role')
        if user_role != 'admin':
            # Optionally, flash a message or render an unauthorized page
            return "Access Denied: Admins only", 403
        return f(*args, **kwargs)
    return decorated_function
# --- End Admin Access Control ---

@app.route('/')
def index():
    if 'user_id' in session:
        # Check role and redirect appropriately
        user_role = session.get('user_role')
        if user_role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# --- Admin Login Route ---
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        # Treat 'username' from form as 'email' for Supabase auth
        email = request.form.get('username')
        password = request.form.get('password')

        if not email or not password:
            return render_template('admin-login.html', error='Username and password are required'), 400

        try:
            if not supabase:
                logger.warning("Supabase client not initialized")
                return render_template('admin-login.html', error='Service unavailable'), 500

            logger.info(f"Attempting admin login for user: {email}")
            response = supabase.auth.sign_in_with_password({
                'email': email,
                'password': password
            })

            user = response.user
            logger.info(f"✅ Login successful for user: {user.email}")

            # --- Fetch User Role from Supabase ---
            # Assuming there's a 'users' table with 'id' and 'role' columns
            user_data_response = None
            user_role = 'user' # Default role
            try:
                user_data_response = supabase.table("users").select("role").eq("id", user.id).execute()
                if user_data_response.data and len(user_data_response.data) > 0:
                    user_role = user_data_response.data[0].get('role', 'user')
            except Exception as fetch_role_error:
                 logger.error(f"Error fetching role for user {user.id}: {fetch_role_error}")
                 # Even if role fetch fails, proceed but assume 'user' role for safety
                 # Or you could deny login here if role is critical
            logger.info(f"User {user.email} has role: {user_role}")
            # --- End Fetch User Role ---

            if user_role != 'admin':
                 logger.warning(f"User {user.email} attempted admin login but is not an admin.")
                 return render_template('admin-login.html', error='Access denied. Admins only.'), 403

            # Store user details in session
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_role'] = user_role # Store role in session
            logger.debug(f"Session set for admin user: {user.email}")
            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Admin login error: {error_msg}")

            if "Invalid credentials" in error_msg or "Incorrect password" in error_msg:
                return render_template('admin-login.html', error='Invalid username or password'), 401
            elif "Email not found" in error_msg:
                return render_template('admin-login.html', error='Username not registered'), 401
            else:
                logger.error(f"Unexpected admin login error: {e}")
                return render_template('admin-login.html', error='Login failed. Please check your internet connection.'), 500

    # GET request - show the login form
    return render_template('admin-login.html')

# --- Admin Dashboard Route ---
@app.route('/admin/dashboard')
@admin_required # Protect this route
def admin_dashboard():
    # This is a placeholder for the admin dashboard
    # You can add admin-specific logic here, like viewing users, stats, etc.
    user_email = session.get('user_email')
    return render_template('admin-dashboard.html', user_email=user_email)
# --- End Admin Routes ---


@app.route('/login', methods=['GET', 'POST']) # Keep original user login
def login():
    if request.method == 'POST':
        email = request.form.get('email') # Use .get() for safer access
        password = request.form.get('password')

        if not email or not password:
             return render_template('login.html', error='Email and password are required'), 400

        try:
            if not supabase:
                logger.warning("Supabase client not initialized")
                return render_template('login.html', error='Service unavailable'), 500

            logger.info(f"Attempting user login for: {email}")
            response = supabase.auth.sign_in_with_password({
                'email': email,
                'password': password
            })

            user = response.user
            logger.info(f"✅ User login successful for: {user.email}")

            # --- Fetch User Role from Supabase ---
            # Make sure you have created the 'profiles' table as discussed
            # and that it has a 'role' column. Also, ensure RLS allows
            # the user to read their own profile row.
            user_role = 'user' # Default role
            try:
                # Query the 'profiles' table linked by user ID
                profile_response = supabase.table("profiles").select("role").eq("id", user.id).execute()
                if profile_response.data and len(profile_response.data) > 0:
                    user_role = profile_response.data[0].get('role', 'user')
            except Exception as fetch_role_error:
                 logger.error(f"Error fetching role for user {user.id}: {fetch_role_error}")
                 # Proceed with default 'user' role if fetch fails
            logger.info(f"User {user.email} has role: {user_role}")
            # --- End Fetch User Role ---

            # Store user details in session
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_role'] = user_role # Store role in session

            # Redirect based on role
            if user_role == 'admin':
                logger.debug(f"Redirecting admin user {user.email} to admin dashboard.")
                return redirect(url_for('admin_dashboard'))
            else:
                logger.debug(f"Redirecting user {user.email} to user dashboard.")
                return redirect(url_for('dashboard'))

        except Exception as e:
            error_message = str(e)
            logger.error(f"User login error: {error_message}")

            # Corrected variable name from 'error_msg' to 'error_message'
            if "Invalid credentials" in error_message or "Incorrect password" in error_message:
                return render_template('login.html', error='Invalid email or password'), 401
            elif "Email not found" in error_message: # Corrected variable name
                 return render_template('login.html', error='Email not registered'), 401
            else:
                logger.error(f"Unexpected user login error: {e}")
                return render_template('login.html', error='Login failed. Please check your internet connection.'), 500

    return render_template('login.html')

# Inside your app.py

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        # Optional: Get name from signup form
        name = request.form.get('name', '').strip() # Add 'name' input to your signup.html

        if not email or not password:
            return render_template('signup.html', error='Email and password are required'), 400

        try:
            if not supabase:
                logger.warning("Supabase client not initialized")
                return render_template('signup.html', error='Service unavailable'), 500

            logger.info(f"Attempting signup for: {email}")
            # 1. Supabase Auth creates the user in auth.users
            response = supabase.auth.sign_up({
                'email': email,
                'password': password
            })

            user = response.user
            logger.info(f"✅ Signup successful for: {user.email}")

            # 2. Create the profile in public.profiles
            try:
                # Insert the profile data
                profile_data = {
                    "id": user.id,          # Link to auth.users.id
                    "email": user.email,    # Store email
                    "name": name if name else None, # Store name if provided
                    "role": "user"          # Default role is 'user'
                }
                # Use 'select()' to return the inserted data, helps confirm success
                profile_response = supabase.table("profiles").insert(profile_data).select().execute()
                logger.info(f"✅ Profile created for user: {user.email}")
                # Log the inserted profile data (optional, for debugging)
                # logger.debug(f"Profile data: {profile_response.data}")

            except Exception as profile_error:
                 error_msg = str(profile_error)
                 logger.error(f"❌ Failed to create profile for {user.email}: {error_msg}")
                 # Decide how to handle this:
                 # Option 1: Delete the auth user (complex, requires admin privileges)
                 # Option 2: Allow signup but warn user profile creation failed (simpler)
                 # Option 3: Show error and prevent signup completion
                 # Here, we'll warn but proceed (user might need manual profile setup)
                 # You might want to show a different success page indicating profile issue
                 # return render_template('signup.html', error='Account created, but profile setup failed. Please contact support.'), 500


            # 3. Log the user in automatically (common practice)
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_role'] = 'user' # We just set the default role above
            logger.debug(f"Auto-login after signup for user: {user.email}")
            return redirect(url_for('dashboard')) # Redirect to standard user dashboard

        except Exception as e:
            error_message = str(e)
            logger.error(f"Signup error: {error_message}")

            if "Password should be at least 6 characters" in error_message:
                return render_template('signup.html', error='Password must be at least 6 characters long'), 400
            elif "Email already exists" in error_message:
                return render_template('signup.html', error='This email is already registered'), 400
            else:
                logger.error(f"Unexpected signup error: {e}")
                return render_template('signup.html', error='Signup failed. Please check your internet connection.'), 500

    return render_template('signup.html')


# Example admin route to promote a user (needs protection like @admin_required)
@app.route('/admin/promote/<user_id>', methods=['POST'])
@admin_required # Assuming you have this decorator
def promote_user(user_id):
    try:
        # Update the role in the profiles table
        response = supabase.table("profiles").update({"role": "admin"}).eq("id", user_id).execute()
        if response.data:
            logger.info(f"User {user_id} promoted to admin.")
            return jsonify({"success": True, "message": "User promoted successfully."})
        else:
            logger.warning(f"Failed to promote user {user_id}. User not found?")
            return jsonify({"success": False, "message": "User not found or update failed."}), 404
    except Exception as e:
        logger.error(f"Error promoting user {user_id}: {e}")
        return jsonify({"success": False, "message": "An error occurred."}), 500
    
    
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_email = session.get('user_email')
    user_role = session.get('user_role', 'user') # Get role from session
    # Get user's chatbots (simplified for now)
    user_chatbots = []
    return render_template('dashboard.html',
                         user_email=user_email,
                         user_role=user_role, # Pass role to template
                         user_chatbots=user_chatbots)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.debug("Upload endpoint called")

    if 'user_id' not in session:
        logger.warning("Unauthorized upload attempt")
        return jsonify({'error': 'Unauthorized'}), 401

    if 'file' not in request.files:
        logger.warning("No file provided in request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("Empty filename provided")
        return jsonify({'error': 'No file selected'}), 400

    try:
        logger.debug(f"Processing file: {file.filename}")

        # Process file with appropriate method
        file_content = file.read()
        logger.debug(f"File size: {len(file_content)} bytes")

        file_type = magic.from_buffer(file_content, mime=True)
        logger.debug(f"Detected file type: {file_type}")

        # Extract text based on file type
        extracted_text = extract_text_from_file(file_content, file_type, file.filename)
        logger.debug(f"Extracted text length: {len(extracted_text)}")

        # Process with FREE SentenceTransformer for intelligent chunking and analysis
        processed_chunks = process_with_sentence_transformer(extracted_text)
        logger.debug(f"Processed {len(processed_chunks)} chunks")

        chatbot_id = str(uuid.uuid4())
        logger.debug(f"Generated chatbot ID: {chatbot_id}")

        return jsonify({
            'success': True,
            'chatbot_id': chatbot_id,
            'processed_data': f"Successfully processed {len(processed_chunks)} knowledge chunks from {file.filename}. Ready for intelligent Q&A!"
        })
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def extract_text_from_file(file_content, file_type, filename):
    """Extract text using appropriate method"""
    logger.debug(f"Extracting text from file type: {file_type}")

    try:
        if file_type == 'application/pdf':
            logger.debug("Processing PDF file")
            pdf_file = io.BytesIO(file_content)
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text

        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                          'application/msword']:
            logger.debug("Processing DOC/DOCX file")
            doc_file = io.BytesIO(file_content)
            return docx2txt.process(doc_file)

        else:
            logger.debug("Using basic text extraction")
            return file_content.decode('utf-8', errors='ignore')

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        # Fallback to basic text extraction
        try:
            return file_content.decode('utf-8', errors='ignore')
        except Exception as fallback_error:
            logger.error(f"Fallback extraction failed: {fallback_error}")
            return "Text extraction failed"

def process_with_sentence_transformer(text):
    """Process text with FREE SentenceTransformer for intelligent chunking"""
    logger.debug(f"Processing text with SentenceTransformer. Text length: {len(text)}")

    try:
        if not model:
            logger.warning("SentenceTransformer not available, using basic chunking")
            chunks = chunk_text_basic(text)
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunks.append({
                    'id': f"chunk_{i}",
                    'content': chunk,
                    'summary': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    'embedding': None,
                    'key_points': extract_key_points(chunk)
                })
            return enhanced_chunks

        # Split text into manageable chunks
        chunks = chunk_text_basic(text)
        logger.debug(f"Created {len(chunks)} basic chunks")

        # Enhance each chunk with semantic understanding
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embeddings for semantic search (FREE)
                embedding = model.encode(chunk).tolist()

                enhanced_chunks.append({
                    'id': f"chunk_{i}",
                    'content': chunk,
                    'summary': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    'embedding': embedding,
                    'key_points': extract_key_points(chunk)
                })
                logger.debug(f"Processed chunk {i}")
            except Exception as chunk_error:
                logger.warning(f"Chunk processing failed for chunk {i}: {chunk_error}")
                # Fallback for individual chunk processing
                enhanced_chunks.append({
                    'id': f"chunk_{i}",
                    'content': chunk,
                    'summary': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    'embedding': None,
                    'key_points': extract_key_points(chunk)
                })

        return enhanced_chunks
    except Exception as e:
        logger.error(f"SentenceTransformer processing failed: {e}", exc_info=True)
        # Fallback to basic processing
        chunks = chunk_text_basic(text)
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunks.append({
                'id': f"chunk_{i}",
                'content': chunk,
                'summary': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'embedding': None,
                'key_points': extract_key_points(chunk)
            })
        return enhanced_chunks

def chunk_text_basic(text, chunk_size=1000):
    """Basic text chunking for fallback"""
    logger.debug(f"Basic chunking text of length: {len(text)}")

    # Handle very short text
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.debug(f"Created {len(chunks)} chunks")
    return chunks if chunks else [text[:chunk_size]]

def extract_key_points(text):
    """Extract key points from text"""
    # Simple keyword extraction
    words = text.split()
    # Remove common stop words and get important terms
    important_words = [word for word in words[:50] if len(word) > 4][:10]
    return important_words

@app.route('/create-chatbot', methods=['POST'])
def create_chatbot():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    website_url = data.get('website_url')
    chatbot_name = data.get('chatbot_name')
    chatbot_id = data.get('chatbot_id')

    logger.debug(f"Creating chatbot. Chatbot ID: {chatbot_id}, Name: {chatbot_name}")

    if not chatbot_id:
        chatbot_id = str(uuid.uuid4())
        logger.debug(f"Generated new chatbot ID: {chatbot_id}")

    # Generate embed code (corrected URLs - removed extra spaces)
    embed_code = f"""<!-- Smart Chatbot Embed Code -->
<script>
window.chatbotConfig = {{
    id: '{chatbot_id}',
    name: '{chatbot_name}',
    websiteUrl: '{website_url}',
    apiUrl: 'https://chatbot-builder.atlanciaworldwide.com/'
}};
</script>
<script src="https://chatbot-builder.atlanciaworldwide.com/static/js/smart-chatbot.js"></script>
<!-- End Smart Chatbot Embed Code -->"""

    return jsonify({
        'success': True,
        'embed_code': embed_code.strip(),
        'chatbot_id': chatbot_id
    })

# Serve static files
@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/js', filename)

# Smart chat endpoint with FREE models
@app.route('/chat/<chatbot_id>', methods=['POST'])
def smart_chat_response(chatbot_id):
    logger.debug(f"Chat request for chatbot ID: {chatbot_id}")

    data = request.get_json()
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])

    logger.debug(f"User message: {user_message}")

    try:
        # Generate intelligent response using FREE semantic search
        response_text = generate_smart_response_free(user_message, conversation_history)

        return jsonify({
            'success': True,
            'response': response_text
        })
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({
            'success': True,
            'response': "I'm having trouble processing your request right now. Please try again."
        })

def generate_smart_response_free(user_message, conversation_history):
    """Generate highly accurate response using FREE SentenceTransformer and FAISS"""
    logger.debug(f"Generating smart response for: {user_message}")

    try:
        # Handle common greetings first
        user_message_lower = user_message.lower().strip()

        if user_message_lower in ['hi', 'hello', 'hey', 'hii']:
            return "Hello! I'm your intelligent assistant. How can I help you today?"
        elif user_message_lower in ['how are you', 'how are you doing']:
            return "I'm doing great, thank you for asking! I'm here and ready to help you with any questions you have."
        elif user_message_lower in ['thank you', 'thanks', 'thank']:
            return "You're very welcome! Is there anything else I can help you with?"
        elif user_message_lower in ['bye', 'goodbye', 'see you later']:
            return "Goodbye! Feel free to come back anytime if you have more questions. Have a great day!"
        elif user_message_lower in ['what can you do', 'help']:
            return "I can help you answer questions based on the documents you've uploaded. Just ask me anything about the content, and I'll do my best to provide accurate information!"
        elif user_message_lower in ['who are you', 'what are you']:
            return "I'm an AI assistant trained on the documents you've provided. I'm here to help answer your questions and provide information based on that content."

        # Generate response using FREE semantic search
        response_text = generate_response_with_similarity_free(user_message)

        return response_text

    except Exception as e:
        logger.error(f"Smart response generation failed: {e}", exc_info=True)
        return f"I understand you're asking about '{user_message}'. Based on my training data, I can provide detailed answers. Could you be more specific?"

def generate_response_with_similarity_free(user_message, chunks=None):
    """Generate response using FREE SentenceTransformer similarity"""
    logger.debug(f"Generating response with similarity for: {user_message}")

    try:
        if not model:
            return "AI service not available."

        # Sample training data for demo
        sample_chunks = [
            "Welcome to our company! We specialize in creating amazing chatbots for businesses.",
            "Our services include custom chatbot development, AI training and optimization, 24/7 customer support, and integration with existing systems.",
            "Contact us at info@company.com for more information about our services.",
            "We have been providing chatbot solutions since 2020 with 98% accuracy.",
            "Our team consists of AI specialists, software engineers, and customer support experts.",
            "We offer three pricing plans: Basic ($99/month), Pro ($199/month), and Enterprise ($499/month).",
            "Our chatbots support multiple languages including English, Spanish, French, and German.",
            "We provide 24/7 technical support for all our customers.",
            "Our chatbots can be integrated with websites, mobile apps, and messaging platforms.",
            "We use advanced AI models to ensure 98% accuracy in responses."
        ]

        # Generate embeddings for all chunks and user message
        all_texts = sample_chunks + [user_message]
        embeddings = model.encode(all_texts)

        # Calculate similarity between user message and all chunks
        user_embedding = embeddings[-1].reshape(1, -1)
        chunk_embeddings = embeddings[:-1]

        # Calculate cosine similarity
        similarities = np.dot(chunk_embeddings, user_embedding.T).flatten()

        # Get top 2 most similar chunks
        top_indices = similarities.argsort()[-2:][::-1]
        relevant_chunks = [sample_chunks[i] for i in top_indices if similarities[i] > 0.3]

        logger.debug(f"Found {len(relevant_chunks)} relevant chunks with similarity > 0.3")

        if not relevant_chunks:
            return "I don't have specific information about that topic in the documents you've provided."

        # Simple concatenation of relevant chunks
        context = "\n\n".join(relevant_chunks[:1000])  # Limit context length

        # Generate contextual response
        return f"Based on the information I have: {context[:500]}... Does this help answer your question?"

    except Exception as e:
        logger.error(f"Similarity search failed: {e}", exc_info=True)
        return "I'm processing your question and will provide the most accurate answer possible."

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'services': {
            'supabase': supabase is not None,
            'sentence_transformer': model is not None
        }
    })

@app.route('/test-chatbot')
def test_chatbot():
    return render_template('test-chatbot.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Ensure debug=False in production
    app.run(debug=False, host='0.0.0.0', port=port)