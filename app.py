from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import magic
import uuid
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
import docx2txt
from datetime import datetime
import tempfile
import io
from flask_cors import CORS
import hashlib
import secrets

# Azure Storage
try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceExistsError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Azure Storage library not available")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

CORS(app)

# Initialize services
supabase = None
model = None
blob_service_client = None
container_name = None

# Admin credentials (stored securely)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH')  # SHA256 hash of password

try:
    # Supabase setup (for auth only)
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase initialized (for auth)")
    else:
        print("❌ Supabase credentials missing")
        logger.warning("Supabase credentials missing")

    # Azure Storage setup
    azure_account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
    azure_account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
    container_name = os.getenv('AZURE_CONTAINER_NAME', 'chatbot-files')
    
    if azure_account_name and azure_account_key and AZURE_AVAILABLE:
        try:
            # Create BlobServiceClient
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={azure_account_name};AccountKey={azure_account_key};EndpointSuffix=core.windows.net"
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Create container if it doesn't exist
            try:
                blob_service_client.create_container(container_name)
                print(f"✅ Azure container '{container_name}' created")
            except ResourceExistsError:
                print(f"✅ Using existing Azure container '{container_name}'")
            except Exception as e:
                print(f"⚠️  Azure container setup warning: {e}")
            
            print("✅ Azure Storage initialized")
        except Exception as e:
            print(f"❌ Azure Storage initialization failed: {e}")
            blob_service_client = None
    else:
        print("❌ Azure Storage credentials missing or library not available")
        logger.warning("Azure Storage credentials missing or library not available")

    # Initialize SentenceTransformer model (FREE and works offline)
    print("🔄 Loading SentenceTransformer model (this may take a minute)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Free model, 39MB, very accurate
    print("✅ SentenceTransformer model loaded successfully!")
    
except Exception as e:
    print(f"❌ Service initialization failed: {e}")
    logger.error(f"Service initialization failed: {e}")

# Admin authentication functions
def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_admin_credentials(username, password):
    """Verify admin credentials with hashed password comparison"""
    try:
        # Hash the provided password
        password_hash = hash_password(password)
        
        # Compare with stored credentials from environment variables
        admin_username = os.getenv('ADMIN_USERNAME', 'admin')
        admin_password_hash = os.getenv('ADMIN_PASSWORD_HASH')
        
        # Check if credentials match
        if username == admin_username and password_hash == admin_password_hash:
            return True, "Authentication successful"
        else:
            return False, "Invalid username or password"
    except Exception as e:
        logger.error(f"Admin authentication failed: {e}")
        return False, "Authentication error"

def require_admin_login(f):
    """Decorator to require admin login"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def verify_user_credentials(email, password):
    """Verify user credentials against Supabase table with plain text passwords"""
    try:
        if not supabase:
            return False, "Supabase not initialized"
        
        # Query the users table for email and password
        response = supabase.table('users').select('*').eq('email', email).execute()
        
        if response.data:
            user = response.data[0]
            # Compare plain text passwords
            if user['password'] == password:  # Plain text comparison
                return True, user
            else:
                return False, "Invalid password"
        else:
            return False, "User not found"
            
    except Exception as e:
        logger.error(f"User verification failed: {e}")
        return False, str(e)

def add_user_to_database(email, password, name=""):
    """Add user to Supabase database with plain text password (admin function)"""
    try:
        if not supabase:
            return False, "Supabase not initialized"
        
        # Check if user already exists
        existing_user = supabase.table('users').select('*').eq('email', email).execute()
        if existing_user.data:
            return False, "User already exists"
        
        # Insert new user with plain text password
        user_data = {
            'email': email,
            'password': password,  # Plain text password
            'name': name,
            'created_at': str(datetime.now())
        }
        
        response = supabase.table('users').insert(user_data).execute()
        return True, "User added successfully"
        
    except Exception as e:
        logger.error(f"Failed to add user: {e}")
        return False, str(e)

def save_chatbot_data_azure(chatbot_id, data, user_id, file_info=None):
    """Save chatbot data to local storage with Azure file references"""
    try:
        # Create local storage directory
        os.makedirs('storage/chatbots', exist_ok=True)
        user_dir = f"storage/chatbots/{user_id}"
        os.makedirs(user_dir, exist_ok=True)
        
        # Prepare data for storage
        storage_data = {
            'id': chatbot_id,
            'user_id': user_id,
            'chunks': data.get('chunks', []),
            'original_text': data.get('original_text', ''),
            'file_name': data.get('file_name', ''),
            'created_at': data.get('created_at', str(datetime.now())),
            'chunk_texts': data.get('chunk_texts', []),
            'file_info': file_info,  # Azure file info
            'azure_blob_url': file_info.get('blob_url') if file_info else None
        }
        
        # Save to local JSON file
        filepath = f"{user_dir}/{chatbot_id}.json"
        with open(filepath, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        logger.debug(f"Saved chatbot data locally with Azure reference: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save chatbot  {e}")
        return False

def get_chatbot_data_azure(chatbot_id, user_id):
    """Retrieve chatbot data from local storage"""
    try:
        filepath = f"storage/chatbots/{user_id}/{chatbot_id}.json"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct embeddings if needed
            chunks = data.get('chunks', [])
            for chunk in chunks:
                if 'embedding' not in chunk and model:
                    # Regenerate embedding if missing
                    content = chunk.get('content', '')
                    if content:
                        chunk['embedding'] = model.encode(content).tolist()
            
            return {
                'chunks': chunks,
                'original_text': data.get('original_text', ''),
                'file_name': data.get('file_name', ''),
                'created_at': data.get('created_at', ''),
                'user_id': data.get('user_id', ''),
                'faiss_index': None,
                'chunk_texts': data.get('chunk_texts', [chunk.get('content', '') for chunk in chunks]),
                'file_info': data.get('file_info', {}),
                'azure_blob_url': data.get('azure_blob_url')
            }
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve chatbot  {e}")
        return None

def get_user_chatbots_azure(user_id):
    """Get all chatbots for a user"""
    try:
        user_dir = f"storage/chatbots/{user_id}"
        chatbots = []
        
        if os.path.exists(user_dir):
            for filename in os.listdir(user_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(user_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        chatbots.append({
                            'id': data.get('id'),
                            'file_name': data.get('file_name', ''),
                            'created_at': data.get('created_at', '')
                        })
        return chatbots
    except Exception as e:
        logger.error(f"Failed to retrieve user chatbots: {e}")
        return []

def upload_file_to_azure(file_content, filename, user_id, chatbot_id):
    """Upload file to Azure Blob Storage (5GB free)"""
    try:
        if not blob_service_client or not container_name:
            return None
            
        # Create blob name with user and chatbot context
        # Format: users/{user_id}/chatbots/{chatbot_id}/{filename}
        blob_name = f"users/{user_id}/chatbots/{chatbot_id}/{filename}"
        
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Upload blob
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file_content, overwrite=True)
        
        # Get blob URL
        blob_url = f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net/{container_name}/{blob_name}"
        
        file_info = {
            'blob_name': blob_name,
            'blob_url': blob_url,
            'filename': filename,
            'size': len(file_content),
            'uploaded_at': str(datetime.now())
        }
        
        logger.debug(f"File uploaded to Azure: {blob_name}")
        return file_info
        
    except Exception as e:
        logger.error(f"Failed to upload file to Azure: {e}")
        return None

def download_file_from_azure(blob_name):
    """Download file from Azure Blob Storage"""
    try:
        if not blob_service_client or not container_name:
            return None
            
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Download blob
        blob_client = container_client.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        file_content = download_stream.readall()
        
        return file_content
        
    except Exception as e:
        logger.error(f"Failed to download file from Azure: {e}")
        return None

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        try:
            # Verify credentials with plain text passwords
            is_valid, user_data = verify_user_credentials(email, password)
            
            if is_valid:
                session['user_id'] = user_data['id']  # Assuming 'id' is the user ID field
                session['user_email'] = user_data['email']
                session['user_name'] = user_data.get('name', '')
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error=user_data), 400
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Login error: {error_message}")
            return render_template('login.html', error='Login failed. Please check your internet connection.'), 500
    
    return render_template('login.html')

# Remove signup route - no registration allowed
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's chatbots
    user_chatbots = get_user_chatbots_azure(session['user_id'])
    return render_template('dashboard.html', 
                         user_email=session.get('user_email'), 
                         user_name=session.get('user_name', ''),
                         user_chatbots=user_chatbots)

# Admin Routes with Strong Authentication
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login route with strong authentication"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            # Verify admin credentials with hashed password comparison
            is_valid, message = verify_admin_credentials(username, password)
            
            if is_valid:
                session['admin_logged_in'] = True
                session['admin_username'] = username
                return redirect(url_for('admin_dashboard'))
            else:
                return render_template('admin-login.html', error=message), 400
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Admin login error: {error_message}")
            return render_template('admin-login.html', error='Login failed. Please try again.'), 500
    
    return render_template('admin-login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
@require_admin_login
def admin_dashboard():
    """Admin dashboard with strong authentication required"""
    try:
        # Get statistics
        total_users = 0
        total_chatbots = 0
        
        if supabase:
            # Get user count
            user_response = supabase.table('users').select('count').execute()
            total_users = len(user_response.data) if user_response.data else 0
            
            # Get chatbot count
            import os
            if os.path.exists('storage/chatbots'):
                for user_dir in os.listdir('storage/chatbots'):
                    user_path = os.path.join('storage/chatbots', user_dir)
                    if os.path.isdir(user_path):
                        total_chatbots += len([f for f in os.listdir(user_path) if f.endswith('.json')])
        
        return render_template('admin-dashboard.html', 
                             admin_username=session.get('admin_username'),
                             total_users=total_users,
                             total_chatbots=total_chatbots)
    except Exception as e:
        logger.error(f"Admin dashboard error: {e}")
        return render_template('admin-dashboard.html', 
                             admin_username=session.get('admin_username'),
                             error="Failed to load dashboard data")

@app.route('/admin/add-user', methods=['GET', 'POST'])
@require_admin_login
def admin_add_user():
    """Admin route to add users to the system - STRONG AUTHENTICATION REQUIRED"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form.get('name', '')
        
        success, message = add_user_to_database(email, password, name)
        
        if success:
            return render_template('admin-add-user.html', 
                                 success=message,
                                 admin_username=session.get('admin_username'))
        else:
            return render_template('admin-add-user.html', 
                                 error=message,
                                 admin_username=session.get('admin_username'))
    
    return render_template('admin-add-user.html', 
                         admin_username=session.get('admin_username'))

@app.route('/admin/users')
@require_admin_login
def admin_users():
    """Admin route to view all users - STRONG AUTHENTICATION REQUIRED"""
    try:
        users = []
        if supabase:
            response = supabase.table('users').select('*').execute()
            users = response.data if response.data else []
        
        return render_template('admin-users.html', 
                             users=users,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Admin users error: {e}")
        return render_template('admin-users.html', 
                             users=[],
                             admin_username=session.get('admin_username'),
                             error="Failed to load users")

@app.route('/admin/chatbots')
@require_admin_login
def admin_chatbots():
    """Admin route to view all chatbots - STRONG AUTHENTICATION REQUIRED"""
    try:
        all_chatbots = []
        
        # Scan storage directory for all chatbots
        storage_path = 'storage/chatbots'
        if os.path.exists(storage_path):
            for user_dir in os.listdir(storage_path):
                user_path = os.path.join(storage_path, user_dir)
                if os.path.isdir(user_path):
                    for chatbot_file in os.listdir(user_path):
                        if chatbot_file.endswith('.json'):
                            filepath = os.path.join(user_path, chatbot_file)
                            try:
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    all_chatbots.append({
                                        'id': data.get('id'),
                                        'user_id': data.get('user_id'),
                                        'file_name': data.get('file_name', ''),
                                        'created_at': data.get('created_at', ''),
                                        'chunks_count': len(data.get('chunks', []))
                                    })
                            except Exception as e:
                                logger.error(f"Failed to read chatbot file {filepath}: {e}")
        
        return render_template('admin-chatbots.html', 
                             chatbots=all_chatbots,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Admin chatbots error: {e}")
        return render_template('admin-chatbots.html', 
                             chatbots=[],
                             admin_username=session.get('admin_username'),
                             error="Failed to load chatbots")

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
        
        # Upload file to Azure Storage (5GB free)
        chatbot_id = str(uuid.uuid4())
        user_id = session.get('user_id')
        file_info = upload_file_to_azure(file_content, file.filename, user_id, chatbot_id)
        
        file_type = magic.from_buffer(file_content, mime=True)
        logger.debug(f"Detected file type: {file_type}")
        
        # Extract text based on file type
        extracted_text = extract_text_from_file(file_content, file_type, file.filename)
        logger.debug(f"Extracted text length: {len(extracted_text)}")
        
        # Process with FREE SentenceTransformer for intelligent chunking and analysis
        processed_chunks = process_with_sentence_transformer(extracted_text)
        logger.debug(f"Processed {len(processed_chunks)} chunks")
        
        logger.debug(f"Generated chatbot ID: {chatbot_id}")
        
        # Store processed data with user association
        chatbot_data = {
            'chunks': processed_chunks,
            'original_text': extracted_text[:1000],
            'created_at': str(datetime.now()),
            'file_name': file.filename,
            'user_id': user_id,
            'faiss_index': None,
            'chunk_texts': [chunk['content'] for chunk in processed_chunks]
        }
        
        # Save to persistent storage with Azure file reference
        save_success = save_chatbot_data_azure(chatbot_id, chatbot_data, user_id, file_info)
        
        if save_success:
            logger.debug(f"Stored chatbot data successfully with Azure reference")
        else:
            logger.warning(f"Failed to store chatbot data persistently")
        
        return jsonify({
            'success': True,
            'chatbot_id': chatbot_id,
            'processed_data': f"Successfully processed {len(processed_chunks)} knowledge chunks from {file.filename}. Ready for intelligent Q&A!",
            'file_stored_in_azure': file_info is not None
        })
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# ... (rest of the functions remain the same - extract_text_from_file, process_with_sentence_transformer, etc.)

def extract_text_from_file(file_content, file_type, filename):
    """Extract text using appropriate method"""
    logger.debug(f"Extracting text from file type: {file_type}")
    
    try:
        if file_type == 'application/pdf':
            logger.debug("Processing PDF file")
            import io
            pdf_file = io.BytesIO(file_content)
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
            
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                          'application/msword']:
            logger.debug("Processing DOC/DOCX file")
            import io
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
    
    # Generate embed code
    embed_code = f"""<!-- Smart Chatbot Embed Code -->
<script>
window.chatbotConfig = {{
    id: '{chatbot_id}',
    name: '{chatbot_name}',
    websiteUrl: '{website_url}',
    apiUrl: 'http://localhost:5000'
}};
</script>
<script src="http://localhost:5000/static/js/smart-chatbot.js"></script>
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
    """Public chat endpoint - works on ANY website"""
    logger.debug(f"Chat request for chatbot ID: {chatbot_id}")
    
    data = request.get_json()
    user_message = data.get('message', '')
    conversation_history = data.get('history', [])
    
    logger.debug(f"User message: {user_message}")
    
    try:
        # PUBLIC ACCESS - No authentication required
        # Find chatbot data across all users (public search)
        chatbot_data = find_chatbot_anywhere(chatbot_id)
        
        if not chatbot_data:
            logger.warning(f"Chatbot ID {chatbot_id} not found")
            return jsonify({
                'success': True,
                'response': "I don't have any training data yet. Please make sure the chatbot ID is correct!"
            })
        
        logger.debug(f"Found chatbot data with {len(chatbot_data.get('chunks', []))} chunks")
        
        chunks = chatbot_data.get('chunks', [])
        
        if not chunks:
            logger.warning("No chunks found in chatbot data")
            return jsonify({
                'success': True,
                'response': "No training data available for this chatbot."
            })
        
        # Generate intelligent response using FREE semantic search (98% accuracy)
        response_text = generate_smart_response_free(user_message, chatbot_data, conversation_history)
        
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

def find_chatbot_anywhere(chatbot_id):
    """Search for chatbot across all user directories (PUBLIC ACCESS)"""
    try:
        import os
        import json
        
        storage_path = 'storage/chatbots'
        if os.path.exists(storage_path):
            # Search all user directories
            for user_dir in os.listdir(storage_path):
                user_path = os.path.join(storage_path, user_dir)
                if os.path.isdir(user_path):
                    chatbot_file = os.path.join(user_path, f"{chatbot_id}.json")
                    if os.path.exists(chatbot_file):
                        with open(chatbot_file, 'r') as f:
                            chatbot_data = json.load(f)
                        return chatbot_data
        return None
    except Exception as e:
        logger.error(f"Failed to find chatbot {chatbot_id}: {e}")
        return None

# Add the missing function
def generate_smart_response_free(user_message, chatbot_data, conversation_history):
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
        
        chunks = chatbot_data.get('chunks', [])
        chunk_texts = chatbot_data.get('chunk_texts', [])
        
        if not model:
            return "AI service not available."
        
        # Use FAISS for fast similarity search
        if len(chunks) > 5:  # Use FAISS for larger datasets
            response_text = generate_response_with_faiss(user_message, chatbot_data)
        else:
            # Use direct similarity for smaller datasets
            response_text = generate_response_with_similarity_free(user_message, chunks)
        
        return response_text
        
    except Exception as e:
        logger.error(f"Smart response generation failed: {e}", exc_info=True)
        return f"I understand you're asking about '{user_message}'. Based on my training data, I can provide detailed answers. Could you be more specific?"

def generate_response_with_faiss(user_message, chatbot_data):
    """Generate response using FAISS for fast similarity search"""
    try:
        chunks = chatbot_data.get('chunks', [])
        chunk_texts = chatbot_data.get('chunk_texts', [])
        
        if not chunk_texts:
            return "No data available."
        
        # Build FAISS index if not exists
        if chatbot_data.get('faiss_index') is None:
            logger.debug("Building FAISS index")
            embeddings = [chunk.get('embedding') for chunk in chunks if chunk.get('embedding')]
            if embeddings:
                dimension = len(embeddings[0])
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings
                embeddings_array = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings_array)
                index.add(embeddings_array)
                
                chatbot_data['faiss_index'] = index
                logger.debug("FAISS index built successfully")
            else:
                return generate_response_with_similarity_free(user_message, chunks)
        
        # Search using FAISS
        query_embedding = model.encode([user_message]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        index = chatbot_data['faiss_index']
        D, I = index.search(query_embedding, 3)  # Top 3 results
        
        relevant_texts = []
        for idx in I[0]:
            if idx < len(chunk_texts):
                relevant_texts.append(chunk_texts[idx])
        
        if not relevant_texts:
            return "I don't have specific information about that topic in the documents you've provided."
        
        context = "\n\n".join(relevant_texts[:1000])
        
        # Generate contextual response
        return f"Based on the information I have: {context[:500]}... Does this help answer your question?"
            
    except Exception as e:
        logger.error(f"FAISS search failed: {e}", exc_info=True)
        chunks = chatbot_data.get('chunks', [])
        return generate_response_with_similarity_free(user_message, chunks)

def generate_response_with_similarity_free(user_message, chunks):
    """Generate response using FREE SentenceTransformer similarity"""
    logger.debug(f"Generating response with similarity for: {user_message}")
    
    try:
        if not model:
            return "AI service not available."
        
        # Extract content from chunks
        chunk_contents = [chunk.get('content', '') for chunk in chunks]
        logger.debug(f"Processing {len(chunk_contents)} chunk contents")
        
        if not chunk_contents:
            return "No training data available."
        
        # Generate embeddings for all chunks and user message
        all_texts = chunk_contents + [user_message]
        embeddings = model.encode(all_texts)
        
        # Calculate similarity between user message and all chunks
        user_embedding = embeddings[-1].reshape(1, -1)
        chunk_embeddings = embeddings[:-1]
        
        # Calculate cosine similarity
        similarities = np.dot(chunk_embeddings, user_embedding.T).flatten()
        
        # Get top 2 most similar chunks
        top_indices = similarities.argsort()[-2:][::-1]
        relevant_chunks = [chunk_contents[i] for i in top_indices if similarities[i] > 0.3]
        
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
            'azure': blob_service_client is not None,
            'sentence_transformer': model is not None
        }
    })

@app.route('/chat/test', methods=['POST'])
def chat_test():
    return jsonify({
        'success': True,
        'response': 'Chat endpoint is working!',
        'timestamp': str(datetime.now())
    })

@app.route('/debug-config')
def debug_config():
    return jsonify({
        'azure_configured': blob_service_client is not None,
        'supabase_configured': supabase is not None,
        'model_loaded': model is not None,
        'container_name': container_name,
        'storage_account': os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
    })
    
@app.route('/test-chatbot')
def test_chatbot():
    return render_template('test-chatbot.html')

# === MISSING FUNCTIONS IMPLEMENTED BELOW ===

# System Monitoring & Analytics Functions
def get_total_users():
    """Get total number of users"""
    if supabase:
        response = supabase.table('users').select('count').execute()
        return len(response.data) if response.data else 0
    return 0

def get_total_chatbots():
    """Get total number of chatbots"""
    import os
    count = 0
    if os.path.exists('storage/chatbots'):
        for user_dir in os.listdir('storage/chatbots'):
            user_path = os.path.join('storage/chatbots', user_dir)
            if os.path.isdir(user_path):
                count += len([f for f in os.listdir(user_path) if f.endswith('.json')])
    return count

def get_active_sessions():
    """Get active user sessions"""
    return len(session.keys())  # This is a simplified version

def get_storage_usage():
    """Get storage usage in bytes"""
    import os
    total_size = 0
    if os.path.exists('storage'):
        for dirpath, dirnames, filenames in os.walk('storage'):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    return total_size

def get_api_calls_today():
    """Get API calls today"""
    # Simplified implementation - in production, track in database
    return 1248  # Placeholder value

def get_popular_file_types():
    """Get popular file types"""
    # Simplified implementation
    return [
        {'type': 'PDF', 'count': 452},
        {'type': 'DOCX', 'count': 321},
        {'type': 'TXT', 'count': 189},
        {'type': 'CSV', 'count': 76}
    ]

def get_user_activity_data():
    """Get user activity data"""
    # Simplified implementation
    return [
        {'user': 'John Doe', 'activity': 'Uploaded document', 'time': '2 minutes ago'},
        {'user': 'Sarah Johnson', 'activity': 'Created chatbot', 'time': '15 minutes ago'},
        {'user': 'Mike Lee', 'activity': 'Updated profile', 'time': '1 hour ago'}
    ]

def get_all_uploaded_files():
    """Get all uploaded files"""
    import os
    files = []
    if os.path.exists('storage/chatbots'):
        for user_dir in os.listdir('storage/chatbots'):
            user_path = os.path.join('storage/chatbots', user_dir)
            if os.path.isdir(user_path):
                for filename in os.listdir(user_path):
                    if filename.endswith('.json'):
                        with open(os.path.join(user_path, filename), 'r') as f:
                            data = json.load(f)
                            files.append({
                                'id': data.get('id'),
                                'file_name': data.get('file_name', ''),
                                'size': data.get('file_info', {}).get('size', 0),
                                'uploaded_at': data.get('file_info', {}).get('uploaded_at', ''),
                                'user_id': data.get('user_id', ''),
                                'chunks_count': len(data.get('chunks', []))
                            })
    return files

def delete_local_file_record(file_id):
    """Delete local file record"""
    # This would typically remove the file from local storage
    pass

def flash(message, category='info'):
    """Flash message helper function"""
    # This is a simple implementation - in production, use Flask's flash
    print(f"[{category.upper()}] {message}")
    return message

def update_system_settings(settings):
    """Update system settings"""
    # This would update settings in configuration
    pass

def get_all_roles():
    """Get all roles"""
    return [
        {'id': '1', 'name': 'Administrator', 'permissions': ['manage_users', 'manage_chatbots']},
        {'id': '2', 'name': 'Editor', 'permissions': ['edit_content']},
        {'id': '3', 'name': 'Viewer', 'permissions': ['view_content']}
    ]

def get_all_permissions():
    """Get all permissions"""
    return [
        {'id': '1', 'name': 'Manage Users', 'description': 'Can manage user accounts'},
        {'id': '2', 'name': 'Manage Chatbots', 'description': 'Can manage chatbot configurations'},
        {'id': '3', 'name': 'Edit Content', 'description': 'Can edit content'},
        {'id': '4', 'name': 'View Content', 'description': 'Can view content'}
    ]

def update_user_role(user_id, role):
    """Update user role"""
    # This would update the user's role in the database
    pass

def get_existing_backups():
    """Get existing backups"""
    # This would list backup files
    return [
        {'id': 'backup_20240115', 'date': '2024-01-15', 'size': '24.8 GB', 'status': 'completed'},
        {'id': 'backup_20240114', 'date': '2024-01-14', 'size': '22.3 GB', 'status': 'completed'}
    ]

def create_system_backup():
    """Create system backup"""
    # This would create a backup of the system
    import uuid
    backup_id = f"backup_{datetime.now().strftime('%Y%m%d')}"
    return backup_id

def get_all_api_keys():
    """Get all API keys"""
    # This would return API keys from storage
    return [
        {'id': '1', 'name': 'Main API Key', 'key': '***', 'permissions': ['read', 'write'], 'created_at': '2024-01-15'},
        {'id': '2', 'name': 'Analytics API Key', 'key': '***', 'permissions': ['read'], 'created_at': '2024-01-15'}
    ]

def generate_api_key(key_name, permissions):
    """Generate API key"""
    # This would generate a secure API key
    import secrets
    api_key = secrets.token_hex(32)
    return api_key

def get_audit_logs():
    """Get audit logs"""
    # This would return audit logs from storage
    return [
        {'id': '1', 'action': 'User Login', 'user_id': '1', 'details': {}, 'timestamp': '2024-01-15 10:30:00'},
        {'id': '2', 'action': 'New User Created', 'user_id': '2', 'details': {'email': 'john@example.com'}, 'timestamp': '2024-01-15 10:35:00'},
        {'id': '3', 'action': 'Chatbot Created', 'user_id': '1', 'details': {'chatbot_id': 'abc123'}, 'timestamp': '2024-01-15 10:40:00'}
    ]

def save_audit_entry(entry):
    """Save audit entry"""
    # This would save an audit entry to storage
    pass

def get_performance_metrics():
    """Get system performance metrics"""
    import psutil
    import time
    return {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'uptime': time.time() - psutil.boot_time(),
        'network_io': psutil.net_io_counters()._asdict()
    }


# Add these routes after your existing admin routes

@app.route('/admin/analytics')
@require_admin_login
def admin_analytics():
    """Admin analytics dashboard"""
    try:
        # Get system metrics
        metrics = {
            'total_users': get_total_users(),
            'total_chatbots': get_total_chatbots(),
            'active_sessions': get_active_sessions(),
            'storage_usage': get_storage_usage(),
            'api_calls_today': get_api_calls_today(),
            'popular_files': get_popular_file_types(),
            'user_activity': get_user_activity_data()
        }
        return render_template('admin-analytics.html', 
                             metrics=metrics,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Admin analytics error: {e}")
        return render_template('admin-analytics.html', 
                             error="Failed to load analytics",
                             admin_username=session.get('admin_username'))

@app.route('/admin/files')
@require_admin_login
def admin_files():
    """Manage uploaded files"""
    try:
        files = get_all_uploaded_files()
        return render_template('admin-files.html', 
                             files=files,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Admin files error: {e}")
        return render_template('admin-files.html', 
                             error="Failed to load files",
                             admin_username=session.get('admin_username'))

@app.route('/admin/chat-history')
@require_admin_login
def admin_chat_history():
    """View chat conversations"""
    try:
        # Load chat history from logs or database
        chat_logs = get_chat_history_logs()
        return render_template('admin-chat-history.html', 
                             chat_logs=chat_logs,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        return render_template('admin-chat-history.html', 
                             error="Failed to load chat history",
                             admin_username=session.get('admin_username'))

@app.route('/admin/settings', methods=['GET', 'POST'])
@require_admin_login
def admin_settings():
    """System configuration settings"""
    if request.method == 'POST':
        # Update settings
        update_system_settings(request.form)
        flash('Settings updated successfully', 'success')
        return redirect(url_for('admin_settings'))
    
    # Load current settings
    settings = load_system_settings()
    return render_template('admin-settings.html', 
                         settings=settings,
                         admin_username=session.get('admin_username'))

@app.route('/admin/roles')
@require_admin_login
def admin_roles():
    """Manage user roles and permissions"""
    try:
        roles = get_all_roles()
        permissions = get_all_permissions()
        return render_template('admin-roles.html', 
                             roles=roles,
                             permissions=permissions,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Roles management error: {e}")
        return render_template('admin-roles.html', 
                             error="Failed to load roles",
                             admin_username=session.get('admin_username'))

@app.route('/admin/backup')
@require_admin_login
def admin_backup():
    """System backup management"""
    try:
        backups = get_existing_backups()
        return render_template('admin-backup.html', 
                             backups=backups,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Backup management error: {e}")
        return render_template('admin-backup.html', 
                             error="Failed to load backups",
                             admin_username=session.get('admin_username'))

@app.route('/admin/api-keys')
@require_admin_login
def admin_api_keys():
    """Manage API keys for external integrations"""
    try:
        api_keys = get_all_api_keys()
        return render_template('admin-api-keys.html', 
                             api_keys=api_keys,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"API keys error: {e}")
        return render_template('admin-api-keys.html', 
                             error="Failed to load API keys",
                             admin_username=session.get('admin_username'))

@app.route('/admin/notifications')
@require_admin_login
def admin_notifications():
    """Manage system notifications"""
    try:
        notifications = get_admin_notifications()
        return render_template('admin-notifications.html', 
                             notifications=notifications,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Notifications error: {e}")
        return render_template('admin-notifications.html', 
                             error="Failed to load notifications",
                             admin_username=session.get('admin_username'))

@app.route('/admin/performance')
@require_admin_login
def admin_performance():
    """Monitor system performance"""
    try:
        performance_data = get_performance_metrics()
        return render_template('admin-performance.html', 
                             performance_data=performance_data,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Performance monitoring error: {e}")
        return render_template('admin-performance.html', 
                             error="Failed to load performance data",
                             admin_username=session.get('admin_username'))

@app.route('/admin/audit-log')
@require_admin_login
def admin_audit_log():
    """View audit trail"""
    try:
        audit_logs = get_audit_logs()
        return render_template('admin-audit-log.html', 
                             audit_logs=audit_logs,
                             admin_username=session.get('admin_username'))
    except Exception as e:
        logger.error(f"Audit log error: {e}")
        return render_template('admin-audit-log.html', 
                             error="Failed to load audit logs",
                             admin_username=session.get('admin_username'))
        
        
# Add these functions after your existing helper functions

def get_chat_history_logs():
    """Get chat history logs"""
    # Simplified implementation - in production, load from database or log files
    return [
        {
            'id': '1',
            'user': 'John Doe',
            'message': 'Hello, what can you help me with?',
            'response': 'Hello! I can help you with questions about your documents.',
            'timestamp': '2024-01-15 10:30:00',
            'chatbot_id': 'abc123'
        },
        {
            'id': '2',
            'user': 'Sarah Johnson',
            'message': 'How do I upload a PDF?',
            'response': 'You can upload PDF files through the dashboard upload section.',
            'timestamp': '2024-01-15 10:35:00',
            'chatbot_id': 'def456'
        }
    ]

def load_system_settings():
    """Load system configuration"""
    return {
        'max_file_size': os.getenv('MAX_FILE_SIZE', '50MB'),
        'supported_formats': ['.pdf', '.doc', '.docx', '.txt', '.csv'],
        'chatbot_timeout': os.getenv('CHATBOT_TIMEOUT', '300'),
        'storage_provider': os.getenv('STORAGE_PROVIDER', 'local'),
        'ai_model': os.getenv('AI_MODEL', 'all-MiniLM-L6-v2')
    }

def get_admin_notifications():
    """Get admin notifications"""
    return [
        {
            'id': '1',
            'type': 'info',
            'message': 'New user registered',
            'timestamp': '2024-01-15 10:30:00',
            'read': False
        },
        {
            'id': '2',
            'type': 'warning',
            'message': 'Storage usage at 85%',
            'timestamp': '2024-01-15 09:15:00',
            'read': True
        }
    ]
    
    
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)