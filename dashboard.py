"""
Simple Web Dashboard with Passkey Authentication
Provides a protected dashboard interface for the Vanna AI application.
"""

import os
from functools import wraps
from dotenv import load_dotenv

import httpx
from flask import Flask, render_template_string, request, redirect, url_for, session, flash

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

DASHBOARD_PASSKEY = os.getenv("DASHBOARD_PASSKEY", "changeme")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))

VANNA_BASE_URL = os.getenv("VANNA_BASE_URL", "http://app:8000")

# HTML Templates
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Login</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
    <div class="w-full max-w-md">
        <div class="bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl p-8 border border-white/20">
            <div class="text-center mb-8">
                <div class="inline-flex items-center justify-center w-16 h-16 bg-purple-500/20 rounded-full mb-4">
                    <i class="fas fa-lock text-purple-400 text-2xl"></i>
                </div>
                <h1 class="text-2xl font-bold text-white">Dashboard Access</h1>
                <p class="text-gray-400 mt-2">Enter your passkey to continue</p>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="mb-4 p-3 rounded-lg {% if category == 'error' %}bg-red-500/20 text-red-300 border border-red-500/30{% else %}bg-green-500/20 text-green-300 border border-green-500/30{% endif %}">
                            <i class="fas {% if category == 'error' %}fa-exclamation-circle{% else %}fa-check-circle{% endif %} mr-2"></i>
                            {{ message }}
                        </div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <form method="POST" action="{{ url_for('login') }}">
                <div class="mb-6">
                    <label for="passkey" class="block text-sm font-medium text-gray-300 mb-2">Passkey</label>
                    <div class="relative">
                        <input type="password" id="passkey" name="passkey" required
                            class="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                            placeholder="Enter your passkey">
                        <i class="fas fa-key absolute right-4 top-1/2 -translate-y-1/2 text-gray-500"></i>
                    </div>
                </div>
                <button type="submit" 
                    class="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-2">
                    <i class="fas fa-sign-in-alt"></i>
                    Access Dashboard
                </button>
            </form>
        </div>
        <p class="text-center text-gray-500 text-sm mt-6">
            <i class="fas fa-shield-alt mr-1"></i> Secured Access
        </p>
    </div>
</body>
</html>
"""

MEMORIES_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Persistent Memories - Vanna AI Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
    <nav class="bg-white/5 backdrop-blur-lg border-b border-white/10">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                        <i class="fas fa-robot text-white"></i>
                    </div>
                    <a class="text-white font-bold text-xl" href="{{ url_for('dashboard') }}">Vanna AI Dashboard</a>
                </div>
                <div class="flex items-center gap-2">
                    <a href="{{ url_for('memories') }}"
                        class="flex items-center gap-2 px-3 py-2 bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-brain"></i>
                        Memories
                    </a>
                    <a href="{{ url_for('golden_queries') }}"
                        class="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-star"></i>
                        Golden Queries
                    </a>
                    <a href="{{ url_for('logout') }}" 
                        class="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-sign-out-alt"></i>
                        Logout
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-white mb-2">Persistent Agent Memory</h1>
            <p class="text-gray-400">Manage text memories stored in ChromaDB</p>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="mb-4 p-3 rounded-lg {% if category == 'error' %}bg-red-500/20 text-red-300 border border-red-500/30{% else %}bg-green-500/20 text-green-300 border border-green-500/30{% endif %}">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10 mb-8">
            <h2 class="text-xl font-bold text-white mb-4">Add Memory</h2>
            <form method="POST" action="{{ url_for('memories_create') }}" class="space-y-4">
                <textarea name="content" required rows="5"
                    class="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                    placeholder="Add a note, schema hint, business rule, etc..."></textarea>
                <button type="submit"
                    class="py-2 px-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all duration-200">
                    Save Memory
                </button>
            </form>
        </div>

        <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-xl font-bold text-white">Recent Memories</h2>
                <a href="{{ url_for('memories') }}" class="text-purple-300 hover:text-purple-200 text-sm">Refresh</a>
            </div>
            <div class="space-y-4">
                {% for item in items %}
                    <div class="bg-white/5 rounded-lg border border-white/10 p-4">
                        <div class="flex items-start justify-between gap-4">
                            <div class="min-w-0">
                                <div class="text-gray-400 text-xs break-all">{{ item.id }}</div>
                                <pre class="text-white whitespace-pre-wrap break-words mt-2 text-sm">{{ item.content }}</pre>
                            </div>
                            <form method="POST" action="{{ url_for('memories_delete', memory_id=item.id) }}" onsubmit="return confirm('Delete this memory?');">
                                <button type="submit" class="text-red-300 hover:text-red-200 text-sm">Delete</button>
                            </form>
                        </div>
                    </div>
                {% endfor %}
                {% if not items %}
                    <div class="text-gray-400">No memories found.</div>
                {% endif %}
            </div>
        </div>
    </main>
</body>
</html>
"""

GOLDEN_QUERIES_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Golden Queries - Vanna AI Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
    <nav class="bg-white/5 backdrop-blur-lg border-b border-white/10">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                        <i class="fas fa-robot text-white"></i>
                    </div>
                    <a class="text-white font-bold text-xl" href="{{ url_for('dashboard') }}">Vanna AI Dashboard</a>
                </div>
                <div class="flex items-center gap-2">
                    <a href="{{ url_for('memories') }}"
                        class="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-brain"></i>
                        Memories
                    </a>
                    <a href="{{ url_for('golden_queries') }}"
                        class="flex items-center gap-2 px-3 py-2 bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-star"></i>
                        Golden Queries
                    </a>
                    <a href="{{ url_for('logout') }}" 
                        class="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-sign-out-alt"></i>
                        Logout
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-white mb-2">Golden Queries</h1>
            <p class="text-gray-400">Store canonical question + SQL pairs as text memories</p>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="mb-4 p-3 rounded-lg {% if category == 'error' %}bg-red-500/20 text-red-300 border border-red-500/30{% else %}bg-green-500/20 text-green-300 border border-green-500/30{% endif %}">
                        {{ message }}
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10 mb-8">
            <h2 class="text-xl font-bold text-white mb-4">Add Golden Query</h2>
            <form method="POST" action="{{ url_for('golden_queries_create') }}" class="space-y-4">
                <input name="question" required
                    class="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                    placeholder="Question (e.g. Top 5 products by revenue)">
                <textarea name="sql" required rows="6"
                    class="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
                    placeholder="SQL (MySQL)"></textarea>
                <button type="submit"
                    class="py-2 px-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all duration-200">
                    Save Golden Query
                </button>
            </form>
        </div>

        <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-xl font-bold text-white">Stored Golden Queries</h2>
                <a href="{{ url_for('golden_queries') }}" class="text-purple-300 hover:text-purple-200 text-sm">Refresh</a>
            </div>
            <div class="space-y-4">
                {% for item in items %}
                    <div class="bg-white/5 rounded-lg border border-white/10 p-4">
                        <div class="flex items-start justify-between gap-4">
                            <div class="min-w-0">
                                <div class="text-gray-400 text-xs break-all">{{ item.id }}</div>
                                <pre class="text-white whitespace-pre-wrap break-words mt-2 text-sm">{{ item.content }}</pre>
                            </div>
                            <form method="POST" action="{{ url_for('golden_queries_delete', memory_id=item.id) }}" onsubmit="return confirm('Delete this golden query?');">
                                <button type="submit" class="text-red-300 hover:text-red-200 text-sm">Delete</button>
                            </form>
                        </div>
                    </div>
                {% endfor %}
                {% if not items %}
                    <div class="text-gray-400">No golden queries found.</div>
                {% endif %}
            </div>
        </div>
    </main>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vanna AI Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
    <!-- Navigation -->
    <nav class="bg-white/5 backdrop-blur-lg border-b border-white/10">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                        <i class="fas fa-robot text-white"></i>
                    </div>
                    <span class="text-white font-bold text-xl">Vanna AI Dashboard</span>
                </div>
                <div class="flex items-center gap-2">
                    <a href="{{ url_for('memories') }}"
                        class="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-brain"></i>
                        Memories
                    </a>
                    <a href="{{ url_for('golden_queries') }}"
                        class="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-star"></i>
                        Golden Queries
                    </a>
                    <a href="{{ url_for('logout') }}" 
                        class="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg transition">
                        <i class="fas fa-sign-out-alt"></i>
                        Logout
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Welcome Section -->
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-white mb-2">Welcome to the Dashboard</h1>
            <p class="text-gray-400">Monitor and manage your Vanna AI application</p>
        </div>

        <!-- Stats Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Vanna App</p>
                        <p class="text-2xl font-bold text-white mt-1">Port 8000</p>
                    </div>
                    <div class="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center">
                        <i class="fas fa-server text-green-400 text-xl"></i>
                    </div>
                </div>
                <div class="mt-4 flex items-center text-green-400 text-sm">
                    <i class="fas fa-circle text-xs mr-2"></i>
                    Running
                </div>
            </div>

            <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">MySQL Database</p>
                        <p class="text-2xl font-bold text-white mt-1">Port 3306</p>
                    </div>
                    <div class="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                        <i class="fas fa-database text-blue-400 text-xl"></i>
                    </div>
                </div>
                <div class="mt-4 flex items-center text-green-400 text-sm">
                    <i class="fas fa-circle text-xs mr-2"></i>
                    Connected
                </div>
            </div>

            <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">ChromaDB</p>
                        <p class="text-2xl font-bold text-white mt-1">Port 8001</p>
                    </div>
                    <div class="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                        <i class="fas fa-brain text-purple-400 text-xl"></i>
                    </div>
                </div>
                <div class="mt-4 flex items-center text-green-400 text-sm">
                    <i class="fas fa-circle text-xs mr-2"></i>
                    Active
                </div>
            </div>

            <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-gray-400 text-sm">Dashboard</p>
                        <p class="text-2xl font-bold text-white mt-1">Port {{ port }}</p>
                    </div>
                    <div class="w-12 h-12 bg-pink-500/20 rounded-lg flex items-center justify-center">
                        <i class="fas fa-chart-line text-pink-400 text-xl"></i>
                    </div>
                </div>
                <div class="mt-4 flex items-center text-green-400 text-sm">
                    <i class="fas fa-circle text-xs mr-2"></i>
                    Online
                </div>
            </div>
        </div>

        <!-- Quick Actions -->
        <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10 mb-8">
            <h2 class="text-xl font-bold text-white mb-4">Quick Actions</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <a href="#" onclick="window.open(window.location.protocol + '//' + window.location.hostname + ':8000', '_blank'); return false;"
                    class="flex items-center gap-3 p-4 bg-white/5 hover:bg-white/10 rounded-lg transition border border-white/10 cursor-pointer">
                    <div class="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                        <i class="fas fa-comments text-purple-400"></i>
                    </div>
                    <div>
                        <p class="text-white font-medium">Open Vanna Chat</p>
                        <p class="text-gray-400 text-sm">Natural language SQL</p>
                    </div>
                </a>
                
                <a href="#" onclick="window.open(window.location.protocol + '//' + window.location.hostname + ':8000/docs', '_blank'); return false;"
                    class="flex items-center gap-3 p-4 bg-white/5 hover:bg-white/10 rounded-lg transition border border-white/10 cursor-pointer">
                    <div class="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                        <i class="fas fa-book text-blue-400"></i>
                    </div>
                    <div>
                        <p class="text-white font-medium">API Documentation</p>
                        <p class="text-gray-400 text-sm">OpenAPI / Swagger</p>
                    </div>
                </a>
                
                <a href="#" onclick="window.open(window.location.protocol + '//' + window.location.hostname + ':8000/health', '_blank'); return false;"
                    class="flex items-center gap-3 p-4 bg-white/5 hover:bg-white/10 rounded-lg transition border border-white/10 cursor-pointer">
                    <div class="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                        <i class="fas fa-heartbeat text-green-400"></i>
                    </div>
                    <div>
                        <p class="text-white font-medium">Health Check</p>
                        <p class="text-gray-400 text-sm">Service status</p>
                    </div>
                </a>
            </div>
        </div>

        <!-- System Info -->
        <div class="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/10">
            <h2 class="text-xl font-bold text-white mb-4">System Information</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="space-y-3">
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-gray-400">Environment</span>
                        <span class="text-white font-medium">{{ env }}</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-gray-400">OpenAI Model</span>
                        <span class="text-white font-medium">{{ openai_model }}</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-gray-400">Database</span>
                        <span class="text-white font-medium">{{ mysql_database }}</span>
                    </div>
                </div>
                <div class="space-y-3">
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-gray-400">Log Level</span>
                        <span class="text-white font-medium">{{ log_level }}</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-gray-400">ChromaDB Collection</span>
                        <span class="text-white font-medium">{{ chroma_collection }}</span>
                    </div>
                    <div class="flex justify-between py-2 border-b border-white/10">
                        <span class="text-gray-400">Embedding Model</span>
                        <span class="text-white font-medium">{{ embedding_model }}</span>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <p class="text-center text-gray-500 text-sm">
            Vanna AI Dashboard &copy; 2024 | Powered by Flask
        </p>
    </footer>
</body>
</html>
"""


def require_auth(f):
    """Decorator to require passkey authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('authenticated'):
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        passkey = request.form.get('passkey', '')
        if passkey == DASHBOARD_PASSKEY:
            session['authenticated'] = True
            flash('Successfully authenticated!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid passkey. Please try again.', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))


@app.route('/')
@require_auth
def dashboard():
    return render_template_string(
        DASHBOARD_TEMPLATE,
        port=DASHBOARD_PORT,
        env=os.getenv('APP_ENV', 'development'),
        openai_model=os.getenv('OPENAI_MODEL', 'gpt-4'),
        mysql_database=os.getenv('MYSQL_DO_DATABASE', 'vanna'),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        chroma_collection=os.getenv('CHROMA_COLLECTION_NAME', 'vanna_memories'),
        embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    )


def _api_get(path: str, params: dict | None = None) -> dict:
    url = f"{VANNA_BASE_URL}{path}"
    with httpx.Client(timeout=10.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def _api_post(path: str, json_body: dict) -> dict:
    url = f"{VANNA_BASE_URL}{path}"
    with httpx.Client(timeout=10.0) as client:
        r = client.post(url, json=json_body)
        r.raise_for_status()
        return r.json() if r.text else {"status": "success"}


def _api_delete(path: str) -> dict:
    url = f"{VANNA_BASE_URL}{path}"
    with httpx.Client(timeout=10.0) as client:
        r = client.delete(url)
        r.raise_for_status()
        return r.json() if r.text else {"status": "success"}


@app.route('/memories')
@require_auth
def memories():
    try:
        data = _api_get('/api/admin/memory/text', params={'limit': 100, 'offset': 0})
        items = data.get('items', [])
        return render_template_string(MEMORIES_TEMPLATE, items=items)
    except Exception as e:
        flash(f"Failed to load memories: {e}", 'error')
        return render_template_string(MEMORIES_TEMPLATE, items=[])


@app.route('/memories/create', methods=['POST'])
@require_auth
def memories_create():
    content = request.form.get('content', '')
    try:
        _api_post('/api/admin/memory/text', {'content': content})
        flash('Memory saved.', 'success')
    except Exception as e:
        flash(f"Failed to save memory: {e}", 'error')
    return redirect(url_for('memories'))


@app.route('/memories/delete/<memory_id>', methods=['POST'])
@require_auth
def memories_delete(memory_id: str):
    try:
        _api_delete(f'/api/admin/memory/text/{memory_id}')
        flash('Memory deleted.', 'success')
    except Exception as e:
        flash(f"Failed to delete memory: {e}", 'error')
    return redirect(url_for('memories'))


@app.route('/golden-queries')
@require_auth
def golden_queries():
    try:
        data = _api_get('/api/admin/golden_queries', params={'limit': 100, 'offset': 0})
        items = data.get('items', [])
        return render_template_string(GOLDEN_QUERIES_TEMPLATE, items=items)
    except Exception as e:
        flash(f"Failed to load golden queries: {e}", 'error')
        return render_template_string(GOLDEN_QUERIES_TEMPLATE, items=[])


@app.route('/golden-queries/create', methods=['POST'])
@require_auth
def golden_queries_create():
    question = request.form.get('question', '')
    sql = request.form.get('sql', '')
    try:
        _api_post('/api/admin/golden_queries', {'question': question, 'sql': sql})
        flash('Golden query saved.', 'success')
    except Exception as e:
        flash(f"Failed to save golden query: {e}", 'error')
    return redirect(url_for('golden_queries'))


@app.route('/golden-queries/delete/<memory_id>', methods=['POST'])
@require_auth
def golden_queries_delete(memory_id: str):
    try:
        _api_delete(f'/api/admin/golden_queries/{memory_id}')
        flash('Golden query deleted.', 'success')
    except Exception as e:
        flash(f"Failed to delete golden query: {e}", 'error')
    return redirect(url_for('golden_queries'))


@app.route('/health')
def health():
    return {'status': 'healthy', 'service': 'dashboard'}


if __name__ == '__main__':
    print(f"Starting Dashboard on port {DASHBOARD_PORT}")
    print(f"Access at: http://localhost:{DASHBOARD_PORT}")
    app.run(host='0.0.0.0', port=DASHBOARD_PORT, debug=False)
