import os

# Define the absolute path to the project's root directory
# This makes the path independent of where you run the script from
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Centralized Database Path ---
# All parts of the application should use this single path for ChromaDB.
DB_PATH = os.path.join(ROOT_DIR, "chroma_db") 