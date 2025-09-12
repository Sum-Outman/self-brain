#!/usr/bin/env python3
"""
Fixed version of app.py with proper socketio configuration
"""
import os
import sys
import threading
import subprocess
import logging
from datetime import datetime
import json
import sqlite3
import shutil
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize SocketIO with proper configuration
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Ensure knowledge base directory exists
os.makedirs('knowledge_base_storage', exist_ok=True)

# ... (rest of the app code from app.py)

# Copy the optimize_knowledge_database function from app.py
@app.route('/api/knowledge/optimize', methods=['POST'])
def optimize_knowledge_database():
    """Optimize knowledge database"""
    try:
        import sqlite3
        import os
        import shutil
        
        # Check knowledge database file
        db_file = 'knowledge_base.db'
        messages = []
        
        # Debug information
        logger.info("Starting knowledge database optimization")
        
        # If database does not exist, create it
        if not os.path.exists(db_file):
            # Create database and table structure
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Create knowledge table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT,
                    content TEXT,
                    summary TEXT,
                    tags TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    views INTEGER DEFAULT 0,
                    file_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create index
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_title ON knowledge(title)')
            
            conn.commit()
            conn.close()
            messages.append("Database created successfully")
            logger.info("Knowledge database created")
        else:
            # Create backup
            backup_file = f'knowledge_base_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
            shutil.copy2(db_file, backup_file)
            messages.append(f"Backup created: {backup_file}")
            logger.info(f"Database backup created: {backup_file}")
            
            # Connect to database and perform optimization
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Execute VACUUM optimization
            cursor.execute('VACUUM')
            
            # Reindex
            cursor.execute('REINDEX')
            
            conn.commit()
            conn.close()
            messages.append("Database optimized successfully")
            logger.info("Database optimization completed")
        
        # Ensure knowledge base storage directory exists
        storage_path = 'knowledge_base_storage'
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
            messages.append("Storage directory created")
            logger.info("Storage directory created")
        else:
            # Clean empty folders in knowledge base storage
            cleaned_folders = 0
            try:
                for item in os.listdir(storage_path):
                    item_path = os.path.join(storage_path, item)
                    if os.path.isdir(item_path):
                        # Check if folder is empty
                        if not os.listdir(item_path):
                            try:
                                os.rmdir(item_path)
                                cleaned_folders += 1
                            except Exception as cleanup_error:
                                logger.warning(f"Failed to clean folder: {cleanup_error}")
                if cleaned_folders > 0:
                    messages.append(f"Cleaned {cleaned_folders} empty folders")
                    logger.info(f"Cleaned {cleaned_folders} empty folders")
            except Exception as e:
                logger.warning(f"Error cleaning storage directory: {e}")
        
        final_message = '; '.join(messages)
        logger.info(f"Knowledge database optimization completed: {final_message}")
        
        return jsonify({
            'success': True,
            'message': final_message
        })
        
    except Exception as e:
        error_msg = f"Failed to optimize knowledge database: {str(e)}"
        logger.error(error_msg)
        return jsonify({'success': False, 'message': str(e)})

# Copy all other routes from app.py
# ... (include all other necessary routes)

# Start A Manager API
threading.Thread(target=lambda: subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), 'backend', 'a_manager_api.py')]), daemon=True).start()

if __name__ == '__main__':
    logger.info("Starting Self Brain AGI System Web Interface")
    logger.info("Visit http://localhost:5000 for main page")
    
    # Run Flask app with socketio using a different approach
    # Use app.run instead of socketio.run for testing
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
