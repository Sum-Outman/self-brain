#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base Expert Model (Model J) Implementation
This model contains comprehensive knowledge across various domains and supports self-learning capabilities
"""

import logging
import json
import os
import time
import threading
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import pickle
from collections import deque, defaultdict
import sqlite3

# Import core components
from manager_model.model_registry import ModelRegistry, get_model_registry
from manager_model.data_bus import DataBus, get_data_bus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KnowledgeBaseModel')

class KnowledgeBaseModel(nn.Module):
    """Knowledge Base Expert Model implementation with comprehensive domain knowledge"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.model_registry = get_model_registry()
        self.data_bus = get_data_bus()
        
        # Model configuration
        self.model_id = 'J'
        self.name = 'Knowledge Base Expert Model'
        self.description = 'Comprehensive knowledge model with expertise across various domains'
        
        # Neural network architecture
        self.hidden_size = 512  # Custom hidden size for our from-scratch implementation
        self.num_layers = 3
        self.attention_heads = 12
        
        # Initialize network layers
        self.initialize_network()
        
        # Training configuration
        self.lr = 0.0001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Knowledge base configuration
        self.knowledge_domains = [
            'physics', 'mathematics', 'chemistry', 'medicine', 'law', 
            'history', 'sociology', 'humanities', 'psychology', 'economics',
            'management', 'mechanical_engineering', 'electrical_engineering',
            'food_engineering', 'chemical_engineering'
        ]
        
        # Knowledge base storage
        self.setup_knowledge_base()
        
        # Self-learning configuration
        self.is_self_learning = False
        self.self_learning_thread = None
        self.self_learning_rate = 0.0005
        self.learning_progress = 0
        
        # Model status
        self.is_initialized = False
        self.is_training = False
        self.performance_metrics = defaultdict(float)
        
        # Cache for recent queries
        self.query_cache = deque(maxlen=1000)
        
        # Start initialization
        self.initialize()
    
    def initialize_network(self):
        """Initialize the neural network architecture"""
        # Input embedding layer (matches sentence transformer output size)
        self.input_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Main processing layers
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.2
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output layers for different domains
        self.domain_classifier = nn.Linear(self.hidden_size, len(self.knowledge_domains))
        self.response_generator = nn.Linear(self.hidden_size, self.hidden_size)
    
    def setup_knowledge_base(self):
        """Set up the knowledge base storage and retrieval system"""
        try:
            # Create directories for knowledge base
            self.kb_dir = './data/knowledge_base'
            os.makedirs(self.kb_dir, exist_ok=True)
            
            # Initialize FAISS index for efficient similarity search
            self.vector_dim = self.hidden_size
            self.index = faiss.IndexFlatL2(self.vector_dim)
            
            # From-scratch embedding generator
            self.embedding_model = self.generate_embeddings_from_scratch
            
            # Set up SQLite database for metadata
            self.db_path = os.path.join(self.kb_dir, 'knowledge_metadata.db')
            self.setup_database()
            
            # Load existing knowledge if available
            self.load_knowledge_base()
            
            logger.info(f"Knowledge base setup complete with {len(self.knowledge_domains)} domains")
            
        except Exception as e:
            logger.error(f"Failed to set up knowledge base: {str(e)}")
    
    def setup_database(self):
        """Set up SQLite database for knowledge metadata"""
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table for knowledge entries
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                domain TEXT NOT NULL,
                source TEXT,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create table for domain statistics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_stats (
                domain TEXT PRIMARY KEY,
                entry_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Initialize domain stats if not exists
            for domain in self.knowledge_domains:
                cursor.execute(
                    "INSERT OR IGNORE INTO domain_stats (domain, entry_count) VALUES (?, ?)",
                    (domain, 0)
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to set up database: {str(e)}")
    
    def initialize(self):
        """Initialize the knowledge base model"""
        try:
            logger.info(f"Initializing {self.name}...")
            
            # Load pre-trained weights if available
            self.load_weights()
            
            # Register message handlers
            self.register_handlers()
            
            # Register with model registry
            self.model_registry.register_model({
                'id': self.model_id,
                'name': self.name,
                'description': self.description,
                'type': 'knowledge',
                'domains': self.knowledge_domains
            })
            
            self.is_initialized = True
            logger.info(f"{self.name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {str(e)}")
            self.is_initialized = False
    
    def generate_embeddings_from_scratch(self, texts):
        """Generate embeddings from scratch without using pre-trained models
        This is a from-scratch implementation of text embedding generation
        """
        embeddings = []
        for text in texts:
            # Simple but effective character-based embedding
            # In a production system, this would be replaced with a more sophisticated method
            # that considers word meanings and context
            embedding = np.zeros(self.vector_dim, dtype='float32')
            
            # Character-level encoding
            for i, char in enumerate(text[:self.vector_dim]):
                # Use character ordinal value and apply hashing to distribute across dimensions
                char_val = ord(char) % 256  # Normalize to 0-255
                # Distribute character information across multiple dimensions
                for j in range(min(4, self.vector_dim - i)):
                    pos = (i * 4 + j) % self.vector_dim
                    embedding[pos] += char_val / 255.0  # Normalize to 0-1
            
            # Simple normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype='float32')
    
    def register_handlers(self):
        """Register message handlers for incoming requests"""
        # Register handler for model requests
        self.data_bus.subscribe(f"model_{self.model_id}_request", self.handle_request)
        
        # Register handler for knowledge updates
        self.data_bus.subscribe("knowledge_update", self.handle_knowledge_update)
        
        # Register handler for self-learning control
        self.data_bus.subscribe("self_learning_control", self.handle_self_learning_control)
    
    def handle_request(self, data):
        """Handle incoming requests from other models or the main management model"""
        try:
            # Extract request details
            request_id = data.get('request_id', '')
            source = data.get('source', '')
            message = data.get('message', '')
            context = data.get('context', [])
            session_id = data.get('session_id', 'default')
            
            if not message:
                return
            
            logger.info(f"Received knowledge request: {message[:50]}... from {source}")
            
            # Process the request
            response = self.process_request(message, context)
            
            # Send the response back to the requester
            response_channel = f"model_{self.model_id}_response_{request_id}"
            self.data_bus.publish(response_channel, {
                'model_id': self.model_id,
                'request_id': request_id,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Log the interaction
            self.log_interaction(session_id, message, response)
            
        except Exception as e:
            logger.error(f"Error handling knowledge request: {str(e)}")
            # Send error response
            request_id = data.get('request_id', '')
            response_channel = f"model_{self.model_id}_response_{request_id}"
            self.data_bus.publish(response_channel, {
                'model_id': self.model_id,
                'request_id': request_id,
                'response': f"I encountered an error processing your knowledge request: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    def process_request(self, query, context=[]):
        """Process a knowledge query and generate a response"""
        try:
            # Check cache first
            cache_key = (query, str(context[-3:] if context else []))
            for cached_query, cached_context, cached_response in self.query_cache:
                if cached_query == query and cached_context == cache_key[1]:
                    logger.debug("Returning cached response")
                    return cached_response
            
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            
            # Find relevant knowledge in the knowledge base
            relevant_knowledge = self.search_knowledge_base(query_embedding, query)
            
            # Generate response based on relevant knowledge
            response = self.generate_response(query, relevant_knowledge)
            
            # Cache the response
            self.query_cache.append((query, cache_key[1], response))
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing knowledge query: {str(e)}")
            return f"I couldn't find the information you're looking for. Error: {str(e)}"
    
    def generate_embedding(self, text):
        """Generate embedding for the given text using our from-scratch implementation"""
        try:
            # Directly use our from-scratch embedding generator
            embedding = self.generate_embeddings_from_scratch([text])[0]
            
            # Ensure embedding is float32 and has the correct dimensions
            embedding = np.array(embedding, dtype='float32')
            if len(embedding) < self.vector_dim:
                embedding = np.pad(embedding, (0, self.vector_dim - len(embedding)), 'constant')
            else:
                embedding = embedding[:self.vector_dim]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return random embedding as fallback
            return np.random.rand(self.vector_dim).astype('float32')
    
    def search_knowledge_base(self, query_embedding, query_text, k=5):
        """Search the knowledge base for relevant information"""
        try:
            # Check if index has any vectors
            if self.index.ntotal == 0:
                logger.warning("Knowledge base is empty")
                return []
            
            # Search for similar vectors
            distances, indices = self.index.search(np.array([query_embedding]), k)
            
            # Retrieve metadata for the top results
            relevant_entries = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0:  # Ensure index is valid
                    cursor.execute("SELECT content, domain, source FROM knowledge_entries WHERE embedding_id = ?", (int(idx),))
                    result = cursor.fetchone()
                    if result:
                        relevant_entries.append({
                            'content': result[0],
                            'domain': result[1],
                            'source': result[2],
                            'similarity': 1.0 - distances[0][i]  # Convert distance to similarity
                        })
            
            conn.close()
            
            # Sort by similarity
            relevant_entries.sort(key=lambda x: x['similarity'], reverse=True)
            
            return relevant_entries[:3]  # Return top 3 most relevant entries
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def generate_response(self, query, knowledge_entries):
        """Generate a response based on the query and relevant knowledge"""
        try:
            if not knowledge_entries:
                return "I don't have specific information about that topic in my knowledge base."
            
            # Extract content from knowledge entries
            knowledge_content = " ".join([entry['content'] for entry in knowledge_entries])
            
            # Create prompt for response generation
            prompt = f"Based on the following information, answer the question:\n"
            prompt += f"Information: {knowledge_content[:1000]}...\n"  # Limit to 1000 characters
            prompt += f"Question: {query}\n"
            prompt += "Answer:"
            
            # Use the model to generate a response
            response = self.generate_text_response(prompt)
            
            # If response is too short or not helpful, use fallback
            if not response or len(response.strip()) < 10:
                response = f"Based on my knowledge, {self.summarize_knowledge(knowledge_entries)}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I couldn't generate a proper response based on the available knowledge."
    
    def generate_text_response(self, prompt):
        """Generate a text response using the model"""
        try:
            # Convert prompt to tensor
            prompt_tensor = torch.tensor([ord(c) for c in prompt[:200]]).float().unsqueeze(0)
            
            # Pad to embedding size
            if len(prompt_tensor[0]) < self.hidden_size:
                padding = torch.zeros((1, self.hidden_size - len(prompt_tensor[0])))
                prompt_tensor = torch.cat([prompt_tensor, padding], dim=1)
            else:
                prompt_tensor = prompt_tensor[:, :self.hidden_size]
            
            # Forward pass through the network
            with torch.no_grad():
                embedded = self.input_projection(prompt_tensor)
                output = self.transformer_encoder(embedded.unsqueeze(0))
                generated = self.response_generator(output.squeeze(0))
            
            # Convert output back to text (simplified approach)
            # In a real implementation, this would use a proper text generation method
            response = "This is a generated response based on the knowledge base. In a production system, this would be a more sophisticated text generation based on the query and relevant knowledge."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return ""
    
    def summarize_knowledge(self, knowledge_entries):
        """Create a summary of the relevant knowledge entries"""
        try:
            # Extract key points from the entries
            key_points = []
            for entry in knowledge_entries:
                # Simple summarization by taking the first sentence
                sentences = entry['content'].split('. ')
                if sentences:
                    key_points.append(sentences[0])
            
            # Combine key points
            summary = ". ".join(key_points[:3]) + "."
            
            # Add domain information
            domains = set([entry['domain'] for entry in knowledge_entries])
            if domains:
                summary += f" This information is from the {', '.join(domains)} domain(s)."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing knowledge: {str(e)}")
            return ""
    
    def handle_knowledge_update(self, data):
        """Handle updates to the knowledge base"""
        try:
            content = data.get('content', '')
            domain = data.get('domain', 'general')
            source = data.get('source', 'unknown')
            
            if not content:
                return
            
            # Add the new knowledge to the base
            self.add_knowledge(content, domain, source)
            
            logger.info(f"Added new knowledge to domain: {domain}")
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
    
    def add_knowledge(self, content, domain, source='unknown'):
        """Add new knowledge to the knowledge base"""
        try:
            # Generate embedding for the content
            embedding = self.generate_embedding(content)
            
            # Add to FAISS index
            embedding_id = self.index.ntotal
            self.index.add(np.array([embedding]))
            
            # Store metadata in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert knowledge entry
            cursor.execute(
                "INSERT INTO knowledge_entries (content, domain, source, embedding_id) VALUES (?, ?, ?, ?)",
                (content, domain, source, embedding_id)
            )
            
            # Update domain stats
            cursor.execute(
                "UPDATE domain_stats SET entry_count = entry_count + 1, last_updated = CURRENT_TIMESTAMP WHERE domain = ?",
                (domain,)
            )
            
            conn.commit()
            conn.close()
            
            # Save the updated knowledge base
            self.save_knowledge_base()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {str(e)}")
            return False
    
    def handle_self_learning_control(self, data):
        """Handle self-learning control commands"""
        action = data.get('action', '').lower()
        
        if action == 'start':
            self.start_self_learning()
        elif action == 'stop':
            self.stop_self_learning()
        elif action == 'status':
            # Publish current self-learning status
            self.data_bus.publish("self_learning_status", {
                'model_id': self.model_id,
                'is_learning': self.is_self_learning,
                'progress': self.learning_progress,
                'rate': self.self_learning_rate
            })
    
    def start_self_learning(self):
        """Start the self-learning process"""
        if self.is_self_learning:
            logger.info("Self-learning is already running")
            return
        
        logger.info("Starting self-learning process")
        self.is_self_learning = True
        self.learning_progress = 0
        
        # Start self-learning in a separate thread
        self.self_learning_thread = threading.Thread(target=self.self_learning_loop)
        self.self_learning_thread.daemon = True
        self.self_learning_thread.start()
    
    def stop_self_learning(self):
        """Stop the self-learning process"""
        if not self.is_self_learning:
            logger.info("Self-learning is not running")
            return
        
        logger.info("Stopping self-learning process")
        self.is_self_learning = False
        
        if self.self_learning_thread and self.self_learning_thread.is_alive():
            self.self_learning_thread.join(timeout=5.0)
            
        self.learning_progress = 0
    
    def self_learning_loop(self):
        """Main loop for self-learning"""
        try:
            while self.is_self_learning:
                # Get training data from existing knowledge
                training_data = self.prepare_self_learning_data()
                
                if training_data:
                    # Perform a self-learning iteration
                    self.self_learning_iteration(training_data)
                    
                    # Update progress
                    self.learning_progress = min(100, self.learning_progress + 1)
                    
                    # Publish progress update
                    self.data_bus.publish("self_learning_progress", {
                        'model_id': self.model_id,
                        'progress': self.learning_progress
                    })
                
                # Sleep for a short interval
                time.sleep(10)  # Adjust based on desired learning frequency
                
        except Exception as e:
            logger.error(f"Error in self-learning loop: {str(e)}")
            self.is_self_learning = False
    
    def prepare_self_learning_data(self, sample_size=10):
        """Prepare data for self-learning from existing knowledge"""
        try:
            # Query random knowledge entries
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT content, domain FROM knowledge_entries ORDER BY RANDOM() LIMIT ?", (sample_size,))
            entries = cursor.fetchall()
            
            conn.close()
            
            if not entries:
                return []
            
            # Prepare training data
            training_data = []
            for content, domain in entries:
                # Create a pseudo-query related to the content
                query = f"Explain {content[:50]}..."
                
                training_data.append({
                    'input': query,
                    'target': content,
                    'domain': domain
                })
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing self-learning data: {str(e)}")
            return []
    
    def self_learning_iteration(self, training_data):
        """Perform one iteration of self-learning"""
        try:
            # Prepare training tensors
            inputs = []
            targets = []
            
            for item in training_data:
                # Generate embeddings for input and target
                input_embedding = torch.tensor(self.generate_embedding(item['input'])).float()
                target_embedding = torch.tensor(self.generate_embedding(item['target'])).float()
                
                inputs.append(input_embedding)
                targets.append(target_embedding)
            
            # Convert to tensors
            input_tensor = torch.stack(inputs)
            target_tensor = torch.stack(targets)
            
            # Forward pass
            output = self(input_tensor)
            loss = self.criterion(output, target_tensor)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log learning progress
            logger.debug(f"Self-learning iteration completed. Loss: {loss.item():.4f}")
            
            # Periodically save weights during self-learning
            if self.learning_progress % 20 == 0:
                self.save_weights()
                
        except Exception as e:
            logger.error(f"Error in self-learning iteration: {str(e)}")
    
    def train(self, training_data, epochs=5, batch_size=16, learning_rate=0.0001):
        """Train the knowledge base model with external data"""
        try:
            self.is_training = True
            self.training_progress = 0
            self.lr = learning_rate
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            
            # Prepare training data
            inputs, targets = self.prepare_training_data(training_data)
            
            # Training loop
            for epoch in range(epochs):
                # Shuffle data
                permutation = torch.randperm(inputs.size()[0])
                inputs = inputs[permutation]
                targets = targets[permutation]
                
                # Mini-batch training
                for i in range(0, inputs.size()[0], batch_size):
                    batch_inputs = inputs[i:i+batch_size]
                    batch_targets = targets[i:i+batch_size]
                    
                    # Forward pass
                    outputs = self(batch_inputs)
                    loss = self.criterion(outputs, batch_targets)
                    
                    # Backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Update progress
                self.training_progress = int(((epoch + 1) / epochs) * 100)
                
                # Log progress
                logger.info(f"Training epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
                
                # Publish progress update
                self.data_bus.publish("training_progress", {
                    'model_id': self.model_id,
                    'progress': self.training_progress,
                    'epoch': epoch + 1,
                    'loss': loss.item()
                })
            
            # Save trained weights
            self.save_weights()
            
            self.is_training = False
            return {
                'status': 'success',
                'message': f'Training completed successfully in {epochs} epochs',
                'loss': loss.item()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.is_training = False
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def prepare_training_data(self, training_data):
        """Prepare training data for the model"""
        inputs = []
        targets = []
        
        for item in training_data:
            # Generate embeddings for input and target
            input_embedding = torch.tensor(self.generate_embedding(item['input'])).float()
            target_embedding = torch.tensor(self.generate_embedding(item['target'])).float()
            
            inputs.append(input_embedding)
            targets.append(target_embedding)
        
        # Convert to tensors
        input_tensor = torch.stack(inputs)
        target_tensor = torch.stack(targets)
        
        return input_tensor, target_tensor
    
    def forward(self, x):
        """Forward pass of the neural network"""
        projected = self.input_projection(x)
        output = self.transformer_encoder(projected.unsqueeze(0) if len(x.shape) == 2 else projected)
        return self.response_generator(output.squeeze(0) if len(output.shape) == 3 else output)
    
    def save_weights(self, path=None):
        """Save the model weights"""
        try:
            if path is None:
                path = f'./models/knowledge_base_model_weights.pth'
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            torch.save(self.state_dict(), path)
            logger.info(f"Model weights saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model weights: {str(e)}")
            return False
    
    def load_weights(self, path=None):
        """Load model weights from file"""
        try:
            if path is None:
                path = f'./models/knowledge_base_model_weights.pth'
            
            if os.path.exists(path):
                self.load_state_dict(torch.load(path))
                logger.info(f"Model weights loaded from {path}")
                return True
            else:
                logger.info(f"No pre-trained weights found at {path}, using randomly initialized weights")
                return False
        except Exception as e:
            logger.error(f"Failed to load model weights: {str(e)}")
            return False
    
    def save_knowledge_base(self):
        """Save the knowledge base to disk"""
        try:
            # Save FAISS index
            index_path = os.path.join(self.kb_dir, 'knowledge_index.faiss')
            faiss.write_index(self.index, index_path)
            
            logger.info(f"Knowledge base index saved to {index_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {str(e)}")
            return False
    
    def load_knowledge_base(self):
        """Load the knowledge base from disk"""
        try:
            # Load FAISS index
            index_path = os.path.join(self.kb_dir, 'knowledge_index.faiss')
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"Knowledge base index loaded from {index_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            return False
    
    def log_interaction(self, session_id, query, response):
        """Log user interactions for future learning"""
        try:
            # This would be expanded to log interactions to a database or file
            # For now, we'll just log to the console
            logger.debug(f"Interaction logged: Session {session_id}, Query: {query[:30]}..., Response: {response[:30]}...")
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")

# Singleton instance of the knowledge base model
global_knowledge_model = None

def get_knowledge_model():
    """Get the singleton instance of the knowledge base model"""
    global global_knowledge_model
    if global_knowledge_model is None:
        global_knowledge_model = KnowledgeBaseModel()
    return global_knowledge_model

# Initialize the model when this module is loaded
if __name__ == "__main__":
    # For testing purposes
    kb_model = KnowledgeBaseModel()
    print("Knowledge Base Model initialized successfully")
    
    # Example: Add some test knowledge
    kb_model.add_knowledge(
        "Newton's laws of motion are three basic laws of classical mechanics that describe the relationship between the motion of an object and the forces acting on it.",
        "physics",
        "test_source"
    )
    
    # Example query
    response = kb_model.process_request("What are Newton's laws of motion?")
    print(f"Response: {response}")