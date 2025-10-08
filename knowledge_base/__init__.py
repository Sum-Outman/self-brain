# Self Brain Knowledge Base
# Author: silencecrowtom@qq.com
# This module provides the core functionality for the knowledge base system

import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SelfBrainKnowledgeBase')

class KnowledgeBase:
    """Base class for the knowledge base system"""
    
    def __init__(self, base_path='d:\\shiyan\\knowledge_base'):
        """Initialize the knowledge base"""
        self.base_path = base_path
        self.knowledge_stores = {
            'general': os.path.join(base_path, 'general_knowledge'),
            'scientific': os.path.join(base_path, 'scientific_knowledge'),
            'technical': os.path.join(base_path, 'technical_knowledge'),
            'medical': os.path.join(base_path, 'medical_knowledge'),
            'legal': os.path.join(base_path, 'legal_knowledge'),
            'historical': os.path.join(base_path, 'historical_knowledge'),
            'social': os.path.join(base_path, 'social_knowledge'),
            'psychological': os.path.join(base_path, 'psychological_knowledge'),
            'economic': os.path.join(base_path, 'economic_knowledge'),
            'management': os.path.join(base_path, 'management_knowledge'),
            'engineering': os.path.join(base_path, 'engineering_knowledge')
        }
        
        # Create directories if they don't exist
        for store_path in self.knowledge_stores.values():
            os.makedirs(store_path, exist_ok=True)
            
        # Create metadata file
        self.metadata_file = os.path.join(base_path, 'metadata.json')
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'total_entries': 0
                }, f, indent=2)
        
    def add_knowledge(self, domain, content, metadata=None):
        """Add knowledge to the specified domain"""
        if domain not in self.knowledge_stores:
            logger.error(f"Domain {domain} not found")
            return False
            
        try:
            # Generate a unique ID
            knowledge_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            knowledge_file = os.path.join(self.knowledge_stores[domain], f"{knowledge_id}.json")
            
            # Prepare data
            knowledge_data = {
                'id': knowledge_id,
                'domain': domain,
                'content': content,
                'metadata': metadata or {},
                'created_at': datetime.now().isoformat(),
                'last_accessed': None
            }
            
            # Write to file
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            self._update_metadata()
            logger.info(f"Added knowledge to domain {domain}")
            return True
        except Exception as e:
            logger.error(f"Failed to add knowledge: {str(e)}")
            return False
            
    def search_knowledge(self, domain, query):
        """Search for knowledge in the specified domain"""
        if domain not in self.knowledge_stores:
            logger.error(f"Domain {domain} not found")
            return []
            
        results = []
        domain_path = self.knowledge_stores[domain]
        
        try:
            for filename in os.listdir(domain_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(domain_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        knowledge_data = json.load(f)
                        
                        # Simple text search
                        if query.lower() in knowledge_data['content'].lower():
                            results.append(knowledge_data)
                            # Update last accessed time
                            knowledge_data['last_accessed'] = datetime.now().isoformat()
                            with open(file_path, 'w', encoding='utf-8') as f_update:
                                json.dump(knowledge_data, f_update, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to search knowledge: {str(e)}")
            
        return results
        
    def get_domain_knowledge(self, domain):
        """Get all knowledge from the specified domain"""
        if domain not in self.knowledge_stores:
            logger.error(f"Domain {domain} not found")
            return []
            
        results = []
        domain_path = self.knowledge_stores[domain]
        
        try:
            for filename in os.listdir(domain_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(domain_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results.append(json.load(f))
        except Exception as e:
            logger.error(f"Failed to get domain knowledge: {str(e)}")
            
        return results
        
    def _update_metadata(self):
        """Update the metadata file"""
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                
            metadata['last_updated'] = datetime.now().isoformat()
            
            # Count total entries
            total_entries = 0
            for store_path in self.knowledge_stores.values():
                total_entries += len([f for f in os.listdir(store_path) if f.endswith('.json')])
                
            metadata['total_entries'] = total_entries
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")

# Create a global instance for the knowledge base
global_knowledge_base = KnowledgeBase()

# Helper functions for the API
def add_knowledge_entry(domain, content, metadata=None):
    """Add a knowledge entry"""
    return global_knowledge_base.add_knowledge(domain, content, metadata)

def search_knowledge_entries(domain, query):
    """Search knowledge entries"""
    return global_knowledge_base.search_knowledge(domain, query)

def get_all_knowledge_for_domain(domain):
    """Get all knowledge entries for a domain"""
    return global_knowledge_base.get_domain_knowledge(domain)