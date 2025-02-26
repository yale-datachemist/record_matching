import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import logging
import regex as re
import os
import json
import time
import ast
import pickle
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Any, Optional, Union

logging.basicConfig(level=logging.INFO, filename="output.log",filemode="w")

logger = logging.getLogger(__name__)

# For Qdrant vector database
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    logging.warning("Qdrant client not available. Installing required packages...")
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "qdrant-client"])
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        QDRANT_AVAILABLE = True
        logging.info("Qdrant client installed successfully.")
    except Exception as e:
        logging.error(f"Failed to install Qdrant client: {e}")
        QDRANT_AVAILABLE = False

# For Levenshtein distance calculation
try:
    from Levenshtein import distance
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    logging.warning("Levenshtein package not available. Installing...")
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "python-Levenshtein"])
        from Levenshtein import distance
        LEVENSHTEIN_AVAILABLE = True
        logging.info("Levenshtein package installed successfully.")
    except Exception as e:
        logging.error(f"Failed to install Levenshtein: {e}")
        LEVENSHTEIN_AVAILABLE = False



# Reduce Qdrant client logging
logging.getLogger('qdrant_client').setLevel(logging.WARNING)

class VectorEnhancedEntityResolver:
    """
    An entity resolver for library catalog data that leverages vector embeddings
    for improved disambiguation of entities with identical names but different domains.
    """
    
    def __init__(self, 
                 min_similarity: float = 0.7, 
                 vector_dim: int = 3072,
                 use_qdrant: bool = True):
        """
        Initialize the resolver with vector enhancement capability.
        
        Args:
            min_similarity: Minimum similarity threshold for entity matching
            vector_dim: Dimension of the vector embeddings
            use_qdrant: Whether to use Qdrant for vector search (if available)
        """
        self.min_similarity = min_similarity
        self.vector_dim = vector_dim
        self.is_trained = False
        self.year_pattern = re.compile(r'(\d{4})-(\d{4}|\s*)')
        self.entity_clusters = None
        self.qdrant_client = None
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE
        
        # Initialize metric weights with defaults
        self.metric_weights = {
            'name_similarity': 0.3,
            'vector_similarity': 0.3,
            'context_similarity': 0.15,
            'temporal_similarity': 0.15,  # Added weight for temporal similarity
            'domain_similarity': 0.1
        }
        
        logging.info(f"Initialized Vector-Enhanced Entity Resolver "
                    f"(vector_dim={vector_dim}, min_similarity={min_similarity}, "
                    f"use_qdrant={self.use_qdrant})")
        
        # Configure enhanced logging
        self.verbose_logging = True
        if self.verbose_logging:
            # Set up a file handler if not already configured
            if not any(isinstance(h, logging.FileHandler) for h in logging.root.handlers):
                file_handler = logging.FileHandler("entity_resolver.log")
                file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
                logging.root.addHandler(file_handler)
            
            # Ensure we're at INFO level
            logging.root.setLevel(logging.INFO)
            logging.info("=== Starting New Entity Resolution Session ===")
    
    def _initialize_qdrant(self, df: pd.DataFrame, vector_df: pd.DataFrame) -> None:
        """
        Initialize Qdrant collection with person name vectors for semantic blocking.
        
        Args:
            df: Preprocessed DataFrame
            vector_df: DataFrame with vector embeddings
        """
        if not QDRANT_AVAILABLE:
            logging.warning("Qdrant not available, skipping vector indexing")
            return
        
        try:
            # Try to connect to local Qdrant instance first
            try:
                logging.info("Attempting to connect to local Qdrant instance...")
                self.qdrant_client = QdrantClient(host="localhost", port=6333)
                # Test the connection
                collections = self.qdrant_client.get_collections().collections
                logging.info(f"Connected to local Qdrant instance with {len(collections)} collections")
            except Exception as local_error:
                # Fall back to in-memory Qdrant if local connection fails
                logging.warning(f"Could not connect to local Qdrant: {local_error}")
                logging.info("Using in-memory Qdrant instance instead")
                self.qdrant_client = QdrantClient(":memory:")
            
            # Check if collection exists and delete if it does
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if "entity_vectors" in collection_names:
                logging.info("Deleting existing entity_vectors collection")
                self.qdrant_client.delete_collection("entity_vectors")
            
            # Create a new collection with HNSW index for faster approximate search
            logging.info(f"Creating Qdrant collection with vector dimension {self.vector_dim}")
            self.qdrant_client.create_collection(
                collection_name="entity_vectors",
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100
                )
            )
            
            # Count records with vectors
            record_count = 0
            vector_count = 0
            
            # Prepare vectors for insertion - use person vectors directly for blocking
            points = []
            for idx, row in df.iterrows():
                record_count += 1
                record_id = row['id']
                
                if record_id in vector_df.index:
                    vector_count += 1
                    
                    # Get the person vector from vector_df - this is the key change!
                    # We're now using the person vector directly from the vector dataset
                    try:
                        person_vector = None
                        # Check if 'person' column exists in vector_df
                        if 'person' in vector_df.columns:
                            person_val = vector_df.loc[record_id, 'person']
                            if isinstance(person_val, str) and '[' in person_val:
                                # Parse string representation of vector
                                import ast
                                person_vector = np.array(ast.literal_eval(person_val), dtype=np.float32)
                            elif isinstance(person_val, np.ndarray):
                                person_vector = person_val
                        
                        # If no person vector available, use the full vector as fallback
                        if person_vector is None:
                            person_vector = vector_df.loc[record_id].values.astype(np.float32)
                            
                        # Create point with more metadata
                        points.append(
                            models.PointStruct(
                                id=idx,
                                vector=person_vector.tolist(),
                                payload={
                                    "record_id": record_id,
                                    "person": row['person'],
                                    "normalized_name": row.get('normalized_name', ''),
                                    "birth_year": int(row['birth_year']) if not pd.isna(row.get('birth_year')) else None,
                                    "death_year": int(row['death_year']) if not pd.isna(row.get('death_year')) else None,
                                    "domains": row.get('domain_context', ''),
                                    "subjects": row.get('subjects', ''),
                                    "genres": row.get('genres', '')
                                }
                            )
                        )
                    except Exception as e:
                        logging.warning(f"Error processing person vector for record {record_id}: {e}")
            
            # Insert vectors in batches with progress logging
            batch_size = 100
            logging.info(f"Inserting {len(points)} person vectors into Qdrant (out of {record_count} records)")
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.qdrant_client.upsert(
                    collection_name="entity_vectors",
                    points=batch
                )
                if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(points):
                    logging.info(f"Inserted {min(i + batch_size, len(points))}/{len(points)} vectors")
            
            logging.info(f"Successfully created Qdrant collection with {len(points)} person vectors "
                        f"({vector_count}/{record_count} records, {vector_count/record_count:.1%})")
            
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant: {e}")
            self.qdrant_client = None

    def preprocess_data(self, 
                   df: pd.DataFrame, 
                   vector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess data with vector enhancement if available.
        
        Args:
            df: DataFrame with entity records
            vector_df: Optional DataFrame with vector embeddings
                
        Returns:
            Preprocessed DataFrame
        """
        logging.info(f"Preprocessing {len(df)} records" + 
                    f" with vector data: {vector_df is not None}")
        
        df_clean = df.copy()
        
        # Check which fields are missing - useful for understanding data quality
        null_counts = df_clean.isnull().sum()
        logging.info(f"Null value counts: {null_counts.to_dict()}")
        
        # Track missing fields for confidence calculation
        important_fields = ['attribution', 'subjects', 'genres', 'relatedWork', 'provision']
        df_clean['null_field_count'] = df_clean[important_fields].isnull().sum(axis=1)
        
        # Extract birth and death years from person names with improved regex
        # This is critical for temporal matching and disambiguation
        df_clean['birth_year'] = None
        df_clean['death_year'] = None
        df_clean['name_without_dates'] = df_clean['person'].copy()
        df_clean['has_life_dates'] = False
        
        # Improved date extraction with logging of matches
        date_extraction_count = 0
        for idx, person in enumerate(df_clean['person']):
            if pd.isna(person):
                continue
                
            # Try multiple date patterns to increase extraction success
            # Primary pattern: Name, FirstName, YYYY-YYYY
            life_dates = self.year_pattern.search(person)
            
            # Alternative pattern for different date formats (e.g., b. 1797 d. 1828)
            alt_pattern = re.search(r'b\.\s*(\d{4}).*?d\.\s*(\d{4})', person)
            
            if life_dates:
                # Extract birth and death years if available
                birth = life_dates.group(1)
                death = life_dates.group(2).strip()
                
                # Store as integers for better comparison
                df_clean.at[idx, 'birth_year'] = int(birth) if birth else None
                df_clean.at[idx, 'death_year'] = int(death) if death and death.strip() else None
                df_clean.at[idx, 'has_life_dates'] = True
                
                # Store name without the dates
                clean_name = re.sub(r',\s*\d{4}-\d{4}|\d{4}-\d{4}|\d{4}-', '', person).strip()
                df_clean.at[idx, 'name_without_dates'] = clean_name
                
                date_extraction_count += 1
            elif alt_pattern:
                # Alternative date format extraction
                birth = alt_pattern.group(1)
                death = alt_pattern.group(2)
                
                df_clean.at[idx, 'birth_year'] = int(birth) if birth else None
                df_clean.at[idx, 'death_year'] = int(death) if death else None
                df_clean.at[idx, 'has_life_dates'] = True
                
                # Clean the name part
                clean_name = re.sub(r'b\.\s*\d{4}.*?d\.\s*\d{4}', '', person).strip()
                df_clean.at[idx, 'name_without_dates'] = clean_name
                
                date_extraction_count += 1
        
        # Log date extraction results
        logging.info(f"Extracted life dates from {date_extraction_count}/{len(df_clean)} person names "
                    f"({date_extraction_count/len(df_clean):.1%})")
        
        # Extract publication years from provision information or title if provision is null
        df_clean['pub_year'] = None
        pub_year_count = 0
        
        for idx, row in df_clean.iterrows():
            years = []
            # Try provision field first
            if not pd.isna(row['provision']):
                years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', row['provision'])  # Years between 1000-2029
            
            # Fall back to title if needed
            if not years and not pd.isna(row['title']):
                years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', row['title'])
                
            if years:
                # Take the earliest year found as the publication year
                # (handles cases where multiple years are mentioned)
                df_clean.at[idx, 'pub_year'] = int(min(years))
                pub_year_count += 1
        
        logging.info(f"Extracted publication years from {pub_year_count}/{len(df_clean)} records "
                    f"({pub_year_count/len(df_clean):.1%})")
        
        # Normalize names for comparison - critical for matching
        df_clean['normalized_name'] = df_clean['name_without_dates'].apply(self._normalize_name)
        
        # Log name normalization results - check for potential issues
        sample_names = df_clean.sample(min(5, len(df_clean)))
        for _, row in sample_names.iterrows():
            logging.info(f"Name normalization: '{row['name_without_dates']}' -> '{row['normalized_name']}'")
        
        # Create consolidated context fields with guaranteed non-null values
        df_clean['context'] = df_clean.apply(
            lambda r: f"{r['person']} {r['title']} {r['roles']} " + 
                    f"{r['subjects'] if not pd.isna(r['subjects']) else ''} " +
                    f"{r['genres'] if not pd.isna(r['genres']) else ''} " +
                    f"{r['relatedWork'] if not pd.isna(r['relatedWork']) else ''}",
            axis=1
        )
        
        # Create context from guaranteed fields
        df_clean['primary_context'] = df_clean.apply(
            lambda r: f"{r['person']} {r['title']} {r['roles']}",
            axis=1
        )
        
        # Extract domain context (crucial for disambiguation)
        df_clean['domain_context'] = df_clean.apply(
            lambda r: f"{r['subjects'] if not pd.isna(r['subjects']) else ''} " +
                    f"{r['genres'] if not pd.isna(r['genres']) else ''}",
            axis=1
        )
        
        # Calculate context completeness score
        df_clean['context_completeness'] = 1.0 - (df_clean['null_field_count'] / len(important_fields))
        
        # Extract entity mentions from subjects and titles
        df_clean['entity_mentions'] = None
        entity_mentions_count = 0
        
        for idx, row in df_clean.iterrows():
            mentions = set()
            
            # Look for capitalized phrases in guaranteed non-null fields
            for field in ['title', 'roles']:
                if not pd.isna(row[field]) and row[field]:
                    pattern = r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
                    matches = re.findall(pattern, row[field])
                    mentions.update([m.strip() for m in matches if len(m) > 3])
            
            # Then check optional fields if available
            if not pd.isna(row['subjects']):
                pattern = r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
                matches = re.findall(pattern, row['subjects'])
                mentions.update([m.strip() for m in matches if len(m) > 3])
            
            if mentions:
                df_clean.at[idx, 'entity_mentions'] = ';'.join(sorted(mentions))
                entity_mentions_count += 1
        
        logging.info(f"Extracted entity mentions from {entity_mentions_count}/{len(df_clean)} records "
                    f"({entity_mentions_count/len(df_clean):.1%})")
        
        # Add vector availability flag if vector data is provided
        df_clean['has_vector'] = False
        
        if vector_df is not None and len(vector_df) > 0:
            # Check if IDs are in vector data
            vector_count = 0
            for idx, row in df_clean.iterrows():
                if row['id'] in vector_df.index:
                    df_clean.at[idx, 'has_vector'] = True
                    vector_count += 1
            
            logging.info(f"Found vectors for {vector_count}/{len(df_clean)} records "
                        f"({vector_count/len(df_clean):.1%})")
        
        # Add more preprocessing steps for domains
        # Extract primary domains from subjects field
        df_clean['primary_domain'] = None
        
        for idx, row in df_clean.iterrows():
            if pd.isna(row['subjects']):
                continue
                
            subjects = row['subjects'].lower()
            
            # Define domain categories to check for
            # domains = {
            #     "general_works": [
            #         "general works", "encyclopedia", "almanac", "reference", "catalog", "yearbook", "bibliography"
            #     ],
            #     "philosophy_psychology_religion": [
            #         "philosophy", "philosophical", "psychology", "psyche", "cognitive science",
            #         "religion", "religious", "theology", "theological", "bible", "church", "ethics",
            #         "metaphysics", "logic", "epistemology", "buddhism", "christianity", "islam",
            #         "judaism", "hinduism", "spirituality"
            #     ],
            #     "auxiliary_sciences_of_history": [
            #         "archaeology", "archaeological", "paleography", "genealogy", "chronology",
            #         "diplomatics", "numismatics", "epigraphy"
            #     ],
            #     "world_history": [
            #         "world history", "civilizations", "ancient", "medieval", "modern", "renaissance",
            #         "revolution", "historical events", "historical figures", "cultural history"
            #     ],
            #     "history_of_the_americas": [
            #         "history of the americas", "colonial period", "american revolution", "civil war",
            #         "indigenous history", "latin american history"
            #     ],
            #     "geography_anthropology_recreation": [
            #         "geography", "geographical", "cartography", "maps", "anthropology", "ethnology",
            #         "cultural studies", "recreation", "travel", "tourism", "sports", "hiking"
            #     ],
            #     "social_sciences": [
            #         "social sciences", "sociology", "demography", "social research",
            #         "economics", "economic theory", "macroeconomics", "microeconomics",
            #         "statistics", "social issues"
            #     ],
            #     "political_science": [
            #         "political science", "political theory", "government", "governance",
            #         "international relations", "public administration", "public policy",
            #         "comparative politics"
            #     ],
            #     "law": [
            #         "law", "legal system", "constitutional law", "criminal law", "civil law",
            #         "jurisprudence", "international law", "intellectual property", "legal ethics"
            #     ],
            #     "education": [
            #         "education", "pedagogy", "teaching", "learning", "curriculum",
            #         "school", "university", "educational psychology", "educational policy"
            #     ],
            #     "music": [
            #         "music", "composition", "sonata", "symphony", "opera", "musical", "musician",
            #         "composer", "piano", "violin", "orchestra", "concert"
            #     ],
            #     "fine_arts": [
            #         "fine arts", "art", "painting", "sculpture", "artist", "portrait",
            #         "drawing", "photograph", "visual art", "gallery", "exhibition"
            #     ],
            #     "language_and_literature": [
            #         "linguistics", "grammar", "language", "literature", "poetry", "novel",
            #         "fiction", "drama", "poem", "literary", "author", "writer", "playwright",
            #         "philology"
            #     ],
            #     "science": [
            #         "science", "scientific", "chemistry", "physics", "biology", "astronomy",
            #         "geology", "mathematics", "statistics", "zoology", "botany"
            #     ],
            #     "medicine": [
            #         "medicine", "medical", "health", "anatomy", "physiology", "pharmacology",
            #         "surgery", "clinical", "nursing", "public health"
            #     ],
            #     "agriculture": [
            #         "agriculture", "farming", "horticulture", "soil science", "aquaculture",
            #         "crop science", "animal husbandry", "sustainability"
            #     ],
            #     "technology": [
            #         "technology", "engineering", "computer science", "informatics", "robotics",
            #         "artificial intelligence", "electronics", "mechanical engineering",
            #         "electrical engineering", "industrial", "construction"
            #     ],
            #     "military_science": [
            #         "military science", "defense", "strategy", "tactics", "security studies",
            #         "armed forces", "military history"
            #     ],
            #     "naval_science": [
            #         "naval science", "navy", "maritime", "navigation", "sea power", "seafaring",
            #         "oceanography"
            #     ],
            #     "bibliography_library_science": [
            #         "bibliography", "library science", "information science", "archives",
            #         "metadata", "cataloging", "classification", "information management"
            #     ]
            # }
            domains = {
                # A — General Works
                'general_works': [
                    'encyclopedia',
                    'reference',
                    'almanac',
                    'bibliography',
                    'library science',
                    'information science',
                    'archives',
                    'periodicals'
                ],
                
                # B — Philosophy, Psychology, Religion
                # (Combined here for simplicity, though LCC splits them as B, BF, BL–BX, etc.)
                'philosophy_psychology_religion': [
                    'philosophy',
                    'logic',
                    'ethics',
                    'epistemology',
                    'metaphysics',
                    'psychology',
                    'psychoanalysis',
                    'cognitive science',
                    'religion',
                    'theology',
                    'spirituality',
                    'religious studies',
                    'bible',
                    'islam',
                    'judaism',
                    'buddhism',
                    'christianity',
                    'hindu'
                ],
                
                # C — Auxiliary Sciences of History
                'auxiliary_sciences_of_history': [
                    'archaeology',
                    'paleography',
                    'epigraphy',
                    'numismatics',
                    'genealogy',
                    'chronology',
                    'diplomatics',
                    'heraldry'
                ],
                
                # D — World History; E-F — History of the Americas
                # (Combined here under 'history' for simplicity)
                'history': [
                    'history',
                    'historical',
                    'ancient',
                    'medieval',
                    'renaissance',
                    'early modern',
                    'contemporary history',
                    'revolution',
                    'civilization',
                    'world history',
                    'american history',
                    'european history',
                    'asian history',
                    'african history',
                    'archival records'
                ],
                
                # G — Geography, Anthropology, Recreation
                'geography_anthropology_recreation': [
                    'geography',
                    'cartography',
                    'anthropology',
                    'ethnology',
                    'cultural studies',
                    'folklore',
                    'travel',
                    'tourism',
                    'sports',
                    'recreation',
                    'leisure'
                ],
                
                # H — Social Sciences
                'social_sciences': [
                    'sociology',
                    'economics',
                    'finance',
                    'demography',
                    'social work',
                    'public policy',
                    'social research',
                    'criminology',
                    'gender studies'
                ],
                
                # J — Political Science
                'political_science': [
                    'political science',
                    'government',
                    'public administration',
                    'comparative politics',
                    'international relations',
                    'political theory',
                    'public affairs',
                    'diplomacy'
                ],
                
                # K — Law
                'law': [
                    'law',
                    'legal studies',
                    'legislation',
                    'jurisprudence',
                    'criminal law',
                    'civil law',
                    'constitutional law',
                    'international law',
                    'legal ethics'
                ],
                
                # L — Education
                'education': [
                    'education',
                    'teaching',
                    'pedagogy',
                    'curriculum',
                    'educational psychology',
                    'educational policy',
                    'instructional design',
                    'learning theory'
                ],
                
                # M — Music
                'music': [
                    'music',
                    'musical composition',
                    'harmony',
                    'counterpoint',
                    'sonata',
                    'symphony',
                    'opera',
                    'musician',
                    'composer',
                    'piano',
                    'violin',
                    'orchestra',
                    'concert',
                    'music theory'
                ],
                
                # N — Fine Arts
                'fine_arts': [
                    'art',
                    'painting',
                    'sculpture',
                    'drawing',
                    'printmaking',
                    'photography',
                    'visual art',
                    'gallery',
                    'exhibition',
                    'installation art',
                    'portrait'
                ],
                
                # P — Language and Literature
                'language_and_literature': [
                    'language',
                    'linguistics',
                    'literature',
                    'poetry',
                    'novel',
                    'fiction',
                    'drama',
                    'poem',
                    'literary criticism',
                    'author',
                    'writer',
                    'playwright',
                    'philology'
                ],
                
                # Q — Science
                'science': [
                    'science',
                    'scientific method',
                    'chemistry',
                    'physics',
                    'biology',
                    'mathematics',
                    'astronomy',
                    'geology',
                    'ecology',
                    'zoology',
                    'botany',
                    'environmental science'
                ],
                
                # R — Medicine
                'medicine': [
                    'medicine',
                    'healthcare',
                    'anatomy',
                    'physiology',
                    'nursing',
                    'pharmacology',
                    'public health',
                    'clinical research',
                    'therapeutics',
                    'epidemiology'
                ],
                
                # S — Agriculture
                'agriculture': [
                    'agriculture',
                    'farming',
                    'horticulture',
                    'forestry',
                    'aquaculture',
                    'agronomy',
                    'soil science',
                    'crop science',
                    'veterinary science'
                ],
                
                # T — Technology
                'technology': [
                    'technology',
                    'engineering',
                    'computer science',
                    'information technology',
                    'mechanical engineering',
                    'electrical engineering',
                    'robotics',
                    'biotechnology',
                    'materials science'
                ],
                
                # U — Military Science
                'military_science': [
                    'military science',
                    'military history',
                    'strategy',
                    'tactics',
                    'defense studies',
                    'security studies',
                    'armed forces',
                    'logistics'
                ],
                
                # V — Naval Science
                'naval_science': [
                    'naval science',
                    'maritime studies',
                    'naval history',
                    'seapower',
                    'naval strategy',
                    'nautical engineering'
                ],
                
                # Z — Bibliography, Library Science, Information Resources
                'bibliography_library_science': [
                    'bibliography',
                    'cataloging',
                    'classification',
                    'library science',
                    'knowledge organization',
                    'archives management',
                    'digital libraries',
                    'metadata'
                ]
            }


            
            # Check which domains are mentioned in the subjects
            found_domains = []
            for domain, keywords in domains.items():
                if any(keyword in subjects for keyword in keywords):
                    found_domains.append(domain)
            
            if found_domains:
                df_clean.at[idx, 'primary_domain'] = ';'.join(found_domains)
        
        # Count records by primary domain (for logging)
        domain_counts = df_clean['primary_domain'].value_counts().to_dict()
        logging.info(f"Domain classification: {domain_counts}")
        
        logging.info("Preprocessing complete")
        return df_clean
    
    def calculate_temporal_similarity(self, record1: Dict, record2: Dict) -> float:
        """
        Calculate temporal similarity based on birth/death years and publication years.
        
        Args:
            record1: First record
            record2: Second record
                
        Returns:
            Float similarity score (0.0-1.0)
        """
        # Initialize with neutral similarity
        temporal_sim = 0.5
        
        # If both records have birth years
        if not pd.isna(record1.get('birth_year')) and not pd.isna(record2.get('birth_year')):
            birth1 = record1['birth_year']
            birth2 = record2['birth_year']
            
            # If birth years match exactly, strong signal for same entity
            if birth1 == birth2:
                temporal_sim = 0.9
                
                # Check death years if available
                if not pd.isna(record1.get('death_year')) and not pd.isna(record2.get('death_year')):
                    death1 = record1['death_year']
                    death2 = record2['death_year']
                    
                    # Both birth and death years match - very strong signal
                    if death1 == death2:
                        temporal_sim = 1.0
                    # Death years are different - potential different entities
                    else:
                        # Calculate how different the death years are
                        year_diff = abs(death1 - death2)
                        # Small differences might be data errors, large differences suggest different entities
                        if year_diff <= 3:
                            # Likely a data error, still high similarity
                            temporal_sim = 0.8
                        else:
                            # Different entities with same birth year
                            temporal_sim = 0.3
                
            # Birth years are different
            else:
                # Calculate how different the birth years are
                year_diff = abs(birth1 - birth2)
                # Small differences might be data errors
                if year_diff <= 3:
                    # Likely a data error, still reasonable similarity
                    temporal_sim = 0.7
                elif year_diff <= 10:
                    # Might be different people born close together
                    temporal_sim = 0.4
                else:
                    # Definitely different people
                    temporal_sim = 0.0
                    
        # Only one record has birth information
        elif pd.isna(record1.get('birth_year')) != pd.isna(record2.get('birth_year')):
            # Get the record with birth year
            record_with_dates = record1 if not pd.isna(record1.get('birth_year')) else record2
            record_without_dates = record2 if not pd.isna(record1.get('birth_year')) else record1
            
            # If names are very similar, check if publication year is compatible with lifespan
            name_similarity = self.calculate_name_similarity(
                record1.get('normalized_name', ''), 
                record2.get('normalized_name', '')
            )
            
            if name_similarity > 0.9 and not pd.isna(record_without_dates.get('pub_year')):
                birth_year = record_with_dates['birth_year']
                death_year = record_with_dates['death_year'] if not pd.isna(record_with_dates['death_year']) else birth_year + 80
                pub_year = record_without_dates['pub_year']
                
                # Check if publication year falls within expanded lifespan (include posthumous works)
                if birth_year <= pub_year <= (death_year + 30):  # Allow for posthumous publications
                    temporal_sim = 0.7  # Compatible publication year
                elif pub_year < birth_year:
                    # Publication before birth - unlikely to be the same person
                    year_diff = birth_year - pub_year
                    if year_diff <= 5:
                        # Might be data error
                        temporal_sim = 0.4
                    else:
                        temporal_sim = 0.2
                else:  # pub_year > death_year + 30
                    # Publication too long after death - unlikely to be the same person
                    year_diff = pub_year - (death_year + 30)
                    if year_diff <= 10:
                        # Could still be posthumous
                        temporal_sim = 0.5
                    else:
                        temporal_sim = 0.3
        
        # If both records have publication years but no birth/death years
        elif not pd.isna(record1.get('pub_year')) and not pd.isna(record2.get('pub_year')):
            pub1 = record1['pub_year']
            pub2 = record2['pub_year']
            
            # Calculate similarity based on publication year proximity
            year_diff = abs(pub1 - pub2)
            if year_diff <= 10:
                # Publications close together, likely same person
                temporal_sim = 0.6
            elif year_diff <= 30:
                # Publications moderately far apart, might be same person
                temporal_sim = 0.4
            else:
                # Publications far apart, less likely to be same person
                temporal_sim = 0.2
        
        return temporal_sim

    def _normalize_name(self, name: str) -> str:
        """
        Normalize a name for comparison.
        
        Args:
            name: Person name to normalize
                
        Returns:
            Normalized name
        """
        if not name or pd.isna(name):
            return ''
        
        # Convert to lowercase
        name = name.lower()
        
        # Handle specific common name variations (e.g., abbreviations, alternate spellings)
        # This can be expanded with domain-specific rules
        substitutions = {
            'wm.': 'william',
            'chas.': 'charles',
            'jas.': 'james',
            'thos.': 'thomas',
            'robt.': 'robert',
            'jno.': 'john',
            'geo.': 'george',
            'jos.': 'joseph',
            'alexr.': 'alexander',
            'edwd.': 'edward',
            'richd.': 'richard',
            'jr.': 'junior',
            'sr.': 'senior'
        }
        
        # Apply substitutions
        for abbrev, full in substitutions.items():
            # Match abbreviations with word boundary to avoid partial matches
            name = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, name)
        
        # Remove punctuation, keeping only letters and spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names with improved handling of name variations.
        
        Args:
            name1: First name
            name2: Second name
                
        Returns:
            Float similarity score (0.0-1.0)
        """
        if pd.isna(name1) or pd.isna(name2) or not name1 or not name2:
            return 0.0
                
        # Normalize names
        norm_name1 = self._normalize_name(name1)
        norm_name2 = self._normalize_name(name2)
        
        # Check for exact match
        if norm_name1 == norm_name2:
            return 1.0
        
        # Check for name parts (e.g., "Smith, John" vs "Smith, John Q.")
        name1_parts = set(norm_name1.split())
        name2_parts = set(norm_name2.split())
        
        # If one name is a subset of the other, high similarity
        if name1_parts.issubset(name2_parts) or name2_parts.issubset(name1_parts):
            # Calculate overlap ratio
            overlap = len(name1_parts.intersection(name2_parts))
            max_parts = max(len(name1_parts), len(name2_parts))
            
            if overlap / max_parts >= 0.7:  # At least 70% of parts match
                return 0.9
            else:
                return 0.8
        
        # Reorder names for better comparison (e.g., "Smith, John" vs "John Smith")
        alt_name1 = self._reorder_name(norm_name1)
        alt_name2 = self._reorder_name(norm_name2)
        
        # Check if reordered versions match
        if alt_name1 == alt_name2:
            return 0.95
                
        # Calculate Levenshtein distance-based similarity
        max_len = max(len(norm_name1), len(norm_name2))
        if max_len == 0:
            return 1.0
                
        if LEVENSHTEIN_AVAILABLE:
            # Use Levenshtein distance between original names
            org_dist = distance(norm_name1, norm_name2)
            org_sim = 1.0 - (org_dist / max_len)
            
            # Also try with reordered names
            alt_dist = distance(alt_name1, alt_name2)
            alt_sim = 1.0 - (alt_dist / max(len(alt_name1), len(alt_name2)))
            
            # Take the best similarity score
            similarity = max(org_sim, alt_sim)
        else:
            # Fallback similarity calculation without Levenshtein
            # Count matching characters
            matches = sum(c1 == c2 for c1, c2 in zip(norm_name1, norm_name2))
            similarity = matches / max_len
        
        return similarity

    def _reorder_name(self, name: str) -> str:
        """
        Reorder name parts to handle different name formats (e.g., "Last, First" vs "First Last").
        
        Args:
            name: Normalized name
                
        Returns:
            Name with parts potentially reordered
        """
        # Look for comma-separated parts
        if ',' in name:
            parts = name.split(',')
            if len(parts) == 2:
                # Reorder "Last, First" to "First Last"
                return f"{parts[1].strip()} {parts[0].strip()}"
        
        # If no comma or multiple commas, leave as is
        return name
    
    def calculate_vector_similarity(self, 
                               record1_id: str, 
                               record2_id: str,
                               vector_df: pd.DataFrame) -> float:
        """
        Calculate vector similarity between two records using embedded vectors.
        
        Args:
            record1_id: ID of first record
            record2_id: ID of second record
            vector_df: DataFrame with vector embeddings
            
        Returns:
            Float similarity score (0.0-1.0)
        """
        # Check if both records exist in vector data
        if record1_id not in vector_df.index and record1_id not in vector_df['id'].values:
            logging.debug(f"Record {record1_id} not found in vector data")
            return 0.0
        
        if record2_id not in vector_df.index and record2_id not in vector_df['id'].values:
            logging.debug(f"Record {record2_id} not found in vector data")
            return 0.0
        
        try:
            # Get vectors - try both index and 'id' column
            vector1 = None
            vector2 = None
            
            # First try using index
            if record1_id in vector_df.index:
                vector1_row = vector_df.loc[record1_id]
            else:
                # Try using 'id' column
                vector1_row = vector_df[vector_df['id'] == record1_id].iloc[0]
                
            if record2_id in vector_df.index:
                vector2_row = vector_df.loc[record2_id]
            else:
                # Try using 'id' column
                vector2_row = vector_df[vector_df['id'] == record2_id].iloc[0]
            
            # Get 'person' column vectors as they represent the name semantics
            if 'person' in vector1_row and 'person' in vector2_row:
                vector1 = vector1_row['person']
                vector2 = vector2_row['person']
                
                # Parse vectors if they're strings
                if isinstance(vector1, str) and '[' in vector1:
                    # Use ast.literal_eval for safer parsing
                    import ast
                    try:
                        vector1 = np.array(ast.literal_eval(vector1), dtype=np.float32)
                        vector2 = np.array(ast.literal_eval(vector2), dtype=np.float32)
                    except (ValueError, SyntaxError) as e:
                        logging.warning(f"Error parsing vector strings: {e}")
                        return 0.0
            else:
                # No person vectors available
                logging.debug(f"No 'person' vectors for records {record1_id} and {record2_id}")
                return 0.0
            
            # Ensure vectors are numpy arrays
            if not isinstance(vector1, np.ndarray):
                vector1 = np.array(vector1, dtype=np.float32)
            
            if not isinstance(vector2, np.ndarray):
                vector2 = np.array(vector2, dtype=np.float32)
            
            # Calculate cosine similarity
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                return 0.0
                
            similarity = float(np.dot(vector1, vector2) / (norm1 * norm2))
            
            # Log some successful calculations to verify it's working
            logging.debug(f"Vector similarity between {record1_id} and {record2_id}: {similarity:.4f}")
            
            return max(0.0, min(1.0, similarity))  # Ensure result is between 0 and 1
            
        except Exception as e:
            logging.warning(f"Error calculating vector similarity between {record1_id} and {record2_id}: {e}")
            return 0.0
        
    def calculate_domain_similarity(self, record1: Dict, record2: Dict) -> float:
        """
        Calculate domain similarity between records, crucial for disambiguation.
        
        Args:
            record1: First record
            record2: Second record
            
        Returns:
            Float similarity score (0.0-1.0)
        """
        # Check for empty domains - use neutral similarity if no domain info
        if ((pd.isna(record1.get('subjects')) and pd.isna(record1.get('genres'))) or
            (pd.isna(record2.get('subjects')) and pd.isna(record2.get('genres')))):
            return 0.5  # Neutral if insufficient domain information
        
        similarities = []
        
        # If primary domains are available, compare them first (strongest signal)
        if not pd.isna(record1.get('primary_domain')) and not pd.isna(record2.get('primary_domain')):
            domains1 = set(record1['primary_domain'].split(';'))
            domains2 = set(record2['primary_domain'].split(';'))
            
            # Check for domain overlap
            if domains1.intersection(domains2):
                # At least one domain in common - strong signal for compatibility
                overlap_ratio = len(domains1.intersection(domains2)) / max(len(domains1), len(domains2))
                similarities.append(0.8 + (0.2 * overlap_ratio))  # High similarity for domain match
            elif not domains1.intersection(domains2) and len(domains1) > 0 and len(domains2) > 0:
                # Different domains - signal for potential disambiguation
                similarities.append(0.2)  # Low similarity for domain mismatch
        
        # **Subject similarity using TF-IDF**
        subject1 = record1.get('subjects', "")
        subject2 = record2.get('subjects', "")

        if isinstance(subject1, str) and isinstance(subject2, str):
            subject1 = subject1.strip().lower()
            subject2 = subject2.strip().lower()

            # **Pre-validation: Ensure valid words exist before using TF-IDF**
            if not subject1 or not subject2 or len(subject1) < 3 or len(subject2) < 3:
                similarities.append(0.5)  # Neutral similarity if subjects are empty or too short
            else:
                import regex as re  # Use regex module for Unicode word matching
                token_pattern = re.compile(r'\p{L}+\d*', re.UNICODE)  # Match words AND numbers

                tokens1 = token_pattern.findall(subject1)
                tokens2 = token_pattern.findall(subject2)

                if not tokens1 or not tokens2:
                    logging.warning(f"Skipping TF-IDF: No valid words in subjects '{subject1}' | '{subject2}'")
                    similarities.append(0.5)  # Neutral similarity
                else:
                    try:
                        vectorizer = TfidfVectorizer(min_df=1, stop_words='english', token_pattern=r'\b\w+\b')
                        tfidf_matrix = vectorizer.fit_transform([" ".join(tokens1), " ".join(tokens2)])

                        if tfidf_matrix.shape[1] > 0:  # Ensure vocabulary is not empty
                            subject_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                            similarities.append(subject_sim)
                        else:
                            similarities.append(0.5)  # Assign neutral similarity if vocab is empty
                    except ValueError as ve:
                        logging.warning(f"TF-IDF error due to empty vocabulary: {ve}")
                        similarities.append(0.5)  # Assign neutral similarity
                    except Exception as e:
                        logging.warning(f"Error calculating subject similarity: {e}")
                        similarities.append(0.5)  # Assign neutral similarity
        
        # Genre similarity
        genre_sim = self._calculate_field_similarity(record1.get('genres'), record2.get('genres'))
        if genre_sim is not None:
            similarities.append(genre_sim)
        
        # Related work similarity - can indicate domain compatibility
        relwork_sim = self._calculate_field_similarity(record1.get('relatedWork'), record2.get('relatedWork'))
        if relwork_sim is not None:
            similarities.append(relwork_sim)
        
        # Entity mentions similarity - useful for identifying context entities
        entity_sim = self._calculate_field_similarity(record1.get('entity_mentions'), record2.get('entity_mentions'))
        if entity_sim is not None:
            similarities.append(entity_sim)
        
        # Return average similarity if available, with more weight to primary domain
        if not similarities:
            return 0.5  # Neutral if no similarity could be calculated
        
        # Weight the first similarity (primary domain) more heavily if it exists
        if len(similarities) > 1 and not pd.isna(record1.get('primary_domain')) and not pd.isna(record2.get('primary_domain')):
            # 60% weight to primary domain, 40% to other metrics
            return (0.6 * similarities[0]) + (0.4 * np.mean(similarities[1:]))
        else:
            return np.mean(similarities)

    def _calculate_field_similarity(self, field1: Any, field2: Any) -> Optional[float]:
        """
        Calculate similarity between two field values, handling nulls properly.
        
        Args:
            field1: First field value
            field2: Second field value
            
        Returns:
            Float similarity score (0.0-1.0) or None if fields are null
        """
        # Return None if either field is null or empty
        if pd.isna(field1) or pd.isna(field2) or not field1 or not field2:
            return None
        
        # Handle list-like fields (separated by semicolons)
        if isinstance(field1, str) and isinstance(field2, str) and ';' in field1 and ';' in field2:
            set1 = set(item.strip().lower() for item in field1.split(';') if item.strip())
            set2 = set(item.strip().lower() for item in field2.split(';') if item.strip())
            
            if not set1 or not set2:
                return None
            
            # Calculate Jaccard similarity for sets
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            if union == 0:
                return 0.0
            
            return intersection / union
        
        # For regular text fields, use TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(min_df=1)
            tfidf_matrix = vectorizer.fit_transform([str(field1), str(field2)])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            logging.warning(f"Error calculating field similarity: {e}")
            return 0.0
        
    def find_candidate_pairs(self, 
                        df: pd.DataFrame, 
                        vector_df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        """
        Find candidate pairs using semantic blocking with person vectors.
        
        Args:
            df: Preprocessed DataFrame
            vector_df: DataFrame with vector embeddings
            
        Returns:
            List of candidate pairs (idx1, idx2, similarity)
        """
        candidate_pairs = []
        
        if self.qdrant_client is not None:
            logging.info("Using semantic blocking with person vectors for candidate pair generation")
            
            # Keep track of searched pairs to avoid duplicates
            searched_pairs = set()
            
            # Process each record with a vector
            processed_count = 0
            vector_record_count = df['has_vector'].sum()
            
            # Process in batches for logging progress
            batch_size = max(1, int(vector_record_count / 10))  # Log progress in ~10 steps
            
            for idx, row in df[df['has_vector']].iterrows():
                processed_count += 1
                
                # Log progress periodically
                if processed_count % batch_size == 0 or processed_count == vector_record_count:
                    logging.info(f"Candidate pair generation progress: {processed_count}/{vector_record_count} "
                                f"({processed_count/vector_record_count:.1%})")
                
                record_id = row['id']
                if record_id not in vector_df.index:
                    continue
                    
                # Get the person vector
                try:
                    person_vector = None
                    
                    # Try to get person vector from vector_df
                    if 'person' in vector_df.columns:
                        person_val = vector_df.loc[record_id, 'person']
                        if isinstance(person_val, str) and '[' in person_val:
                            # Parse string representation of vector
                            import ast
                            person_vector = np.array(ast.literal_eval(person_val), dtype=np.float32)
                        elif isinstance(person_val, np.ndarray):
                            person_vector = person_val
                    
                    # Fall back to full vector if person vector is not available
                    if person_vector is None:
                        person_vector = vector_df.loc[record_id].values
                        if isinstance(person_vector, str):
                            import ast
                            person_vector = np.array(ast.literal_eval(person_vector), dtype=np.float32)
                        else:
                            person_vector = np.array(person_vector, dtype=np.float32)
                except Exception as e:
                    logging.warning(f"Error getting person vector for record {record_id}: {e}")
                    continue
                
                # Skip if vector is invalid
                if person_vector is None or person_vector.size == 0:
                    continue
                
                # Set search parameters for semantic name blocking
                # Using a higher threshold for name vector similarity to focus on strong matches
                search_params = {
                    "collection_name": "entity_vectors",
                    "query_vector": person_vector.tolist(),
                    "limit": 20,  # Return up to 20 similar names
                    "score_threshold": 0.80  # High threshold for name similarity
                }
                
                birth_year = row.get('birth_year')
                if birth_year is not None and not pd.isna(birth_year):
                    query_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="birth_year",
                                match=models.MatchValue(value=int(birth_year))
                            )
                        ]
                    )
                else:
                    query_filter = None
                
                    search_params["query_filter"] = query_filter
                
                # Search for semantically similar person names
                try:
                    results = self.qdrant_client.search(**search_params)
                except Exception as e:
                    logging.warning(f"Qdrant search error: {e}")
                    continue
                
                # Process results
                for result in results:
                    # Skip self-matches
                    if result.id == idx:
                        continue
                    
                    # Skip already processed pairs
                    pair_key = tuple(sorted([idx, result.id]))
                    if pair_key in searched_pairs:
                        continue
                    
                    searched_pairs.add(pair_key)
                    
                    # Person vector similarity score
                    name_sim = result.score
                    
                    # Calculate additional similarity measures for precision
                    temporal_sim = self.calculate_temporal_similarity(
                        df.iloc[idx], df.iloc[result.id])
                    
                    domain_sim = self.calculate_domain_similarity(
                        df.iloc[idx], df.iloc[result.id])
                    
                    # For potential disambiguation cases (very similar names but different entities)
                    # check if birth/death years conflict
                    birth1 = df.iloc[idx].get('birth_year')
                    birth2 = df.iloc[result.id].get('birth_year')
                    
                    # If both have birth years and they're different, this is likely a different entity
                    if (not pd.isna(birth1) and not pd.isna(birth2) and 
                        abs(birth1 - birth2) > 5):  # Allow small differences due to data errors
                        # Different entities with similar names
                        adjusted_score = name_sim * 0.5  # Significantly reduce score
                    # If domains are very different, also likely different entities
                    elif domain_sim < 0.3:
                        # Different domains suggest different entities
                        adjusted_score = name_sim * 0.7  # Moderately reduce score
                    else:
                        # Likely the same entity
                        adjusted_score = name_sim
                    
                    # Add as candidate if still above threshold
                    if adjusted_score >= 0.7:
                        candidate_pairs.append((idx, result.id, adjusted_score))
            
            logging.info(f"Semantic blocking found {len(candidate_pairs)} candidate pairs "
                        f"from {len(searched_pairs)} unique comparisons")
            
        else:
            # Fallback method
            logging.warning("Using pairwise calculation for candidate pairs (Qdrant not available)")
            
            # Only process records with vectors
            vector_records = df[df['has_vector']].index.tolist()
            vector_record_count = len(vector_records)
            
            logging.info(f"Pairwise comparison of {vector_record_count} records with vectors "
                        f"({vector_record_count * (vector_record_count - 1) / 2} potential pairs)")
            
            for i, idx1 in enumerate(vector_records):
                # Log progress every 100 records
                if i % 100 == 0 or i == len(vector_records) - 1:
                    logging.info(f"Pairwise comparison progress: {i+1}/{len(vector_records)} "
                                f"({(i+1)/len(vector_records):.1%})")
                    
                record1 = df.loc[idx1]
                
                for j, idx2 in enumerate(vector_records[i+1:], i+1):
                    record2 = df.loc[idx2]
                    
                    # Skip comparisons between records with different birth/death years
                    if (not pd.isna(record1.get('birth_year')) and not pd.isna(record2.get('birth_year')) and
                        record1['birth_year'] != record2['birth_year']):
                        continue
                    
                    # Calculate vector similarity
                    vector_sim = self.calculate_vector_similarity(record1['id'], record2['id'], vector_df)
                    
                    # Add high-similarity pairs as candidates
                    if vector_sim > 0.65:
                        candidate_pairs.append((idx1, idx2, vector_sim))
            
        # Sort candidate pairs by similarity score (descending)
        candidate_pairs.sort(key=lambda x: x[2], reverse=True)
        
        logging.info(f"Found {len(candidate_pairs)} candidate pairs")
        return candidate_pairs
    
    def build_similarity_matrix(self, 
                           df: pd.DataFrame, 
                           vector_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build similarity matrix between entity pairs with enhanced metrics.
        
        Args:
            df: Preprocessed DataFrame
            vector_df: Optional DataFrame with vector embeddings
            
        Returns:
            Tuple of (similarity_matrix, confidence_matrix)
        """
        n_records = len(df)
        similarity_matrix = np.zeros((n_records, n_records))
        confidence_matrix = np.zeros((n_records, n_records))
        
        logging.info(f"Building similarity matrix for {n_records} records")
        
        # Check if we have vector data
        has_vectors = vector_df is not None and len(vector_df) > 0
        
        # Generate candidate pairs for efficiency
        candidate_pairs = []
        if has_vectors:
            candidate_pairs = self.find_candidate_pairs(df, vector_df)
            
            # Initialize from candidate pairs
            for idx1, idx2, sim in candidate_pairs:
                similarity_matrix[idx1, idx2] = similarity_matrix[idx2, idx1] = sim
                
                # Set initial confidence based on vector similarity
                confidence_matrix[idx1, idx2] = confidence_matrix[idx2, idx1] = 0.8
        
        # Calculate similarities for all pairs (or only missing pairs)
        total_pairs = n_records * (n_records - 1) // 2
        processed = 0
        for i in range(n_records):
            # Identical records have similarity 1.0 with full confidence
            similarity_matrix[i, i] = 1.0
            confidence_matrix[i, i] = 1.0
            
            # Log progress every 1% or 1000 records, whichever is more frequent
            log_interval = max(1, total_pairs // 100)
            
            for j in range(i+1, n_records):
                # Skip if similarity already set from candidate pairs
                if similarity_matrix[i, j] > 0:
                    processed += 1
                    continue
                
                # Log progress
                processed += 1
                if processed % log_interval == 0 or processed == total_pairs:
                    logging.info(f"Building similarity matrix: {processed}/{total_pairs} pairs "
                                f"({processed/total_pairs:.1%})")
                
                # Get record completeness information
                completeness1 = df.iloc[i].get('context_completeness', 1.0)
                completeness2 = df.iloc[j].get('context_completeness', 1.0)
                
                # Average completeness score affects confidence
                avg_completeness = (completeness1 + completeness2) / 2.0
                
                # Calculate name similarity
                name_sim = self.calculate_name_similarity(
                    df.iloc[i]['normalized_name'],
                    df.iloc[j]['normalized_name']
                )
                
                # Quick filter: Skip further calculation if names are very dissimilar
                if name_sim < 0.5:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0.0
                    confidence_matrix[i, j] = confidence_matrix[j, i] = avg_completeness
                    continue
                
                # Calculate temporal similarity - important for disambiguation
                temporal_sim = self.calculate_temporal_similarity(
                    df.iloc[i], df.iloc[j])
                
                # If temporal data clearly indicates different entities, skip further comparison
                if temporal_sim < 0.2:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 0.1
                    confidence_matrix[i, j] = confidence_matrix[j, i] = avg_completeness
                    continue
                
                # Calculate vector similarity if vectors are available
                vector_sim = 0.0
                has_field_vectors = False
                
                if has_vectors:
                    id1 = df.iloc[i]['id']
                    id2 = df.iloc[j]['id']
                    
                    if df.iloc[i]['has_vector'] and df.iloc[j]['has_vector']:
                        vector_sim = self.calculate_vector_similarity(id1, id2, vector_df)
                        has_field_vectors = True
                
                # Calculate context similarity
                context_sim = self._calculate_field_similarity(
                    df.iloc[i]['context'],
                    df.iloc[j]['context']
                ) or 0.0
                
                # Calculate domain similarity (crucial for disambiguation)
                domain_sim = self.calculate_domain_similarity(
                    df.iloc[i], df.iloc[j])
                
                # Special handling for identical names - critical for disambiguation
                if name_sim > 0.9:
                    # If names are identical but temporal data conflicts, likely different entities
                    if temporal_sim < 0.4:
                        weights = {
                            'name_sim': 0.15,
                            'temporal_sim': 0.3,
                            'vector_sim': 0.25 if has_field_vectors else 0.0,
                            'context_sim': 0.1,
                            'domain_sim': 0.2
                        }
                    # If names are identical but domains are very different, likely different entities
                    elif domain_sim < 0.3:
                        weights = {
                            'name_sim': 0.2,
                            'temporal_sim': 0.2,
                            'vector_sim': 0.2 if has_field_vectors else 0.0,
                            'context_sim': 0.1,
                            'domain_sim': 0.3
                        }
                    else:
                        # Normal weights when both names and domains are compatible
                        weights = {
                            'name_sim': 0.3,
                            'temporal_sim': 0.2,
                            'vector_sim': 0.25 if has_field_vectors else 0.0,
                            'context_sim': 0.15,
                            'domain_sim': 0.1
                        }
                else:
                    # Standard weights for different names
                    weights = {
                        'name_sim': 0.35,
                        'temporal_sim': 0.2,
                        'vector_sim': 0.25 if has_field_vectors else 0.0,
                        'context_sim': 0.1,
                        'domain_sim': 0.1
                    }
                
                # Normalize weights
                weight_sum = sum(weights.values())
                if weight_sum > 0:
                    weights = {k: v/weight_sum for k, v in weights.items()}
                
                # Calculate weighted similarity
                combined_sim = (
                    weights['name_sim'] * name_sim +
                    weights['temporal_sim'] * temporal_sim +
                    weights['vector_sim'] * vector_sim +
                    weights['context_sim'] * context_sim +
                    weights['domain_sim'] * domain_sim
                )
                
                # Store results
                similarity_matrix[i, j] = similarity_matrix[j, i] = combined_sim
                confidence_matrix[i, j] = confidence_matrix[j, i] = avg_completeness
        
        return similarity_matrix, confidence_matrix
    
    def _refine_clusters_with_domain_coherence(self, 
                                          df: pd.DataFrame, 
                                          initial_clusters: List[Set[int]], 
                                          similarity_matrix: np.ndarray) -> List[Set[int]]:
        """
        Refine clusters by checking domain coherence, particularly for
        entities with identical names but different domains.
        
        Args:
            df: Preprocessed DataFrame
            initial_clusters: Initial cluster assignments
            similarity_matrix: Similarity matrix
                
        Returns:
            Refined clusters
        """
        refined_clusters = []
        
        # Track refinement statistics
        total_clusters = len(initial_clusters)
        split_clusters = 0
        
        logging.info(f"Refining {total_clusters} initial clusters")
        
        for cluster_idx, cluster in enumerate(initial_clusters):
            # Skip small clusters
            if len(cluster) <= 1:
                refined_clusters.append(cluster)
                continue
            
            # Check for potential disambiguation cases (identical names, different domains)
            names = defaultdict(list)
            for node_id in cluster:
                normalized_name = df.iloc[node_id]['normalized_name']
                names[normalized_name].append(node_id)
            
            # Check for names with multiple entities
            needs_splitting = False
            same_name_different_domains = []
            
            # Log every 100 clusters for progress tracking
            if cluster_idx % 100 == 0 or cluster_idx == total_clusters - 1:
                logging.info(f"Refining clusters: {cluster_idx+1}/{total_clusters} "
                            f"({(cluster_idx+1)/total_clusters:.1%})")
            
            for name, node_ids in names.items():
                if len(node_ids) > 1:
                    # Check temporal coherence first - split if birth/death years conflict
                    years_conflict = False
                    birth_years = set()
                    death_years = set()
                    
                    for idx in node_ids:
                        if not pd.isna(df.iloc[idx].get('birth_year')):
                            birth_years.add(df.iloc[idx]['birth_year'])
                        if not pd.isna(df.iloc[idx].get('death_year')):
                            death_years.add(df.iloc[idx]['death_year'])
                    
                    # If we have multiple different birth or death years, potential conflict
                    if len(birth_years) > 1 or len(death_years) > 1:
                        years_conflict = True
                        
                        # Log this case
                        logging.info(f"Potential temporal conflict for name '{name}': "
                                    f"birth years {birth_years}, death years {death_years}")
                    
                    # Check domain coherence only if needed
                    if years_conflict or len(birth_years) == 0:
                        # Calculate pairwise domain similarity
                        domain_sims = []
                        incoherent_pairs = []
                        
                        for i in range(len(node_ids)):
                            for j in range(i+1, len(node_ids)):
                                idx1 = node_ids[i]
                                idx2 = node_ids[j]
                                
                                # Use already computed similarity if available
                                if similarity_matrix[idx1, idx2] < self.min_similarity:
                                    # Already determined to be dissimilar
                                    domain_sims.append(0.2)  # Low similarity
                                    incoherent_pairs.append((idx1, idx2))
                                else:
                                    # Calculate domain similarity directly
                                    domain_sim = self.calculate_domain_similarity(
                                        df.iloc[idx1], df.iloc[idx2])
                                    domain_sims.append(domain_sim)
                                    
                                    # Track pairs with low domain similarity
                                    if domain_sim < 0.25:  # Lowered threshold for splitting
                                        incoherent_pairs.append((idx1, idx2))
                        
                        # If domains are not coherent or years conflict, mark for splitting
                        if (domain_sims and np.mean(domain_sims) < 0.25) or years_conflict:
                            needs_splitting = True
                            same_name_different_domains.append((name, node_ids, incoherent_pairs))
            
            if needs_splitting:
                # Split cluster using domain and temporal coherence
                subclusters = self._split_cluster_by_coherence(df, cluster, same_name_different_domains)
                
                # Log the split
                split_clusters += 1
                original_size = len(cluster)
                subcluster_sizes = [len(sc) for sc in subclusters]
                
                logging.info(f"Split cluster of size {original_size} into {len(subclusters)} subclusters: {subcluster_sizes}")
                
                # Add the resulting subclusters
                refined_clusters.extend(subclusters)
            else:
                # Keep cluster as is
                refined_clusters.append(cluster)
        
        logging.info(f"Cluster refinement complete: {split_clusters}/{total_clusters} clusters split "
                    f"({split_clusters/total_clusters:.1%})")
        
        return refined_clusters

    def _split_cluster_by_coherence(self, 
                                df: pd.DataFrame, 
                                cluster: Set[int],
                                incoherent_names: List[Tuple[str, List[int], List[Tuple[int, int]]]]) -> List[Set[int]]:
        """
        Split a cluster into coherent subclusters by building a refined graph
        that preserves only compatible relationships.
        
        Args:
            df: Preprocessed DataFrame
            cluster: Original cluster to split
            incoherent_names: List of tuples (name, node_ids, incoherent_pairs) for names requiring splitting
                
        Returns:
            List of subclusters
        """
        # Create a subgraph for this cluster
        G = nx.Graph()
        
        # Add all nodes from the cluster
        for node_id in cluster:
            G.add_node(node_id, 
                    person=df.iloc[node_id]['person'],
                    name=df.iloc[node_id]['normalized_name'],
                    birth_year=df.iloc[node_id].get('birth_year'),
                    death_year=df.iloc[node_id].get('death_year'),
                    domains=df.iloc[node_id].get('primary_domain'))
        
        # Collect all incoherent pairs to avoid connecting them
        all_incoherent_pairs = set()
        for _, _, incoherent_pairs in incoherent_names:
            all_incoherent_pairs.update(incoherent_pairs)
        
        # Add edges between compatible nodes
        for i in cluster:
            for j in cluster:
                if i >= j:
                    continue
                    
                # Skip known incoherent pairs
                if (i, j) in all_incoherent_pairs or (j, i) in all_incoherent_pairs:
                    continue
                
                # Check if names are identical
                name_i = df.iloc[i]['normalized_name']
                name_j = df.iloc[j]['normalized_name']
                
                name_similar = self.calculate_name_similarity(name_i, name_j) > 0.9
                
                # For different names, connect if otherwise similar
                if not name_similar:
                    G.add_edge(i, j, weight=0.7)
                    continue
                
                # For identical names, check additional compatibility
                # Check birth/death years first - most reliable signal
                birth_i = df.iloc[i].get('birth_year')
                birth_j = df.iloc[j].get('birth_year')
                death_i = df.iloc[i].get('death_year')
                death_j = df.iloc[j].get('death_year')
                
                # If both have birth years and they match, connect strongly
                if not pd.isna(birth_i) and not pd.isna(birth_j):
                    if birth_i == birth_j:
                        # Connect with high weight
                        G.add_edge(i, j, weight=0.9)
                        continue
                    else:
                        # Different birth years - don't connect
                        continue
                
                # Check domain compatibility
                domain_sim = self.calculate_domain_similarity(df.iloc[i], df.iloc[j])
                
                # Only connect if domains are compatible
                if domain_sim >= 0.25:  # Lowered threshold
                    G.add_edge(i, j, weight=domain_sim)
        
        # Get connected components as subclusters
        subclusters = list(nx.connected_components(G))
        
        # If no subclusters were created (all nodes disconnected),
        # use a more permissive approach to avoid excessive fragmentation
        if not subclusters or all(len(c) == 1 for c in subclusters):
            # Group by normalized name as a fallback
            name_groups = defaultdict(set)
            for node_id in cluster:
                name_groups[df.iloc[node_id]['normalized_name']].add(node_id)
            
            # Subclusters for each unique name
            subclusters = list(name_groups.values())
            
            logging.info(f"Fallback clustering by name produced {len(subclusters)} subclusters")
        
        return subclusters
    
    def resolve_entities(self, 
                    df: pd.DataFrame, 
                    vector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Resolve entities with a two-phase approach: first exact matching, then
        similarity-based clustering with domain-aware refinement.
        
        Args:
            df: DataFrame with entity records
            vector_df: Optional DataFrame with vector embeddings
            
        Returns:
            DataFrame with cluster assignments
        """
        logging.info(f"Starting entity resolution for {len(df)} records")
        
        # Phase 1: Preprocessing and exact matching
        logging.info("Phase 1: Preprocessing data")
        processed_df = self.preprocess_data(df, vector_df)
        
        # Initialize Qdrant for vector search if vectors available
        if vector_df is not None and len(vector_df) > 0:
            logging.info("Initializing vector search with Qdrant")
            try:
                self._initialize_qdrant(processed_df, vector_df)
            except Exception as e:
                logging.error(f"Failed to initialize Qdrant: {e}")
        
        # Phase 2: Create initial clusters based on exact matches
        logging.info("Phase 2: Creating initial clusters with exact matching")
        
        # First, group by exact name+dates matches
        exact_clusters = {}
        next_cluster_id = 0
        
        # Track statistics
        exact_match_count = 0
        
        # Group records with exact name and date matches
        for idx, row in processed_df.iterrows():
            name = row['normalized_name']
            birth = row['birth_year'] if not pd.isna(row.get('birth_year')) else None
            death = row['death_year'] if not pd.isna(row.get('death_year')) else None
            
            # Create match keys with different levels of specificity
            # 1. Most specific: name + birth + death
            if birth is not None and death is not None:
                key = f"{name}|{birth}|{death}"
            # 2. Name + birth year only
            elif birth is not None:
                key = f"{name}|{birth}|"
            # 3. Just name (least specific)
            else:
                key = f"{name}||"
            
            # Assign to cluster
            if key not in exact_clusters:
                exact_clusters[key] = next_cluster_id
                next_cluster_id += 1
            
            processed_df.at[idx, 'temp_cluster'] = exact_clusters[key]
            exact_match_count += 1
        
        logging.info(f"Created {len(exact_clusters)} initial clusters through exact matching")
        
        # Phase 3: Build similarity matrix
        logging.info("Phase 3: Building similarity matrix")
        similarity_matrix, confidence_matrix = self.build_similarity_matrix(
            processed_df, vector_df)
        
        # Phase 4: Create initial cluster groups from the connected components
        logging.info("Phase 4: Creating connected components from similarity matrix")
        
        # Create graph for clustering
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(processed_df)):
            G.add_node(i, 
                    person=processed_df.iloc[i]['person'],
                    temp_cluster=processed_df.iloc[i]['temp_cluster'],
                    domains=processed_df.iloc[i].get('primary_domain', ''))
        
        # Determine threshold based on training
        threshold = self.min_similarity if self.is_trained else 0.7
        
        # Add edges above threshold, connecting only across different temp clusters
        edge_count = 0
        for i in range(len(processed_df)):
            for j in range(i+1, len(processed_df)):
                # Skip pairs in the same temp cluster (already matched exactly)
                if processed_df.iloc[i]['temp_cluster'] == processed_df.iloc[j]['temp_cluster']:
                    continue
                    
                if similarity_matrix[i, j] >= threshold:
                    # Add edge with similarity as weight
                    G.add_edge(i, j, 
                            weight=similarity_matrix[i, j],
                            confidence=confidence_matrix[i, j])
                    edge_count += 1
        
        logging.info(f"Added {edge_count} edges between records above similarity threshold {threshold}")
        
        # Get connected components
        initial_components = list(nx.connected_components(G))
        logging.info(f"Found {len(initial_components)} connected components")
        
        # Phase 5: Map initial clusters to connected components
        logging.info("Phase 5: Mapping exact match clusters to connected components")
        
        # Create mapping from temp_cluster to component
        temp_to_component = {}
        for component_id, component in enumerate(initial_components):
            # Get all temp clusters in this component
            temp_clusters = set()
            for node_id in component:
                temp_clusters.add(processed_df.iloc[node_id]['temp_cluster'])
            
            # Map each temp cluster to this component
            for temp_cluster in temp_clusters:
                temp_to_component[temp_cluster] = component_id
        
        # Phase 6: Refine clusters with domain coherence
        logging.info("Phase 6: Refining clusters with domain coherence")
        refined_components = self._refine_clusters_with_domain_coherence(
            processed_df, initial_components, similarity_matrix)
        
        # Phase 7: Assign final cluster IDs
        logging.info(f"Phase 7: Assigning final cluster IDs to {len(refined_components)} clusters")
        cluster_assignments = {}
        for cluster_id, cluster in enumerate(refined_components):
            for node_id in cluster:
                cluster_assignments[node_id] = cluster_id
        
        # Add cluster IDs to DataFrame
        processed_df['cluster_id'] = processed_df.index.map(
            lambda x: cluster_assignments.get(x))
        
        # Create canonical forms
        canonical_forms = self._create_canonical_forms(processed_df)
        
        # Add canonical names
        processed_df['canonical_name'] = processed_df['cluster_id'].map(
            lambda x: canonical_forms.get(x, {}).get('canonical_name'))
        
        # Store entity clusters
        self.entity_clusters = canonical_forms
        
        # Log cluster statistics
        cluster_sizes = processed_df.groupby('cluster_id').size()
        logging.info(f"Clustering complete: {len(cluster_sizes)} clusters")
        logging.info(f"  Average cluster size: {cluster_sizes.mean():.2f}")
        logging.info(f"  Largest cluster size: {cluster_sizes.max()}")
        logging.info(f"  Singleton clusters: {sum(cluster_sizes == 1)} ({sum(cluster_sizes == 1)/len(cluster_sizes):.2%})")
        
        # Log disambiguation success cases
        name_clusters = defaultdict(list)
        for _, row in processed_df.iterrows():
            name_clusters[row['normalized_name']].append(row['cluster_id'])
        
        # Find names split across multiple clusters
        disambiguated = {
            name: sorted(list(set(clusters))) 
            for name, clusters in name_clusters.items() 
            if len(set(clusters)) > 1
        }
        
        logging.info(f"Successfully disambiguated {len(disambiguated)} names into multiple entities")
        
        # Log specific examples (e.g., Franz Schubert)
        target_names = ['schubert franz', 'smith john']
        for name in target_names:
            if name in disambiguated:
                logging.info(f"Example disambiguation: '{name}' split into {len(disambiguated[name])} entities")
                for cluster_id in disambiguated[name]:
                    examples = processed_df[processed_df['cluster_id'] == cluster_id].head(1)
                    if not examples.empty:
                        for _, ex in examples.iterrows():
                            domain = ex['domain_context'][:50] + "..." if not pd.isna(ex['domain_context']) else "No domain info"
                            logging.info(f"  - Cluster {cluster_id}: {ex['person']} | {domain}")
        
        return processed_df
    
    def _summarize_domains(self, cluster_records: pd.DataFrame) -> str:
        """
        Create a comprehensive summary of domains for this entity by analyzing
        subjects, genres, and domain context.
        
        Args:
            cluster_records: Records in this cluster
            
        Returns:
            Domain summary string
        """
        from collections import Counter
        
        # Initialize domain collections
        all_domains = []
        
        # First check if primary_domain is available (highest priority)
        if 'primary_domain' in cluster_records.columns:
            for domains in cluster_records['primary_domain'].dropna():
                all_domains.extend([d.strip() for d in domains.split(';') if d.strip()])
        
        # Collect all subject terms
        subjects = []
        for subj in cluster_records['subjects'].dropna():
            subjects.extend([s.strip() for s in subj.split(';')])
        
        # Add genres if available
        genres = []
        if 'genres' in cluster_records.columns:
            for genre in cluster_records['genres'].dropna():
                genres.extend([g.strip() for g in genre.split(';')])
        
        # Count frequencies
        domain_counts = Counter(all_domains)
        subject_counts = Counter(subjects)
        genre_counts = Counter(genres)
        
        # Combine results, prioritizing explicit domains, then subjects, then genres
        all_counts = domain_counts + subject_counts + genre_counts
        
        # Get top terms
        top_domains = [domain for domain, _ in all_counts.most_common(7)]
        
        if top_domains:
            return "; ".join(top_domains)
        else:
            return "Unknown domain"

    def _create_canonical_forms(self, df_with_clusters: pd.DataFrame) -> Dict:
        """
        Create canonical forms for each entity cluster with improved handling of
        ambiguous entities and metadata aggregation.
        
        Args:
            df_with_clusters: DataFrame with cluster assignments
            
        Returns:
            Dictionary of canonical forms
        """
        canonical_forms = {}
        
        for cluster_id in sorted(df_with_clusters['cluster_id'].unique()):
            cluster_records = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id]
            
            # Select canonical name using a priority system:
            # 1. Prefer names with complete birth and death dates
            # 2. Then names with just birth date
            # 3. Then names with most complete metadata
            
            # First, check for names with both birth and death years
            complete_dates = cluster_records[
                (~cluster_records['birth_year'].isna()) & 
                (~cluster_records['death_year'].isna())
            ]
            
            if not complete_dates.empty:
                # Use the record with most complete metadata as canonical
                most_complete = complete_dates.sort_values('context_completeness', ascending=False).iloc[0]
                canonical_name = most_complete['person']
                birth_year = most_complete['birth_year']
                death_year = most_complete['death_year']
            else:
                # Check for records with just birth year
                birth_only = cluster_records[~cluster_records['birth_year'].isna()]
                
                if not birth_only.empty:
                    most_complete = birth_only.sort_values('context_completeness', ascending=False).iloc[0]
                    canonical_name = most_complete['person']
                    birth_year = most_complete['birth_year']
                    death_year = None
                else:
                    # Use record with most complete metadata
                    most_complete = cluster_records.sort_values('context_completeness', ascending=False).iloc[0]
                    canonical_name = most_complete['person']
                    birth_year = None
                    death_year = None
            
            # Aggregate text fields with duplicate elimination and weight by frequency
            aggregated_fields = {
                'roles': [],
                'subjects': [],
                'genres': [],
                'domains': []
            }
            
            # Collect values from non-null fields
            for _, row in cluster_records.iterrows():
                for field in ['roles', 'subjects', 'genres']:
                    if not pd.isna(row[field]) and row[field]:
                        # Split multi-value fields
                        if field in ['subjects', 'genres']:
                            values = [v.strip() for v in row[field].split(';') if v.strip()]
                            aggregated_fields[field].extend(values)
                        else:
                            # Single value fields
                            aggregated_fields[field].append(row[field])
                
                # Handle domains specially
                if not pd.isna(row.get('primary_domain')) and row['primary_domain']:
                    domains = [d.strip() for d in row['primary_domain'].split(';') if d.strip()]
                    aggregated_fields['domains'].extend(domains)
            
            # Count frequency of each value
            counted_fields = {}
            for field, values in aggregated_fields.items():
                counter = Counter(values)
                # Keep only values that appear multiple times or in small clusters
                min_count = 2 if len(cluster_records) > 5 else 1
                counted_fields[field] = [(value, count) for value, count in counter.most_common() 
                                        if count >= min_count]
            
            # Join values with appropriate separators, including frequency information
            joined_fields = {}
            for field, counted_values in counted_fields.items():
                if counted_values:
                    # Format as "value (count)" for each value
                    formatted_values = [f"{value}" for value, _ in counted_values[:5]]
                    joined_fields[field] = '; '.join(formatted_values)
                else:
                    joined_fields[field] = None
            
            # Get representative titles, handling nulls
            titles = []
            for _, row in cluster_records.iterrows():
                if not pd.isna(row['title']) and row['title']:
                    titles.append(row['title'])
            
            sample_titles = '; '.join(titles[:3]) if titles else None
            
            # Create a domain summary - crucial for disambiguation
            domain_summary = self._summarize_domains(cluster_records)
            
            # Combine all info into canonical form
            canonical_forms[cluster_id] = {
                'canonical_name': canonical_name,
                'birth_year': int(birth_year) if birth_year is not None else None,
                'death_year': int(death_year) if death_year is not None else None,
                'record_count': len(cluster_records),
                'roles': joined_fields.get('roles'),
                'subjects': joined_fields.get('subjects'),
                'genres': joined_fields.get('genres'),
                'domains': joined_fields.get('domains'),
                'sample_titles': sample_titles,
                'domain_summary': domain_summary,
                'context_completeness': round(cluster_records['context_completeness'].mean(), 2)
            }
        
        logging.info(f"Created {len(canonical_forms)} canonical entity records")
        return canonical_forms
    
    def evaluate_results(self, 
                    resolved_df: pd.DataFrame, 
                    ground_truth_df: pd.DataFrame) -> Dict:
        """
        Evaluate entity resolution results against ground truth with enhanced metrics.
        
        Args:
            resolved_df: DataFrame with resolved entities
            ground_truth_df: DataFrame with ground truth matches
            
        Returns:
            Evaluation metrics
        """
        # Log evaluation start
        logging.info(f"Evaluating resolution results against {len(ground_truth_df)} ground truth pairs")
        
        # Basic cluster statistics
        metrics = {
            'num_clusters': len(set(resolved_df['cluster_id'])),
            'avg_cluster_size': resolved_df.groupby('cluster_id').size().mean(),
            'max_cluster_size': resolved_df.groupby('cluster_id').size().max(),
            'singleton_clusters': sum(resolved_df.groupby('cluster_id').size() == 1),
            'multi_record_clusters': sum(resolved_df.groupby('cluster_id').size() > 1)
        }
        
        # Add percentages for better interpretation
        metrics['singleton_percent'] = metrics['singleton_clusters'] / metrics['num_clusters']
        metrics['multi_record_percent'] = metrics['multi_record_clusters'] / metrics['num_clusters']
        
        # Create ID to cluster mapping
        id_to_cluster = dict(zip(resolved_df['id'], resolved_df['cluster_id']))
        
        # Evaluate against ground truth
        y_true = []
        y_pred = []
        pair_details = []
        
        # Collect pairs by type for detailed analysis
        true_positives = []
        false_positives = []
        false_negatives = []
        
        # Track pairs with missing IDs
        missing_ids = 0
        
        for _, pair in ground_truth_df.iterrows():
            left_id = pair['left']
            right_id = pair['right']
            is_match = bool(pair['match'])  # Ensure boolean type
            
            # Skip if either ID is not in dataset
            if left_id not in id_to_cluster or right_id not in id_to_cluster:
                missing_ids += 1
                continue
                
            left_cluster = id_to_cluster[left_id]
            right_cluster = id_to_cluster[right_id]
            
            # Check if same cluster in results
            predicted_match = (left_cluster == right_cluster)
            
            # Create a detailed description for this pair
            left_record = resolved_df[resolved_df['id'] == left_id].iloc[0]
            right_record = resolved_df[resolved_df['id'] == right_id].iloc[0]
            
            pair_info = {
                'left_id': left_id,
                'right_id': right_id,
                'left_name': left_record['person'],
                'right_name': right_record['person'],
                'left_cluster': left_cluster,
                'right_cluster': right_cluster,
                'true_match': is_match,
                'predicted_match': predicted_match,
                'correct': is_match == predicted_match,
                'error_type': None
            }
            
            # Classify error types
            if is_match and not predicted_match:
                pair_info['error_type'] = 'false_negative'
                false_negatives.append(pair_info)
            elif not is_match and predicted_match:
                pair_info['error_type'] = 'false_positive'
                false_positives.append(pair_info)
            elif is_match and predicted_match:
                true_positives.append(pair_info)
            
            # Add to evaluation arrays
            y_true.append(1 if is_match else 0)
            y_pred.append(1 if predicted_match else 0)
            pair_details.append(pair_info)
        
        # Log missing IDs
        if missing_ids > 0:
            logging.warning(f"Skipped {missing_ids} ground truth pairs due to missing IDs in results")
        
        # Calculate metrics
        if y_true:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            metrics['accuracy'] = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            metrics['evaluated_pairs'] = len(y_true)
            metrics['missing_pairs'] = missing_ids
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['true_negative'] = cm[0, 0] if cm.shape == (2, 2) else 0
            metrics['false_positive'] = cm[0, 1] if cm.shape == (2, 2) else 0
            metrics['false_negative'] = cm[1, 0] if cm.shape == (2, 2) else 0
            metrics['true_positive'] = cm[1, 1] if cm.shape == (2, 2) else 0
            
            # Log detailed metrics
            logging.info(f"Evaluation metrics:")
            logging.info(f"  Precision: {metrics['precision']:.3f}")
            logging.info(f"  Recall: {metrics['recall']:.3f}")
            logging.info(f"  F1 Score: {metrics['f1']:.3f}")
            logging.info(f"  Accuracy: {metrics['accuracy']:.3f}")
            
            # Analyze errors
            if false_negatives:
                logging.info(f"\nFalse Negatives (Should be matched but weren't): {len(false_negatives)}")
                # Sample a few false negatives to understand error patterns
                for fn in false_negatives[:min(5, len(false_negatives))]:
                    logging.info(f"  {fn['left_name']} (cluster {fn['left_cluster']}) vs "
                                f"{fn['right_name']} (cluster {fn['right_cluster']})")
            
            if false_positives:
                logging.info(f"\nFalse Positives (Shouldn't be matched but were): {len(false_positives)}")
                # Sample a few false positives
                for fp in false_positives[:min(5, len(false_positives))]:
                    logging.info(f"  {fp['left_name']} and {fp['right_name']} "
                                f"incorrectly matched in cluster {fp['left_cluster']}")
            
            # Store all detailed pair information
            metrics['pair_details'] = pair_details
            metrics['error_count'] = len(false_positives) + len(false_negatives)
        
        # Check for ambiguous name handling
        name_clusters = defaultdict(list)
        for _, row in resolved_df.iterrows():
            name_clusters[row['normalized_name']].append(row['cluster_id'])
        
        # Find names split across multiple clusters (disambiguation success)
        disambiguated = {
            name: sorted(list(set(clusters))) 
            for name, clusters in name_clusters.items() 
            if len(set(clusters)) > 1
        }
        
        metrics['disambiguated_names'] = len(disambiguated)
        metrics['percent_disambiguated'] = len(disambiguated) / len(name_clusters) if name_clusters else 0
        
        # Log disambiguation success
        logging.info(f"\nDisambiguation results:")
        logging.info(f"  Successfully disambiguated {len(disambiguated)} names")
        
        # Check for specific test cases (e.g., Franz Schubert)
        test_names = ['schubert franz', 'smith john', 'mueller john']
        for test_name in test_names:
            if test_name in disambiguated:
                logging.info(f"  Test case: '{test_name}' split into {len(disambiguated[test_name])} entities")
        
        return metrics
    
    def train_with_ground_truth(self, 
                           df: pd.DataFrame, 
                           ground_truth_df: pd.DataFrame,
                           vector_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the resolver using ground truth data.
        
        Args:
            df: DataFrame with entity records
            ground_truth_df: DataFrame with ground truth matches
            vector_df: Optional DataFrame with vector embeddings
            
        Returns:
            Dictionary with training metrics
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        logging.info("Training with ground truth data and vector enhancement")
        
        # Preprocess data with vector enhancement
        processed_df = self.preprocess_data(df, vector_df)
        
        # Create ID lookup
        id_to_index = {row['id']: i for i, row in processed_df.iterrows()}
        
        # Create features and labels for training
        X_train = []
        y_train = []
        pair_ids = []
        
        # Calculate features for all ground truth pairs
        for _, pair in ground_truth_df.iterrows():
            left_id = pair['left']
            right_id = pair['right']
            is_match = pair['match']
            
            # Skip if either ID is not in dataset
            if left_id not in id_to_index or right_id not in id_to_index:
                continue
                
            left_idx = id_to_index[left_id]
            right_idx = id_to_index[right_id]
            
            left_record = processed_df.iloc[left_idx]
            right_record = processed_df.iloc[right_idx]
            
            # Calculate name similarity
            name_sim = self.calculate_name_similarity(
                left_record['normalized_name'],
                right_record['normalized_name']
            )
            
            # Calculate vector similarity if available
            vector_sim = 0.0
            if vector_df is not None:
                if left_record['has_vector'] and right_record['has_vector']:
                    vector_sim = self.calculate_vector_similarity(
                        left_record['id'], right_record['id'], vector_df)
            
            # Calculate context similarity
            context_sim = self._calculate_field_similarity(
                left_record['context'],
                right_record['context']
            ) or 0.0
            
            # Calculate domain similarity
            domain_sim = self.calculate_domain_similarity(
                left_record, right_record)
            
            # Calculate temporal similarity
            temporal_sim = self.calculate_temporal_similarity(
                left_record, right_record)
            
            # Create feature vector
            features = [
                name_sim,
                vector_sim,
                context_sim,
                domain_sim,
                temporal_sim
            ]
            
            X_train.append(features)
            y_train.append(1 if is_match else 0)
            pair_ids.append((left_id, right_id))
        
        if not X_train:
            raise ValueError("No valid training pairs")
                
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train using RandomForest for better generalization
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        
        # Use cross-validation for evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = {
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            model.fit(X_fold_train, y_fold_train)
            y_fold_pred = model.predict(X_fold_val)
            
            cv_metrics['precision'].append(precision_score(y_fold_val, y_fold_pred))
            cv_metrics['recall'].append(recall_score(y_fold_val, y_fold_pred))
            cv_metrics['f1'].append(f1_score(y_fold_val, y_fold_pred))
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = model.feature_importances_
        feature_names = [
            'name_similarity',
            'vector_similarity',
            'context_similarity',
            'domain_similarity',
            'temporal_similarity'
        ]
        
        # Update metric weights based on learned importance
        for i, name in enumerate(feature_names):
            self.metric_weights[name] = feature_importance[i]
        
        # Log vector similarity stats to see if it's providing signal
        vector_sims = [features[1] for features in X_train]  # Assuming vector_sim is the second feature
        logging.info(f"Vector similarity stats in training: min={min(vector_sims):.4f}, max={max(vector_sims):.4f}, mean={np.mean(vector_sims):.4f}")

        # Ensure vector similarity has some minimum importance
        if self.metric_weights.get('vector_similarity', 0) < 0.1:
            logging.info("Enforcing minimum weight for vector similarity")
            self.metric_weights['vector_similarity'] = 0.1
            
            # Normalize weights to sum to 1
            weight_sum = sum(self.metric_weights.values())
            for key in self.metric_weights:
                self.metric_weights[key] /= weight_sum
        
        # Normalize to sum to 1
        weight_sum = sum(self.metric_weights.values())
        for key in self.metric_weights:
            self.metric_weights[key] /= weight_sum
        
        # Find optimal threshold
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 1.0, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_train, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Store model and parameters
        self.model = model
        self.min_similarity = best_threshold
        self.is_trained = True
        
        # Analyze errors
        y_pred = model.predict(X_train)
        errors = [(i, pair_ids[i]) for i in range(len(y_train)) if y_train[i] != y_pred[i]]
        
        # Return training metrics
        metrics = {
            'feature_importance': dict(zip(feature_names, feature_importance)),
            'metric_weights': self.metric_weights,
            'optimal_threshold': best_threshold,
            'cv_precision': np.mean(cv_metrics['precision']),
            'cv_recall': np.mean(cv_metrics['recall']),
            'cv_f1': np.mean(cv_metrics['f1']),
            'error_count': len(errors)
        }
        
        logging.info(f"Training complete. Optimal threshold: {best_threshold:.2f}")
        logging.info(f"Metric weights: {self.metric_weights}")
        logging.info(f"Error count: {len(errors)}")
        
        return metrics
    
    def export_canonical_entities(self, filepath: str) -> pd.DataFrame:
        """
        Export canonical entities to a file.
        
        Args:
            filepath: Path to save canonical entities
            
        Returns:
            DataFrame with canonical entities
        """
        if not self.entity_clusters:
            raise ValueError("Must run resolve_entities first")
            
        canonical_df = pd.DataFrame.from_dict(
            self.entity_clusters,
            orient='index'
        )
        
        canonical_df.to_csv(filepath, index_label='cluster_id')
        logging.info(f"Exported {len(canonical_df)} canonical entities to {filepath}")
        
        return canonical_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        import pickle
        
        model_data = {
            'metric_weights': self.metric_weights,
            'min_similarity': self.min_similarity,
            'is_trained': self.is_trained,
            'vector_dim': self.vector_dim
        }
        
        if hasattr(self, 'model'):
            # Save sklearn model separately
            model_path = filepath + '.sklearn'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            model_data['model_path'] = model_path
        
        # Save configuration
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
            
        logging.info(f"Saved model to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        import pickle
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.metric_weights = model_data['metric_weights']
        self.min_similarity = model_data['min_similarity']
        self.is_trained = model_data['is_trained']
        self.vector_dim = model_data.get('vector_dim', 3072)
        
        if 'model_path' in model_data and os.path.exists(model_data['model_path']):
            with open(model_data['model_path'], 'rb') as f:
                self.model = pickle.load(f)
                
        logging.info(f"Loaded model from {filepath}")

def load_vector_data(entity_df: pd.DataFrame, vector_file: str) -> pd.DataFrame:
    """
    Load vector data from a CSV file, ensuring proper ID indexing and conversion.
    Each row contains multiple vector columns (e.g., 'title_vector', 'person_vector').
    Fields that are NaN or non-numeric strings are skipped.

    Returns:
        DataFrame where each vector column contains NumPy arrays instead of strings.
    """
    print(f"📂 Loading vector data from {vector_file}")

    try:
        # Load CSV file
        vector_df = pd.read_csv(vector_file, header=0)

        # Print raw column names for debugging
        print("📌 Raw vector_df columns:", vector_df.columns.tolist())

        # Normalize column names (strip spaces, lowercase)
        vector_df.columns = vector_df.columns.str.strip().str.lower()

        # Ensure the ID column is properly set
        if 'id' in vector_df.columns:
            vector_df = vector_df.set_index('id')
            vector_df.index = vector_df.index.astype(str)
        else:
            print("⚠️ No valid ID column found in vector data. Matching may fail.")

        # Function to safely parse vector strings into NumPy arrays
        def safe_parse_vector(vector_str):
            """Safely parse vector strings into NumPy arrays, skipping invalid data."""
            if pd.isna(vector_str) or not isinstance(vector_str, str) or '[' not in vector_str:
                return None  # Skip invalid or non-vector fields
            
            try:
                parsed_vector = np.array(ast.literal_eval(vector_str), dtype=np.float32)
                
                # Ensure it's a valid numerical vector
                if parsed_vector.ndim == 1 and parsed_vector.size > 0:
                    return parsed_vector
                else:
                    return None  # Skip malformed vectors
            except Exception as e:
                print(f"❌ Error parsing vector: {vector_str} - {e}")
                return None  # Skip bad vectors

        # Identify all vector columns (excluding non-vector metadata)
        vector_columns = [col for col in vector_df.columns if vector_df[col].notna().sum() > 0]  # Ignore empty cols

        valid_vector_columns = []
        for col in vector_columns:
            # Check if the first non-null value is a serialized vector
            first_valid_value = vector_df[col].dropna().iloc[0] if not vector_df[col].dropna().empty else None
            if isinstance(first_valid_value, str) and '[' in first_valid_value:
                valid_vector_columns.append(col)

        if valid_vector_columns:
            print(f"✅ Detected vector columns: {valid_vector_columns}")

            # Apply conversion to all detected vector columns
            for col in valid_vector_columns:
                vector_df[col] = vector_df[col].apply(safe_parse_vector)

            # Drop columns where all values are None after processing
            vector_df = vector_df.dropna(axis=1, how='all')

        print(f"✅ Successfully loaded {len(vector_df)} records with vector embeddings.")
        return vector_df

    except Exception as e:
        print(f"❌ Error loading vector data: {e}")
        return pd.DataFrame()