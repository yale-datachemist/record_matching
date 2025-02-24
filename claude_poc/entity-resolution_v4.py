import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from typing import List, Tuple, Dict, Optional, Union
import json
from collections import defaultdict, Counter
import os
import zlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from qdrant_client import QdrantClient
from qdrant_client.http import models
import warnings
from Levenshtein import distance as levenshtein_distance
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


logging.basicConfig(level=logging.INFO, filename="output.log",filemode="w")

logger = logging.getLogger(__name__)

# Reduce Qdrant client logging
logging.getLogger('qdrant_client').setLevel(logging.WARNING)

BASE_PATH = "/Users/tt434/Dropbox/YUL/2025/msu/code"

def generate_stable_hash(s: str) -> int:
    """Generate a stable hash for string IDs that's compatible with Qdrant"""
    return zlib.crc32(str(s).encode()) & 0xffffffff

# Expanded vector fields to include all relevant metadata
VECTOR_FIELDS = [
    'record',
    'title',
    'attribution',
    'provision',
    'subjects',
    'genres',
    #'relatedWork'
    #'person'  # Added person field for improved matching
]


IMPUTATION_FIELDS = [
    'genres',
    'attribution', 
    'subjects',
    'provision'
]

# Features now include name similarity, record threshold, and imputation flags
ALL_FEATURES = (VECTOR_FIELDS + 
                ['name_similarity', 'record_threshold'] + 
                [f'{field}_is_original' for field in IMPUTATION_FIELDS])

# Additional blocking fields for better candidate generation
BLOCKING_FIELD = ['person']  # Added title as blocking field

class PrecisionRules:
    """Enhanced configuration class for precision rules and thresholds"""
    def __init__(self,
                 title_threshold: float = 0.45,  # Lowered from 0.5
                 title_penalty: float = 0.2,    # Reduced penalty
                 record_threshold: float = 0.5,  # Lowered from 0.6
                 record_penalty: float = 0.3,    # Reduced penalty
                 high_sim_threshold: float = 0.5, # Lowered from 0.6
                 high_sim_required: int = 2,      # Reduced from 3
                 low_sim_threshold: float = 0.3,  # Lowered from 0.4
                 low_sim_max: int = 3,           # Increased from 2
                 low_sim_penalty: float = 0.3):  # Reduced penalty
        self.title_threshold = title_threshold
        self.title_penalty = title_penalty
        self.record_threshold = record_threshold
        self.record_penalty = record_penalty
        self.high_sim_threshold = high_sim_threshold
        self.high_sim_required = high_sim_required
        self.low_sim_threshold = low_sim_threshold
        self.low_sim_max = low_sim_max
        self.low_sim_penalty = low_sim_penalty

class MatchAnalysis:
    """Class to store and analyze match prediction details"""
    def __init__(self):
        self.scores = []
        self.field_similarities = []
        self.ids = []
        
    def add_prediction(self, id1: str, id2: str, base_score: float, final_score: float, 
                      rules_applied: List[str], similarities: Dict[str, float], 
                      confidence: float = None):
        """Store prediction details for analysis"""
        self.ids.append((id1, id2))
        self.scores.append(final_score)  # Just store the final score
        self.field_similarities.append(similarities)

    def get_analysis_summary(self) -> Dict:
        """Generate summary statistics of match analysis"""
        return {
            "total_predictions": len(self.ids),
            "avg_score": np.mean(self.scores) if self.scores else 0,
            "min_score": min(self.scores) if self.scores else 0,
            "max_score": max(self.scores) if self.scores else 0
        }

class EntityResolver:
    def __init__(self, plot_dir: str = None):
        self.qdrant_client = QdrantClient("localhost", port=6333)
        self.id_mapping = {}
        self.match_analysis = MatchAnalysis()
        self.weights = None
        self.field_importance = None
        self.plot_dir = plot_dir

    def analyze_blocking_keys(self, catalog_df: pd.DataFrame, matches_df: pd.DataFrame) -> Dict:
        """
        Analyze blocking key (person vector) similarities for ground truth matches
        """
        try:
            # Filter for true matches only
            true_matches = matches_df[matches_df['match'] == True]
            logger.info(f"Analyzing blocking keys for {len(true_matches)} true matches")
            
            # Calculate similarities for all true matches
            similarities = []
            missing_vectors = []
            
            for _, match in true_matches.iterrows():
                try:
                    # Get person vectors for both records
                    record1 = catalog_df[catalog_df['id'] == match['left']].iloc[0]
                    record2 = catalog_df[catalog_df['id'] == match['right']].iloc[0]
                    
                    vec1 = self.get_vector_from_field(record1['person'])
                    vec2 = self.get_vector_from_field(record2['person'])
                    
                    if vec1 is not None and vec2 is not None:
                        # Calculate cosine similarity
                        similarity = float(cosine_similarity(
                            vec1.reshape(1, -1),
                            vec2.reshape(1, -1)
                        )[0][0])
                        
                        similarities.append({
                            'left_id': match['left'],
                            'right_id': match['right'],
                            'similarity': similarity
                        })
                    else:
                        missing_vectors.append({
                            'left_id': match['left'],
                            'right_id': match['right'],
                            'reason': 'missing_vector'
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing match {match['left']}, {match['right']}: {str(e)}")
                    continue
            
            # Convert to DataFrame for analysis
            sim_df = pd.DataFrame(similarities)
            
            # Calculate statistics
            stats = {
                'total_true_matches': len(true_matches),
                'matches_with_vectors': len(similarities),
                'matches_missing_vectors': len(missing_vectors),
                'mean_similarity': float(sim_df['similarity'].mean()) if not sim_df.empty else 0.0,
                'min_similarity': float(sim_df['similarity'].min()) if not sim_df.empty else 0.0,
                'max_similarity': float(sim_df['similarity'].max()) if not sim_df.empty else 0.0,
                'std_similarity': float(sim_df['similarity'].std()) if not sim_df.empty else 0.0,
                'below_threshold': len(sim_df[sim_df['similarity'] < 0.70]) if not sim_df.empty else 0,
                'above_threshold': len(sim_df[sim_df['similarity'] >= 0.70]) if not sim_df.empty else 0,
            }
            
            # Create visualizations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = os.path.join(BASE_PATH, f"blocking_analysis_{timestamp}")
            os.makedirs(plot_dir, exist_ok=True)
            
            if not sim_df.empty:
                # 1. Histogram of similarities
                plt.figure(figsize=(12, 6))
                plt.hist(sim_df['similarity'], bins=50, edgecolor='black')
                plt.axvline(x=0.80, color='r', linestyle='--', label='Threshold (0.80)')
                plt.title('Distribution of Blocking Key Similarities for True Matches')
                plt.xlabel('Cosine Similarity')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plot_dir, 'similarity_distribution.png'))
                plt.close()
            
            # 2. Save detailed report
            report_path = os.path.join(plot_dir, 'blocking_analysis_report.txt')
            with open(report_path, 'w') as f:
                f.write("Blocking Key Analysis Report\n")
                f.write("===========================\n\n")
                
                f.write("Summary Statistics:\n")
                f.write("-----------------\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                
                if not sim_df.empty:
                    f.write("\nSimilarity Distribution:\n")
                    f.write("----------------------\n")
                    
                    # Calculate percentiles
                    percentiles = [0, 10, 25, 50, 75, 90, 100]
                    for p in percentiles:
                        value = float(sim_df['similarity'].quantile(p/100))
                        f.write(f"{p}th percentile: {value:.3f}\n")
                    
                    f.write("\nPotentially Problematic Matches (similarity < 0.70):\n")
                    f.write("----------------------------------------------\n")
                    problem_matches = sim_df[sim_df['similarity'] < 0.70].sort_values('similarity')
                    for _, row in problem_matches.iterrows():
                        f.write(f"Match: {row['left_id']} - {row['right_id']}, ")
                        f.write(f"Similarity: {row['similarity']:.3f}\n")
                
                if missing_vectors:
                    f.write("\nMatches with Missing Vectors:\n")
                    f.write("--------------------------\n")
                    for case in missing_vectors:
                        f.write(f"Match: {case['left_id']} - {case['right_id']}, ")
                        f.write(f"Reason: {case['reason']}\n")
            
            logger.info(f"Blocking analysis saved to: {plot_dir}")
            return {
                'stats': stats,
                'plot_dir': plot_dir,
                'report_path': report_path,
                'similarity_data': sim_df.to_dict('records') if not sim_df.empty else []
            }
            
        except Exception as e:
            logger.error(f"Error in blocking key analysis: {str(e)}")
            raise

    def load_data(self, catalog_file: str, ground_truth_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Enhanced data loading with validation"""
        try:
            catalog_df = pd.read_csv(catalog_file)
            matches_df = pd.read_csv(ground_truth_file)
            
            # Validate required columns
            required_cols = ['id'] + VECTOR_FIELDS
            missing_cols = [col for col in required_cols if col not in catalog_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in catalog data: {missing_cols}")
            
            # Validate matches dataframe
            required_match_cols = ['left', 'right', 'match']
            missing_match_cols = [col for col in required_match_cols if col not in matches_df.columns]
            if missing_match_cols:
                raise ValueError(f"Missing required columns in matches data: {missing_match_cols}")
            
            logger.info(f"Loaded {len(catalog_df)} catalog records and {len(matches_df)} ground truth pairs")
            return catalog_df, matches_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def setup_vector_index(self, collection_name: str = "entity_vectors"):
        """Initialize Qdrant vector database with optimized settings"""
        try:
            if self.qdrant_client.collection_exists(collection_name):
                self.qdrant_client.delete_collection(collection_name)
                
            # Create collection with optimized parameters
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=3072,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=10,
                    indexing_threshold=20000,
                    memmap_threshold=50000
                )
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error setting up Qdrant: {str(e)}")
            raise
            
    def get_ann_neighbors(self, person_vector: np.ndarray, 
                            collection_name: str = "entity_vectors",
                            threshold: float = 0.70) -> List[str]:
        """Get nearest neighbors above similarity threshold using Qdrant"""
        try:
            if person_vector is None:
                return []
                
            neighbors = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=person_vector.tolist(),
                limit=100,  # Get more candidates as we'll filter by threshold
                score_threshold=threshold  # Only get matches above threshold
            )
            
            # Get original IDs and scores
            neighbor_info = [(self.id_mapping[p.id], p.score) for p in neighbors]
            logger.debug(f"Found {len(neighbor_info)} neighbors above threshold {threshold}")
            
            return neighbor_info
            
        except Exception as e:
            logger.error(f"Error getting ANN neighbors: {str(e)}")
            return []

    def generate_comparison_pairs(self, catalog_df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Generate pairs for comparison using ANN blocking"""
        comparison_pairs = set()
        processed_records = set()
        
        for idx, row in catalog_df.iterrows():
            try:
                person_vector = self.get_vector_from_field(row['person'])
                if person_vector is None:
                    continue
                
                # Get neighbors above threshold
                neighbors = self.get_ann_neighbors(person_vector)
                
                # Generate pairs within neighborhood
                record_id = str(row['id'])
                for neighbor_id, score in neighbors:
                    if neighbor_id != record_id:
                        # Order pairs consistently
                        pair = tuple(sorted([record_id, neighbor_id]))
                        if pair not in processed_records:
                            comparison_pairs.add(pair)
                            processed_records.add(pair)
                
            except Exception as e:
                logger.error(f"Error generating pairs for record {row['id']}: {str(e)}")
                continue
                
        logger.info(f"Generated {len(comparison_pairs)} comparison pairs using ANN blocking")
        return list(comparison_pairs)

    def impute_vector_field(self, record_id: str, field: str, catalog_df: pd.DataFrame, 
                       collection_name: str = "entity_vectors") -> Optional[np.ndarray]:
        """Impute missing vector field using neighborhood or global average"""
        try:
            # Get person vector for the record
            record = catalog_df[catalog_df['id'] == record_id].iloc[0]
            blocking_vector = self.get_vector_from_field(record[BLOCKING_FIELD])
            
            if blocking_vector is None:
                logger.debug(f"Cannot get neighbors for record {record_id}: No blocking vector")
                # Try global average
                if self.global_averages[field] is not None:
                    logger.debug(f"Using global average for {field} (no blocking vector)")
                    return self.global_averages[field]
                return None
                    
            # Get neighbors above threshold
            neighbors = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=blocking_vector.tolist(),
                limit=100,
                score_threshold=0.70
            )
            
            # Get neighbor IDs and their records
            neighbor_ids = [self.id_mapping[p.id] for p in neighbors if p.id in self.id_mapping]
            neighbor_records = catalog_df[catalog_df['id'].isin(neighbor_ids)]
            
            # Get field vectors from neighbors
            valid_vectors = []
            for _, nbr in neighbor_records.iterrows():
                if nbr['id'] != record_id:  # Exclude self
                    vec = self.get_vector_from_field(nbr[field])
                    if vec is not None and not np.isnan(vec).any():
                        valid_vectors.append(vec)
            
            if valid_vectors:
                return np.mean(valid_vectors, axis=0)
            else:
                # Try global average before giving up
                if self.global_averages[field] is not None:
                    logger.debug(f"Using global average for {field} (no valid neighbor vectors)")
                    return self.global_averages[field]
                logger.warning(f"No valid vectors or global average for imputing {field} in record {record_id}")
                return None
                
        except Exception as e:
            logger.debug(f"Error imputing {field} for record {record_id}: {str(e)}")
            # Try global average on error
            if self.global_averages[field] is not None:
                logger.debug(f"Using global average for {field} (after error)")
                return self.global_averages[field]
            return None
    
    def compute_global_averages(self, catalog_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute average vectors for each field across entire dataset"""
        logger.info("Computing global averages for fallback imputation...")
        global_averages = {}
        
        for field in VECTOR_FIELDS:
            valid_vectors = []
            for _, row in catalog_df.iterrows():
                vec = self.get_vector_from_field(row[field])
                if vec is not None and not np.isnan(vec).any():
                    valid_vectors.append(vec)
            
            if valid_vectors:
                global_averages[field] = np.mean(valid_vectors, axis=0)
                logger.info(f"Computed global average for {field} using {len(valid_vectors)} vectors")
            else:
                global_averages[field] = None
                logger.warning(f"No valid vectors found for field {field} in entire dataset")
                
        return global_averages

    def preprocess_catalog(self, catalog_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess catalog data with imputation tracking"""
        processed_df = catalog_df.copy()
        
        # Track which fields were imputed for each record
        self.imputation_status = defaultdict(dict)
        
        # Use the already computed global averages
        if not hasattr(self, 'global_averages'):
            logger.info("Computing global averages...")
            self.global_averages = self.compute_global_averages(catalog_df)
        else:
            logger.info("Using previously computed global averages")
        
        # Track imputation statistics
        imputation_stats = defaultdict(int)
        
        for idx, row in processed_df.iterrows():
            record_id = str(row['id'])
            self.imputation_status[record_id] = {}
            
            for field in VECTOR_FIELDS:
                field_vector = self.get_vector_from_field(row[field])
                
                if field_vector is None:
                    # Track original/imputed status for relevant fields
                    if field in IMPUTATION_FIELDS:
                        self.imputation_status[record_id][field] = False  # False = imputed
                    
                    # First try neighborhood imputation
                    imputed_vector = self.impute_vector_field(row['id'], field, catalog_df)
                    
                    if imputed_vector is not None:
                        processed_df.at[idx, field] = imputed_vector.tolist()
                        imputation_stats[f"{field}_neighborhood"] += 1
                    else:
                        # Fall back to global average if available
                        if self.global_averages[field] is not None:
                            processed_df.at[idx, field] = self.global_averages[field].tolist()
                            imputation_stats[f"{field}_global"] += 1
                        else:
                            # Only use zero vector if no global average available
                            processed_df.at[idx, field] = np.zeros(3072).tolist()
                            imputation_stats[f"{field}_failed"] += 1
                else:
                    # Track original values
                    if field in IMPUTATION_FIELDS:
                        self.imputation_status[record_id][field] = True  # True = original
        
        # Log imputation statistics
        logger.info("\nImputation Statistics:")
        for field in VECTOR_FIELDS:
            logger.info(f"\n{field}:")
            logger.info(f"  Neighborhood imputation: {imputation_stats[f'{field}_neighborhood']}")
            logger.info(f"  Global average fallback: {imputation_stats[f'{field}_global']}")
            logger.info(f"  Complete failures: {imputation_stats[f'{field}_failed']}")
                
        return processed_df


    def get_vector_from_field(self, field_value) -> Optional[np.ndarray]:
        """Enhanced vector parsing with strict type checking"""
        # Handle pd.Series
        if isinstance(field_value, pd.Series):
            if len(field_value) == 1:
                field_value = field_value.iloc[0]
            else:
                return None

        # Handle numpy arrays - MUST come before string check
        if isinstance(field_value, np.ndarray):
            return None if np.isnan(field_value).any() else field_value

        # Handle string "NaN" values or empty strings
        if isinstance(field_value, str):
            if field_value == "NaN" or field_value == "":
                return None
            if field_value.startswith('[') and field_value.endswith(']'):
                try:
                    vector_str = field_value.strip('[]')
                    values = [float('nan') if val.strip() == 'NaN' else float(val) 
                            for val in vector_str.split(',')]
                    vector = np.array(values)
                    return None if np.isnan(vector).any() else vector
                except:
                    return None

        # Handle lists
        if isinstance(field_value, list):
            try:
                vector = np.array(field_value)
                return None if np.isnan(vector).any() else vector
            except:
                return None

        return None

    def compute_pairwise_features(self, record1: pd.Series, record2: pd.Series) -> np.ndarray:
        """Feature computation with imputation flags"""
        features = []
        record_sim = 0.0  # Initialize record similarity
        
        # First compute vector similarities
        for field in VECTOR_FIELDS:
            try:
                # Get vectors
                vec1 = self.get_vector_from_field(record1[field])
                vec2 = self.get_vector_from_field(record2[field])
                
                # Early returns if either vector is None
                if vec1 is None or vec2 is None:
                    logger.debug(f"  {field}: No comparison possible - missing vector(s)")
                    features.append(0.0)
                    continue
                
                # Convert to numpy arrays if not already
                vec1 = np.asarray(vec1, dtype=float)
                vec2 = np.asarray(vec2, dtype=float)
                
                # Check norms with numerical tolerance
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 < 1e-10 or norm2 < 1e-10:
                    logger.debug(f"  {field}: No comparison possible - zero vector(s)")
                    features.append(0.0)
                    continue
                
                # Normalize vectors
                vec1_norm = vec1 / norm1
                vec2_norm = vec2 / norm2
                
                # Compute similarity
                similarity = float(cosine_similarity(
                    vec1_norm.reshape(1, -1),
                    vec2_norm.reshape(1, -1)
                )[0][0])
                
                # Store record similarity for threshold feature
                if field == 'record':
                    record_sim = similarity
                
                # Handle numerical issues
                if not np.isfinite(similarity):
                    logger.debug(f"  {field}: Non-finite similarity")
                    features.append(0.0)
                else:
                    similarity = float(np.clip(similarity, -1.0, 1.0))
                    features.append(similarity)
                    logger.debug(f"  {field}: similarity = {similarity:.4f}")
                
            except Exception as e:
                logger.warning(f"Error computing similarity for field {field} in records {record1['id']}, {record2['id']}: {str(e)}")
                features.append(0.0)
        
        # Add name similarity feature
        try:
            name1 = str(record1['name'])
            name2 = str(record2['name'])
            if name1 == "NaN" or name2 == "NaN" or name1 == "" or name2 == "":
                features.append(1.0)
            else:
                max_len = max(len(name1), len(name2))
                if max_len > 0:
                    similarity = 1.0 - (levenshtein_distance(name1, name2) / max_len)
                    features.append(similarity)
                else:
                    features.append(1.0)
        except Exception as e:
            logger.warning(f"Error computing name similarity for records {record1['id']}, {record2['id']}: {str(e)}")
            features.append(1.0)
        
        # Add record threshold feature
        features.append(1.0 if record_sim > 0.50 else 0.0)
        
        # Add imputation flags (1 if both records have original values, 0 if either was imputed)
        for field in IMPUTATION_FIELDS:
            id1, id2 = str(record1['id']), str(record2['id'])
            is_original1 = self.imputation_status[id1].get(field, False)
            is_original2 = self.imputation_status[id2].get(field, False)
            features.append(1.0 if (is_original1 and is_original2) else 0.0)
        
        return np.array(features)

    def index_vectors(self, df: pd.DataFrame, collection_name: str = "entity_vectors"):
        """Enhanced vector indexing with improved blocking"""
        vectors = []
        
        for idx, row in df.iterrows():
            # Get vectors for all blocking fields
            blocking_vectors = {}
            for field in BLOCKING_FIELD:
                vector = self.get_vector_from_field(row[field])
                if vector is not None:
                    blocking_vectors[field] = vector.tolist()
            
            if not blocking_vectors:
                logger.warning(f"Skipping record {row['id']} - no valid blocking vectors")
                continue

            # Generate stable hash ID
            original_id = str(row['id'])
            hash_id = generate_stable_hash(original_id)
            
            self.id_mapping[hash_id] = original_id
            
            # Create point with multiple vectors
            point = models.PointStruct(
                id=hash_id,
                vector=blocking_vectors['person'],  # Primary vector is already a list
                payload={
                    "original_id": original_id,
                    "blocking_vectors": blocking_vectors
                }
            )
            vectors.append(point)
        
        # Upload vectors in optimized batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.qdrant_client.upload_points(
                collection_name=collection_name,
                points=batch
            )
        
        logger.info(f"Indexed {len(vectors)} vectors")

    def generate_training_pairs(self, catalog_df: pd.DataFrame, 
                              matches_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced training pair generation with negative sampling"""
        X_train = []
        y_train = []
        
        # Process positive matches
        positive_pairs = set()
        for _, row in matches_df.iterrows():
            try:
                if row['match']:  # Only consider positive matches
                    record1 = catalog_df[catalog_df['id'] == row['left']].iloc[0]
                    record2 = catalog_df[catalog_df['id'] == row['right']].iloc[0]
                    
                    features = self.compute_pairwise_features(record1, record2)
                    X_train.append(features)
                    y_train.append(1)
                    positive_pairs.add((row['left'], row['right']))
                    
            except Exception as e:
                logger.warning(f"Error processing positive match pair: {str(e)}")
                continue
        
        # Generate hard negative examples
        num_positive = len(positive_pairs)
        negative_pairs = set()
        
        while len(negative_pairs) < num_positive:
            idx1, idx2 = np.random.choice(len(catalog_df), 2, replace=False)
            id1 = catalog_df.iloc[idx1]['id']
            id2 = catalog_df.iloc[idx2]['id']
            
            pair = tuple(sorted([id1, id2]))
            if pair not in positive_pairs and pair not in negative_pairs:
                try:
                    record1 = catalog_df.iloc[idx1]
                    record2 = catalog_df.iloc[idx2]
                    
                    # Only add as negative if some fields are similar
                    features = self.compute_pairwise_features(record1, record2)
                    if np.max(features) > 0.3:  # Threshold for hard negatives
                        X_train.append(features)
                        y_train.append(0)
                        negative_pairs.add(pair)
                        
                except Exception as e:
                    logger.warning(f"Error processing negative pair: {str(e)}")
                    continue
        
        return np.array(X_train), np.array(y_train)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid implementation"""
        mask = x >= 0
        result = np.zeros_like(x)
        
        result[mask] = 1 / (1 + np.exp(-x[mask]))
        exp_x = np.exp(x[~mask])
        result[~mask] = exp_x / (1 + exp_x)
        
        return result

    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train classifier with enhanced diagnostics"""
        n_features = X_train.shape[1]
        logger.info(f"Training classifier with {n_features} features")
        logger.info(f"Training data: X shape {X_train.shape}, y shape {y_train.shape}")
        
        # Check for data issues
        logger.info(f"X_train statistics: mean={X_train.mean():.4f}, std={X_train.std():.4f}, min={X_train.min():.4f}, max={X_train.max():.4f}")
        logger.info(f"y_train: positive={sum(y_train)}, negative={len(y_train)-sum(y_train)}")
        
        # Initialize weights with small random values instead of zeros
        self.weights = np.random.randn(n_features) * 0.01
        
         # Reduce L2 regularization
        learning_rate = 0.01
        lambda_l2 = 0.001  # Reduced from 0.1
        n_iterations = 50000
        patience = 5000
        min_improvement = 1e-7
        
        # Initialize tracking variables
        best_loss = float('inf')
        best_weights = None
        patience_counter = 0
        loss_history = []
        weight_history = []  # Track weight evolution
        
        # Create timestamp and directory for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = os.path.join(BASE_PATH, f"training_plots_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            for i in range(n_iterations):
                # Store current weights
                weight_history.append(self.weights.copy())
                
                # Forward pass
                scores = np.dot(X_train, self.weights)
                predictions = self.sigmoid(scores)
                
                # Compute stable cross-entropy loss
                epsilon = 1e-15
                predictions = np.clip(predictions, epsilon, 1 - epsilon)
                loss = -np.mean(
                    y_train * np.log(predictions) + 
                    (1 - y_train) * np.log(1 - predictions)
                )
                
                # Add L2 regularization
                loss += lambda_l2 * np.sum(self.weights ** 2)
                loss_history.append(loss)
                
                # Check for improvement
                if loss < best_loss - min_improvement:
                    best_loss = loss
                    best_weights = self.weights.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at iteration {i}")
                        break
                
                # Compute gradients
                gradients = np.dot(X_train.T, (predictions - y_train)) / len(y_train)
                gradients += 2 * lambda_l2 * self.weights  # Stronger L2 regularization gradient
                
                # Update weights
                self.weights -= learning_rate * gradients
                
                # Log progress
                if i % 1000 == 0:
                    logger.info(f"Iteration {i}, loss: {loss:.4f}")
                    
                    # Log max weight magnitude for monitoring
                    max_weight = np.max(np.abs(self.weights))
                    logger.info(f"Max weight magnitude: {max_weight:.4f}")
            
            # Restore best weights
            if best_weights is not None:
                self.weights = best_weights
            
            # Calculate and log field importance (relative weights)
            total_importance = np.sum(np.abs(self.weights))
            if total_importance > 0:
                self.field_importance = dict(zip(ALL_FEATURES, 
                                            self.weights / total_importance))
                
            logger.info("\nTraining Summary:")
            logger.info(f"Initial loss: {loss_history[0]:.4f}")
            logger.info(f"Final loss: {loss_history[-1]:.4f}")
            logger.info(f"Best loss: {best_loss:.4f}")
            
            # Log both actual weights and relative importance
            logger.info("\nActual Weights:")
            for field, weight in zip(ALL_FEATURES, self.weights):
                logger.info(f"{field:15} weight: {weight:8.4f}")
                
            logger.info("\nField Importance (Normalized Weights):")
            for field, importance in sorted(self.field_importance.items(), 
                                        key=lambda x: abs(x[1]), reverse=True):
                logger.info(f"{field:15} importance: {importance:8.4f}")
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

                
            # Create visualizations
            # self.plot_training_history(loss_history, weight_history)
            # self.create_training_visualizations(plot_dir)
            
            #logger.info(f"Training visualizations saved to: {plot_dir}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def plot_training_history(self, loss_history: List[float], weight_history: List[np.ndarray]) -> None:
        """Create plots showing the training progress and weight evolution"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = os.path.join(BASE_PATH, f"training_plots_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            # Plot 1: Loss over time
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history)
            plt.title('Loss vs. Training Iteration')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.yscale('log')  # Log scale to better show loss changes
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, 'loss_history.png'))
            plt.close()
            
            # Plot 2: Weight evolution
            weight_history = np.array(weight_history)
            plt.figure(figsize=(12, 6))
            for i, field in enumerate(VECTOR_FIELDS):
                plt.plot(weight_history[:, i], label=field)
            plt.title('Weight Evolution During Training')
            plt.xlabel('Iteration')
            plt.ylabel('Weight Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'weight_evolution.png'))
            plt.close()
            
            # Plot 3: Final weight comparison
            plt.figure(figsize=(10, 6))
            plt.bar(VECTOR_FIELDS, self.weights)
            plt.title('Final Field Weights')
            plt.xticks(rotation=45)
            plt.ylabel('Weight Value')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'final_weights.png'))
            plt.close()
            
            logger.info(f"Training plots saved to: {plot_dir}")
            
        except Exception as e:
            logger.error(f"Error creating training plots: {str(e)}")

    def predict_matches(self, catalog_df: pd.DataFrame, 
                   test_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """Make predictions based purely on computed similarities and trained weights"""
        predictions = []
        
        for id1, id2 in test_pairs:
            try:
                record1 = catalog_df[catalog_df['id'] == id1].iloc[0]
                record2 = catalog_df[catalog_df['id'] == id2].iloc[0]
                
                # Compute similarities for each field
                features = self.compute_pairwise_features(record1, record2)
                
                # Debug output for specific pair
                #if id1 == "10810231#Agent600-42" and id2 == "14668405#Agent700-41":
                logger.info(f"Debug for specific pair: {id1} and {id2}")
                logger.info(f"Features: {features}")
                logger.info(f"Weights: {self.weights}")
                dot_product = np.dot(features, self.weights)
                logger.info(f"Dot product: {dot_product}")
                score = float(self.sigmoid(dot_product))
                logger.info(f"Score after sigmoid: {score}")

                # Get base prediction score
                score = float(self.sigmoid(np.dot(features, self.weights)))
                
                # Store similarities for analysis
                field_sims = dict(zip(ALL_FEATURES, features))
                
                # Store prediction info - now matching the method signature
                self.match_analysis.add_prediction(
                    id1, id2,  # record IDs
                    score, score,  # base and final scores are the same
                    [],  # no rules applied
                    field_sims,  # field similarities
                    score  # confidence is just the score
                )
                
                predictions.append((id1, id2, score))
                
            except Exception as e:
                logger.error(f"Error predicting match for pair {id1}, {id2}: {str(e)}")
                continue
        
        # After generating all predictions
        #self.visualize_precision_rules()

        return predictions

    def visualize_precision_rules(self):
        """Create visualizations showing the effect of precision rules"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = os.path.join(BASE_PATH, f"precision_analysis_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            # Plot 1: Score distribution before and after rules
            plt.figure(figsize=(12, 6))
            plt.hist(self.match_analysis.base_scores, alpha=0.5, label='Before Rules', bins=50)
            plt.hist(self.match_analysis.final_scores, alpha=0.5, label='After Rules', bins=50)
            plt.title('Score Distribution Before and After Precision Rules')
            plt.xlabel('Score')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(os.path.join(plot_dir, 'score_distribution.png'))
            plt.close()
            
            # Plot 2: Rule application frequency
            all_rules = [rule for rules in self.match_analysis.applied_rules for rule in rules]
            from collections import Counter
            rule_counts = pd.Series(Counter(all_rules))
            
            if not rule_counts.empty:
                plt.figure(figsize=(10, 6))
                rule_counts.plot(kind='bar')
                plt.title('Frequency of Rule Application')
                plt.xlabel('Rule')
                plt.ylabel('Times Applied')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'rule_frequency.png'))
                plt.close()
            
            # Plot 3: Score reduction by rule
            rule_effects = defaultdict(list)
            for base, final, rules in zip(
                self.match_analysis.base_scores,
                self.match_analysis.final_scores,
                self.match_analysis.applied_rules
            ):
                for rule in rules:
                    rule_effects[rule].append((base - final) / base)
                    
            plt.figure(figsize=(12, 6))
            rule_data = []
            for rule, effects in rule_effects.items():
                rule_data.extend([(rule, effect) for effect in effects])
            if rule_data:
                df = pd.DataFrame(rule_data, columns=['Rule', 'Score Reduction'])
                sns.boxplot(data=df, x='Rule', y='Score Reduction')
                plt.title('Score Reduction by Rule')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'rule_effects.png'))
            plt.close()
            
            # Plot 4: Field similarity distribution for matches vs non-matches
            match_sims = defaultdict(list)
            non_match_sims = defaultdict(list)
            
            for sims, score in zip(self.match_analysis.field_similarities, 
                                 self.match_analysis.final_scores):
                for field, sim in sims.items():
                    if score > 0.5:
                        match_sims[field].append(sim)
                    else:
                        non_match_sims[field].append(sim)
            
            plt.figure(figsize=(15, 8))
            for i, field in enumerate(VECTOR_FIELDS, 1):
                plt.subplot(2, 4, i)
                if match_sims[field]:
                    plt.hist(match_sims[field], alpha=0.5, label='Matches', bins=20)
                if non_match_sims[field]:
                    plt.hist(non_match_sims[field], alpha=0.5, label='Non-matches', bins=20)
                plt.title(field)
                if i == 1:  # Only show legend for first subplot to avoid clutter
                    plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'field_distributions.png'))
            plt.close()
            
            logger.info(f"Precision rule analysis plots saved to: {plot_dir}")
            
            # Generate summary statistics
            with open(os.path.join(plot_dir, 'rule_analysis.txt'), 'w') as f:
                f.write("Precision Rules Analysis\n")
                f.write("=======================\n\n")
                
                f.write("Rule Application Frequency:\n")
                for rule, count in rule_counts.items():
                    f.write(f"{rule}: {count}\n")
                
                f.write("\nAverage Score Reduction by Rule:\n")
                for rule, effects in rule_effects.items():
                    f.write(f"{rule}: {np.mean(effects):.3f}\n")
                    
                f.write("\nField Similarity Statistics for Matches:\n")
                for field in VECTOR_FIELDS:
                    if match_sims[field]:
                        f.write(f"{field}:\n")
                        f.write(f"  Mean: {np.mean(match_sims[field]):.3f}\n")
                        f.write(f"  Std: {np.std(match_sims[field]):.3f}\n")
                        
        except Exception as e:
            logger.error(f"Error creating precision rule plots: {str(e)}")

    def evaluate(self, predictions: List[Tuple[str, str, float]], 
           matches_df: pd.DataFrame,
           threshold: float = 0.5) -> Dict[str, float]:
        """Enhanced evaluation with detailed metrics and visualizations"""
        try:
            # Create timestamp and directory for evaluation plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = os.path.join(BASE_PATH, f"evaluation_plots_{timestamp}")
            os.makedirs(plot_dir, exist_ok=True)

            y_true = []
            y_pred = []
            y_scores = []
            
            # Create lookup dictionary for faster matching
            match_dict = {}
            for _, row in matches_df.iterrows():
                key1 = (row['left'], row['right'])
                key2 = (row['right'], row['left'])
                match_val = int(row['match'])
                match_dict[key1] = match_val
                match_dict[key2] = match_val
            
            # Process predictions in batches
            batch_size = 1000
            for i in range(0, len(predictions), batch_size):
                batch = predictions[i:i + batch_size]
                
                for id1, id2, score in batch:
                    try:
                        # Fast dictionary lookup instead of DataFrame filtering
                        pair_key = (id1, id2)
                        if pair_key in match_dict:
                            y_true.append(match_dict[pair_key])
                            y_pred.append(int(score > threshold))
                            y_scores.append(score)
                    except Exception as e:
                        logger.warning(f"Error evaluating pair {id1}, {id2}: {str(e)}")
                        continue
            
            # Convert to numpy arrays for efficiency
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_scores = np.array(y_scores)
            
            # Calculate metrics
            metrics = {
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred)),
                "threshold": threshold,
                "num_pairs": len(predictions),
                "num_evaluated": len(y_true)
            }
            
            # Calculate confusion matrix
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            metrics.update({
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn)
            })
            
            # Additional metrics
            metrics["accuracy"] = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
            metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            
            # Plot evaluation metrics
            # self.plot_evaluation_metrics(metrics, plot_dir)
            # logger.info(f"Evaluation plots saved to: {plot_dir}")
            
            # Log evaluation results
            logger.info("\nEvaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value}")
            
            return metrics
                
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise

    def save_predictions(self, predictions: List[Tuple[str, str, float]], 
                    catalog_df: pd.DataFrame,
                    matches_df: pd.DataFrame,
                    output_file: str = "predicted_matches.csv") -> pd.DataFrame:
        """Save predictions with imputation status"""
        output_path = os.path.join(BASE_PATH, output_file)
        
        results = []  # Initialize empty results list
        for id1, id2, score in predictions:
            result = {
                'left_id': id1,
                'right_id': id2,
                'confidence_score': score,
                'predicted_match': score > 0.5
            }
            
            # Get ground truth
            ground_truth_match = matches_df[
                ((matches_df['left'] == id1) & (matches_df['right'] == id2)) |
                ((matches_df['left'] == id2) & (matches_df['right'] == id1))
            ]['match'].iloc[0] if len(matches_df[
                ((matches_df['left'] == id1) & (matches_df['right'] == id2)) |
                ((matches_df['left'] == id2) & (matches_df['right'] == id1))
            ]) > 0 else False
            
            result['ground_truth_match'] = ground_truth_match
            result['prediction_correct'] = (result['predicted_match'] == ground_truth_match)
            
            # Get features including blocking similarity
            record1 = catalog_df[catalog_df['id'] == id1].iloc[0]
            record2 = catalog_df[catalog_df['id'] == id2].iloc[0]
            
            # Get blocking similarity
            blocking_vec1 = self.get_vector_from_field(record1[BLOCKING_FIELD])
            blocking_vec2 = self.get_vector_from_field(record2[BLOCKING_FIELD])
            if blocking_vec1 is not None and blocking_vec2 is not None:
                result['blocking_similarity'] = float(cosine_similarity(
                    blocking_vec1.reshape(1, -1),
                    blocking_vec2.reshape(1, -1)
                )[0][0])
            else:
                result['blocking_similarity'] = None
                
            # Get all features
            features = self.compute_pairwise_features(record1, record2)
            
            # Add vector field similarities
            for field, similarity in zip(VECTOR_FIELDS, features[:len(VECTOR_FIELDS)]):
                result[f'{field}_similarity'] = similarity
                
            # Add name similarity
            result['name_similarity'] = features[len(VECTOR_FIELDS)]
            
            # Add record threshold
            result['record_threshold'] = features[len(VECTOR_FIELDS) + 1]
            
            # Add imputation flags
            for i, field in enumerate(IMPUTATION_FIELDS):
                result[f'{field}_is_original'] = features[len(VECTOR_FIELDS) + 2 + i]
            
            results.append(result)
            
        # Now create DataFrame
        pred_df = pd.DataFrame(results)
        
        # Order columns
        columns = [
            'left_id', 'right_id',
            'confidence_score', 
            'predicted_match', 
            'ground_truth_match',
            'prediction_correct',
            'blocking_similarity',
            'name_similarity',
            'record_threshold'
        ] + [f'{field}_similarity' for field in VECTOR_FIELDS] + [
            f'{field}_is_original' for field in IMPUTATION_FIELDS
        ]
        
        pred_df = pred_df[columns]
        
        # Save predictions
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
        
        return pred_df

    def create_detailed_visualizations(self, analysis_dir: str, pred_df: pd.DataFrame) -> None:
        """Create detailed visualizations of matching results and patterns"""
        try:
            # Get similarity columns
            similarity_cols = [col for col in pred_df.columns if col.endswith('_similarity')]
            
            # 1. Distribution plots for each field's similarity scores
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(similarity_cols, 1):
                plt.subplot(3, 3, i)
                sns.histplot(data=pred_df, x=col, bins=50)
                plt.title(col.replace('_similarity', ''))
                plt.xlabel('Similarity Score')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'field_similarity_distributions.png'))
            plt.close()
            
            # 2. Correlation heatmap
            plt.figure(figsize=(12, 10))
            correlation_matrix = pred_df[similarity_cols].corr()
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='RdYlBu', 
                       center=0, 
                       fmt='.2f')
            plt.title('Field Similarity Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'similarity_correlations.png'))
            plt.close()
            
            # 3. Confidence score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data=pred_df, x='confidence_score', bins=50)
            plt.title('Distribution of Confidence Scores')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'confidence_distribution.png'))
            plt.close()
            
            # 4. Field importance comparison
            if self.field_importance:
                plt.figure(figsize=(10, 6))
                importance_data = pd.DataFrame({
                    'field': list(self.field_importance.keys()),
                    'importance': list(self.field_importance.values())
                })
                sns.barplot(data=importance_data, x='field', y='importance')
                plt.title('Field Importance in Matching')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, 'field_importance.png'))
                plt.close()
            
            # 5. Scatter plot matrix for highly correlated fields
            highly_correlated = correlation_matrix.abs() > 0.5
            highly_correlated_fields = [col for col in similarity_cols 
                                      if highly_correlated[col].sum() > 1]
            
            if highly_correlated_fields:
                sns.pairplot(pred_df[highly_correlated_fields], diag_kind='kde')
                plt.savefig(os.path.join(analysis_dir, 'high_correlation_pairs.png'))
                plt.close()
            
            # 6. Match prediction distribution
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=pred_df, x='predicted_match', y='confidence_score')
            plt.title('Confidence Scores by Prediction')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'prediction_confidence.png'))
            plt.close()
            
            # Save descriptive statistics
            stats_file = os.path.join(analysis_dir, 'descriptive_statistics.txt')
            with open(stats_file, 'w') as f:
                f.write("Descriptive Statistics\n")
                f.write("=====================\n\n")
                
                f.write("Field Similarity Statistics:\n")
                for col in similarity_cols:
                    stats = pred_df[col].describe()
                    f.write(f"\n{col}:\n")
                    f.write(f"  Mean: {stats['mean']:.3f}\n")
                    f.write(f"  Std: {stats['std']:.3f}\n")
                    f.write(f"  Min: {stats['min']:.3f}\n")
                    f.write(f"  Max: {stats['max']:.3f}\n")
                
                f.write("\nConfidence Score Statistics:\n")
                conf_stats = pred_df['confidence_score'].describe()
                f.write(f"  Mean: {conf_stats['mean']:.3f}\n")
                f.write(f"  Std: {conf_stats['std']:.3f}\n")
                f.write(f"  Min: {conf_stats['min']:.3f}\n")
                f.write(f"  Max: {conf_stats['max']:.3f}\n")
                
                if self.field_importance:
                    f.write("\nField Importance Scores:\n")
                    for field, importance in self.field_importance.items():
                        f.write(f"  {field}: {importance:.3f}\n")
            
            logger.info(f"Created detailed visualizations in: {analysis_dir}")
            
        except Exception as e:
            logger.error(f"Error creating detailed visualizations: {str(e)}")
    
    def precompute_global_averages(self, catalog_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute average vectors for each field across entire dataset"""
        global_averages = {}
        
        for field in VECTOR_FIELDS:
            valid_vectors = []
            for _, record in catalog_df.iterrows():
                vec = self.get_vector_from_field(record[field])
                if vec is not None and not np.isnan(vec).any():
                    valid_vectors.append(vec)
            
            if valid_vectors:
                global_averages[field] = np.mean(valid_vectors, axis=0)
                logger.info(f"Computed global average for {field} using {len(valid_vectors)} vectors")
            else:
                global_averages[field] = None
                logger.warning(f"No valid vectors found for field {field} in entire dataset")
                
        return global_averages


    def precompute_neighborhoods(self, catalog_df: pd.DataFrame, 
                           threshold: float = 0.70,
                           collection_name: str = "entity_vectors") -> Dict[str, Dict]:
        """Precompute neighborhoods and field averages"""
        # Ensure global averages are computed
        if not hasattr(self, 'global_averages'):
            self.global_averages = self.compute_global_averages(catalog_df)
        
        neighborhoods = {}
        
        # Statistics tracking
        stats = {
            'limit_hits': 0,
            'total_neighbors': 0,
            'similarity_ranges': {
                '0.95-1.00': 0,
                '0.90-0.95': 0,
                '0.85-0.90': 0,
                '0.80-0.85': 0,
                '0.75-0.80': 0,
                '0.70-0.75': 0
            },
            'records_processed': 0,
            'records_with_neighbors': 0,
            'max_neighbors': 0,
            'records_without_blocking_vector': 0
        }
        
        try:
            # Process records in batches
            batch_size = 100
            for start_idx in range(0, len(catalog_df), batch_size):
                end_idx = min(start_idx + batch_size, len(catalog_df))
                batch_records = catalog_df.iloc[start_idx:end_idx]
                
                for idx, record in batch_records.iterrows():
                    record_id = str(record.loc['id'])  # Use .loc to access Series values
                    stats['records_processed'] += 1
                    
                    # Get blocking vector using .loc
                    blocking_vector = self.get_vector_from_field(record.loc[BLOCKING_FIELD])
                    
                    if blocking_vector is None:
                        logger.warning(f"No blocking vector for record {record_id}")
                        stats['records_without_blocking_vector'] += 1
                        continue
                    
                    # Get neighbors above threshold
                    try:
                        neighbors = self.qdrant_client.search(
                            collection_name=collection_name,
                            query_vector=blocking_vector.tolist(),
                            limit=100,
                            score_threshold=threshold
                        )
                        
                        num_neighbors = len(neighbors)
                        stats['total_neighbors'] += num_neighbors
                        
                        if num_neighbors > 0:
                            stats['records_with_neighbors'] += 1
                            stats['max_neighbors'] = max(stats['max_neighbors'], num_neighbors)
                        
                        # Log if hitting limit
                        if num_neighbors == 100:
                            stats['limit_hits'] += 1
                            logger.warning(f"Record {record_id} hit the 100-neighbor limit, may be missing matches above threshold {threshold}")
                        
                        # Track similarity score distribution
                        for neighbor in neighbors:
                            sim_score = neighbor.score
                            if sim_score >= 0.95: stats['similarity_ranges']['0.95-1.00'] += 1
                            elif sim_score >= 0.90: stats['similarity_ranges']['0.90-0.95'] += 1
                            elif sim_score >= 0.85: stats['similarity_ranges']['0.85-0.90'] += 1
                            elif sim_score >= 0.80: stats['similarity_ranges']['0.80-0.85'] += 1
                            elif sim_score >= 0.75: stats['similarity_ranges']['0.75-0.80'] += 1
                            else: stats['similarity_ranges']['0.70-0.75'] += 1
                        
                        logger.debug(f"Record {record_id} found {num_neighbors} neighbors with similarity >= {threshold}")
                        if num_neighbors > 0:
                            logger.debug(f"Similarity range: {min(n.score for n in neighbors):.3f} - {max(n.score for n in neighbors):.3f}")
                        
                        # Get neighbor IDs and their records
                        neighbor_ids = [self.id_mapping[p.id] for p in neighbors if p.id in self.id_mapping]
                        neighbor_records = catalog_df[catalog_df['id'].isin(neighbor_ids)]
                        
                        # Calculate field averages for the neighborhood
                        field_averages = {}
                        for field in VECTOR_FIELDS:
                            valid_vectors = []
                            for _, nbr in neighbor_records.iterrows():
                                if nbr.loc['id'] != record_id:  # Exclude self
                                    vec = self.get_vector_from_field(nbr.loc[field])
                                    if vec is not None and not np.isnan(vec).any():
                                        valid_vectors.append(vec)
                            
                            if valid_vectors:
                                field_averages[field] = np.mean(valid_vectors, axis=0)
                            else:
                                field_averages[field] = None
                        
                        # Store neighborhood info
                        neighborhoods[record_id] = {
                            'neighbor_ids': neighbor_ids,
                            'field_averages': field_averages,
                            'neighbor_similarities': [n.score for n in neighbors]
                        }
                        
                    except Exception as e:
                        logger.error(f"Error computing neighborhood for {record_id}: {str(e)}")
                        continue
                
                logger.info(f"Processed {end_idx}/{len(catalog_df)} records")
            
            # Log final statistics
            logger.info("\nNeighborhood Computation Statistics:")
            logger.info(f"Total records processed: {stats['records_processed']}")
            logger.info(f"Records with valid blocking vectors: {stats['records_processed'] - stats['records_without_blocking_vector']}")
            logger.info(f"Records that found neighbors: {stats['records_with_neighbors']}")
            logger.info(f"Records hitting 100-neighbor limit: {stats['limit_hits']}")
            logger.info(f"Maximum neighbors for any record: {stats['max_neighbors']}")
            logger.info(f"Average neighbors per record: {stats['total_neighbors']/stats['records_processed']:.2f}")
            
            logger.info("\nSimilarity Score Distribution:")
            for range_name, count in stats['similarity_ranges'].items():
                logger.info(f"{range_name}: {count} neighbors ({count/stats['total_neighbors']*100:.1f}% if total_neighbors > 0 else 0)%)")
            
            logger.info(f"\nPrecomputed neighborhoods for {len(neighborhoods)} records")
            return neighborhoods
            
        except Exception as e:
            logger.error(f"Error in neighborhood precomputation: {str(e)}")
            raise

    def save_imputed_dataset(self, catalog_df: pd.DataFrame, output_file: str = "imputed_dataset.csv") -> None:
        output_path = os.path.join(BASE_PATH, output_file)
        imputed_df = catalog_df.copy()
        
        try:
            records_with_imputations = set()  # Track unique records that had imputations
            fields_imputed = defaultdict(int)
            
            # For each record that needed imputation
            for record_id, neighborhood in self.neighborhoods.items():
                if 'field_averages' in neighborhood:
                    # Get the index of this record in the DataFrame
                    record_idx = imputed_df[imputed_df['id'] == record_id].index[0]
                    
                    # For each field that might need imputation
                    for field in VECTOR_FIELDS:
                        original_vec = self.get_vector_from_field(imputed_df.loc[record_idx, field])
                        
                        # If field needs imputation
                        if original_vec is None:
                            # Get imputed value (either from neighborhood or global average)
                            imputed_vec = neighborhood['field_averages'].get(field)
                            if imputed_vec is None and hasattr(self, 'global_averages'):
                                imputed_vec = self.global_averages.get(field)
                                
                            if imputed_vec is not None:
                                # Convert numpy array to list and format as string
                                vec_list = imputed_vec.tolist()
                                imputed_df.loc[record_idx, field] = f"[{','.join(str(x) for x in vec_list)}]"
                                fields_imputed[field] += 1
                                records_with_imputations.add(record_id)
            
            # Save to CSV
            imputed_df.to_csv(output_path, index=False)
            
            # Log statistics
            logger.info(f"\nImputation Statistics:")
            logger.info(f"Records that needed imputation: {len(records_with_imputations)}")
            logger.info("\nImputations by field:")
            for field, count in fields_imputed.items():
                logger.info(f"{field}: {count} imputations")
            logger.info(f"\nSaved imputed dataset to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving imputed dataset: {str(e)}")
            raise
    
    def create_graphml(self, catalog_df: pd.DataFrame, predictions: List[Tuple[str, str, float]], 
                  output_file: str = "matches_graph.graphml") -> None:
        """
        Create GraphML representation of predicted matches.
        
        Args:
            catalog_df: DataFrame containing catalog records
            predictions: List of (id1, id2, confidence) tuples
            output_file: Path to output GraphML file
        """
        try:
            import networkx as nx
            
            # Create empty graph
            G = nx.Graph()
            
            # Create lookup for names
            name_lookup = {}
            for _, row in catalog_df.iterrows():
                record_id = str(row['id'])
                name_val = row['name']
                if isinstance(name_val, str) and name_val != "NaN":
                    name_lookup[record_id] = name_val
                else:
                    name_lookup[record_id] = "Unknown Person"
            
            # Add nodes and edges
            for id1, id2, confidence in predictions:
                # Only add edges for predicted matches
                if confidence > 0.5:
                    # Create label for node 1
                    if id1 not in G:
                        label1 = f"{name_lookup[id1]} | {id1}"
                        G.add_node(id1, label=label1, name=name_lookup[id1])
                    
                    # Create label for node 2
                    if id2 not in G:
                        label2 = f"{name_lookup[id2]} | {id2}"
                        G.add_node(id2, label=label2, name=name_lookup[id2])
                    
                    # Add edge with confidence score as weight
                    G.add_edge(id1, id2, weight=confidence)
            
            # Save graph to GraphML file
            output_path = os.path.join(BASE_PATH, output_file)
            nx.write_graphml(G, output_path)
            
            # Log statistics
            logger.info("\nGraphML Export Statistics:")
            logger.info(f"Number of nodes: {G.number_of_nodes()}")
            logger.info(f"Number of edges: {G.number_of_edges()}")
            
            # Calculate and log component statistics
            components = list(nx.connected_components(G))
            logger.info(f"Number of connected components: {len(components)}")
            
            # Log component sizes
            component_sizes = [len(comp) for comp in components]
            if component_sizes:
                logger.info(f"Largest component size: {max(component_sizes)}")
                logger.info(f"Average component size: {sum(component_sizes)/len(component_sizes):.2f}")
                
                # Log size distribution of top 10 largest components
                top_sizes = sorted(component_sizes, reverse=True)[:10]
                logger.info("Top 10 component sizes: " + ", ".join(str(s) for s in top_sizes))
            
            # Log high-degree nodes
            degrees = dict(G.degree())
            high_degree_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("\nTop 10 highest-degree nodes:")
            for node, degree in high_degree_nodes:
                logger.info(f"Node: {G.nodes[node]['label']}, Degree: {degree}")
            
            logger.info(f"\nSaved graph to: {output_path}")
            
        except ImportError:
            logger.error("NetworkX library required for GraphML export")
        except Exception as e:
            logger.error(f"Error creating GraphML file: {str(e)}")

def main(use_imputed_dataset: bool = False):
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories for visualizations
    plot_dir = os.path.join(BASE_PATH, f"visualizations_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Pass plot_dir to relevant methods
    resolver = EntityResolver(plot_dir=plot_dir)
    
    
    
    # Load data
    if use_imputed_dataset:
        catalog_df = pd.read_csv(os.path.join(BASE_PATH, "imputed_dataset.csv"))
        logger.info("Loaded pre-imputed dataset")
    else:
        catalog_df = pd.read_csv(os.path.join(BASE_PATH, "vector_dataset.csv"))
        logger.info("Loaded original dataset")
        
    matches_df = pd.read_csv(os.path.join(BASE_PATH, "benchmark_data_matches_expanded.csv"))
    logger.info(f"Loaded {len(catalog_df)} catalog records and {len(matches_df)} ground truth pairs")
    
    # Analyze blocking keys first
    logger.info("Analyzing blocking key similarities...")
    blocking_analysis = resolver.analyze_blocking_keys(catalog_df, matches_df)
    logger.info("\nBlocking Key Analysis Results:")
    logger.info(json.dumps(blocking_analysis['stats'], indent=2))
    
    # Setup vector index
    resolver.setup_vector_index()
    
    # Index vectors (needed for both imputation and blocking)
    resolver.index_vectors(catalog_df)
    
    if not use_imputed_dataset:
        # Precompute neighborhoods and field averages
        logger.info("Precomputing neighborhoods and field averages...")
        resolver.neighborhoods = resolver.precompute_neighborhoods(catalog_df)
        
        # Save imputed dataset for future use
        logger.info("Saving imputed dataset...")
        resolver.save_imputed_dataset(catalog_df)
        
        # Process catalog data with imputation
        logger.info("Processing catalog data with imputation...")
        catalog_df = resolver.preprocess_catalog(catalog_df)
    
    # Generate all valid comparison pairs using ANN blocking
    logger.info("Generating comparison pairs using ANN blocking...")
    comparison_pairs = resolver.generate_comparison_pairs(catalog_df)
    
    # Split pairs into train/test sets
    comparison_pairs_df = pd.DataFrame(comparison_pairs, columns=['left', 'right'])
    
    # Merge with ground truth to get match labels
    comparison_pairs_df['pair_key'] = comparison_pairs_df.apply(
        lambda x: tuple(sorted([x['left'], x['right']])), axis=1)
    matches_df['pair_key'] = matches_df.apply(
        lambda x: tuple(sorted([x['left'], x['right']])), axis=1)
    
    labeled_pairs = comparison_pairs_df.merge(
        matches_df[['pair_key', 'match']], 
        on='pair_key', 
        how='left'
    )
    labeled_pairs['match'] = labeled_pairs['match'].fillna(False)
    
    # Split into train/test
    train_pairs, test_pairs = train_test_split(
        labeled_pairs,
        test_size=0.3,
        random_state=42,
        stratify=labeled_pairs['match']
    )
    
    # Generate training data
    logger.info("Generating training pairs...")
    X_train, y_train = resolver.generate_training_pairs(
        catalog_df, 
        train_pairs[['left', 'right', 'match']]
    )
    
    # Train classifier
    logger.info("Training classifier...")
    resolver.train_classifier(X_train, y_train)
    
    # Generate test pairs from test split
    test_pair_list = list(zip(test_pairs['left'], test_pairs['right']))
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = resolver.predict_matches(catalog_df, test_pair_list)
    
    # Save predictions with detailed analysis
    pred_df = resolver.save_predictions(predictions, catalog_df, matches_df, "predicted_matches.csv")

    # After saving predictions
    logger.info("Creating GraphML representation...")
    resolver.create_graphml(catalog_df, predictions)
    
    # Evaluate
    results = resolver.evaluate(predictions, test_pairs[['left', 'right', 'match']])
    logger.info(f"Results: {json.dumps(results, indent=2)}")

    # After all processing
    #logger.info(f"\nVisualizations saved to: {plot_dir}")
    
    return resolver, pred_df, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-imputed', action='store_true', 
                       help='Use pre-imputed dataset instead of performing imputation')
    args = parser.parse_args()
    main(use_imputed_dataset=args.use_imputed)