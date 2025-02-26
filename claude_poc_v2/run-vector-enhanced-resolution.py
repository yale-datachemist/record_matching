#!/usr/bin/env python3
"""
Run Script for Vector-Enhanced Entity Resolution

This script runs the vector-enhanced entity resolution process on the provided dataset.
It demonstrates how to use the VectorEnhancedEntityResolver class to resolve entities
with improved disambiguation using vector embeddings.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import json
import ast
import re
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Set, Optional

# Import the vector-enhanced entity resolver
from vector_enhanced_entity_resolution_v2 import VectorEnhancedEntityResolver, load_vector_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("vector_resolution.log"),
        logging.StreamHandler()
    ]
)

def create_dummy_vectors(entity_df: pd.DataFrame, vector_dim: int = 3072) -> Dict[str, np.ndarray]:
    """
    Create dummy vector embeddings for testing when real vectors are not available.
    
    Args:
        entity_df: DataFrame with entity records
        vector_dim: Dimension of vectors to create
        
    Returns:
        Dictionary mapping entity ID to vector embedding
    """
    logging.info(f"Creating dummy vectors of dimension {vector_dim} for {len(entity_df)} entities")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    dummy_vectors = {}
    for _, row in entity_df.iterrows():
        # For entities with the same name, create similar vectors
        # This simulates the behavior of real embeddings where similar entities have similar vectors
        name = row['person'] if 'person' in row else ''
        
        # Create a seed from the name
        name_seed = sum(ord(c) for c in name)
        np.random.seed(name_seed)
        
        # Create base vector
        base_vector = np.random.random(vector_dim).astype(np.float32)
        
        # Add random noise
        np.random.seed(int(time.time() * 1000) % 1000000)
        noise = np.random.random(vector_dim).astype(np.float32) * 0.2
        
        # Create final vector
        dummy_vectors[row['id']] = base_vector + noise
    
    return dummy_vectors

def run_resolution(entity_file: str, 
                   ground_truth_file: str, 
                   vector_file: str = None,
                   output_dir: str = 'results',
                verbose: bool = True):
    """
    Run the vector-enhanced entity resolution process.
    
    Args:
        entity_file: Path to entity data CSV
        ground_truth_file: Path to ground truth matches CSV
        vector_file: Path to vector data file (optional)
        output_dir: Directory to save results
        verbose: Enable verbose logging
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(output_dir, "vector_resolution.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=== Starting Vector-Enhanced Entity Resolution ===")
    logging.info(f"Entity file: {entity_file}")
    logging.info(f"Ground truth file: {ground_truth_file}")
    logging.info(f"Vector file: {vector_file}")
    logging.info(f"Output directory: {output_dir}")
    
    # Load data
    logging.info(f"Loading entity data from {entity_file}")
    entities_df = pd.read_csv(entity_file)
    
    logging.info(f"Loading ground truth from {ground_truth_file}")
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    # Load vector data with improved error handling
    vector_df = None
    if vector_file and os.path.exists(vector_file):
        logging.info(f"Loading vector data from {vector_file}")
        try:
            # Check file extension to determine loading method
            if vector_file.endswith('.csv'):
                vector_df = load_vector_data(entities_df, vector_file)
            elif vector_file.endswith('.pkl'):
                with open(vector_file, 'rb') as f:
                    vector_data = pickle.load(f)
                    # Convert dictionary to DataFrame if needed
                    if isinstance(vector_data, dict):
                        vector_df = pd.DataFrame.from_dict(vector_data, orient='index')
                    else:
                        vector_df = vector_data
            else:
                logging.warning(f"Unsupported vector file format: {vector_file}")
                vector_df = None
                
            # Check if we successfully loaded vector data
            if vector_df is None or len(vector_df) == 0:
                logging.warning("Failed to load vectors. Creating dummy vectors instead.")
                dummy_vectors = create_dummy_vectors(entities_df)
                vector_df = pd.DataFrame.from_dict(dummy_vectors, orient='index')
                
                # Save for future reference
                dummy_path = os.path.join(output_dir, 'dummy_vectors.pkl')
                with open(dummy_path, 'wb') as f:
                    pickle.dump(dummy_vectors, f)
                logging.info(f"Saved dummy vectors to {dummy_path}")

            # Add this line to inspect the vector data
            logging.info(f"Vector data sample: {vector_df.iloc[0] if len(vector_df) > 0 else 'Empty'}")
            # Also check the column names
            logging.info(f"Vector data columns: {vector_df.columns.tolist() if hasattr(vector_df, 'columns') else 'No columns'}")
            # Log the shape
            logging.info(f"Vector data shape: {vector_df.shape if hasattr(vector_df, 'shape') else 'No shape'}")

            # Add this after loading vector_df
            if vector_df is not None and len(vector_df) > 0:
                # Check if there's an ID mismatch issue
                logging.info(f"Vector DataFrame index example: {vector_df.index[0]}")
                logging.info(f"Entity DataFrame 'id' example: {entities_df['id'].iloc[0]}")
                
                # Look for an ID column or create one from the index
                if 'id' not in vector_df.columns:
                    logging.info("No 'id' column found in vector data, using index as ID")
                    # Create an ID column from the index, or extract it from a compound index
                    try:
                        # Try extracting the numeric part if it's a compound ID like "53144#Agent700-22"
                        # Use the full index string as the ID
                        vector_df['id'] = vector_df.index.astype(str)
                        logging.info(f"Using full index as 'id': {vector_df['id'].iloc[0]}")
                    except:
                        # Just use the index as is
                        vector_df['id'] = vector_df.index
                        logging.info(f"Using full index as 'id': {vector_df['id'].iloc[0]}")
                    
                    # Check if any IDs match between datasets
                    entity_ids = set(entities_df['id'].astype(str))
                    vector_ids = set(vector_df['id'].astype(str))
                    common_ids = entity_ids.intersection(vector_ids)
                    
                    logging.info(f"Entity dataset has {len(entity_ids)} unique IDs")
                    logging.info(f"Vector dataset has {len(vector_ids)} unique IDs")
                    logging.info(f"Found {len(common_ids)} common IDs between datasets")
                    
                    if len(common_ids) == 0:
                        logging.error("No matching IDs between datasets! Vector similarity will be ineffective.")
                        # Try to find an alternative matching strategy
                        logging.info("Attempting to match records using 'recordid' if available")
                        if 'recordid' in vector_df.columns and 'recordId' in entities_df.columns:
                            vector_df['id'] = vector_df['recordid'].astype(str)
                            common_ids = set(entities_df['recordId'].astype(str)).intersection(set(vector_df['id']))
                            logging.info(f"Found {len(common_ids)} matches using recordId")

        except Exception as e:
            logging.error(f"Error loading vector data: {e}")
            logging.warning("Creating dummy vectors instead.")
            dummy_vectors = create_dummy_vectors(entities_df)
            vector_df = pd.DataFrame.from_dict(dummy_vectors, orient='index')
    else:
        logging.warning("Vector file not provided or not found. Creating dummy vectors.")
        # Create dummy vectors
        dummy_vectors = create_dummy_vectors(entities_df)
        
        # Convert to DataFrame
        vector_df = pd.DataFrame.from_dict(dummy_vectors, orient='index')
        
        # Save to file for future use
        temp_vector_file = os.path.join(output_dir, 'dummy_vectors.pkl')
        with open(temp_vector_file, 'wb') as f:
            pickle.dump(dummy_vectors, f)
        logging.info(f"Saved dummy vectors to {temp_vector_file}")
    
    # Log data statistics
    logging.info(f"Loaded {len(entities_df)} entities, {len(ground_truth_df)} ground truth pairs, "
                f"and {len(vector_df)} vectors")
    
    # Initialize resolver with improved configuration
    #vector_dim = vector_df.shape[1] if hasattr(vector_df, 'shape') else 3072
    resolver = VectorEnhancedEntityResolver(
        vector_dim=3072,
        min_similarity=0.65  # Lower threshold for better recall
    )
    
    # Train the model if ground truth data is available
    if not ground_truth_df.empty:
        start_time = time.time()
        logging.info("Training model with ground truth data")
        try:
            training_metrics = resolver.train_with_ground_truth(
                entities_df, ground_truth_df, vector_df)
            train_time = time.time() - start_time
            
            logging.info(f"Training completed in {train_time:.2f} seconds")
            
            # Save model
            model_path = os.path.join(output_dir, 'vector_resolver_model.json')
            resolver.save_model(model_path)
            logging.info(f"Saved trained model to {model_path}")
            
            # Save training metrics
            metrics_file = os.path.join(output_dir, 'training_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(training_metrics, f, indent=2)
            logging.info(f"Saved training metrics to {metrics_file}")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            logging.warning("Proceeding with untrained resolver.")
    else:
        logging.warning("No ground truth data available. Skipping training step.")
    
    # Resolve entities
    start_time = time.time()
    logging.info("Resolving entities")
    try:
        resolved_df = resolver.resolve_entities(entities_df, vector_df)
        resolve_time = time.time() - start_time
        
        logging.info(f"Entity resolution completed in {resolve_time:.2f} seconds")
        
        # Save resolved entities
        resolved_file = os.path.join(output_dir, 'resolved_entities.csv')
        resolved_df.to_csv(resolved_file, index=False)
        logging.info(f"Saved resolved entities to {resolved_file}")
    except Exception as e:
        logging.error(f"Error during entity resolution: {e}")
        return None, None, None
    
    # Evaluate results
    evaluation_metrics = None
    if not ground_truth_df.empty:
        logging.info("Evaluating results")
        try:
            evaluation_metrics = resolver.evaluate_results(resolved_df, ground_truth_df)
            
            # Save evaluation metrics
            metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
            with open(metrics_file, 'w') as f:
                # Remove large data structures
                eval_metrics_copy = {k: v for k, v in evaluation_metrics.items() 
                                if k != 'pair_details'}
                
                # Convert NumPy types to Python types for JSON serialization
                eval_metrics_copy = {
                    k: int(v) if isinstance(v, np.integer) else 
                    float(v) if isinstance(v, np.floating) else v 
                    for k, v in eval_metrics_copy.items()
                }
                
                json.dump(eval_metrics_copy, f, indent=2)
            logging.info(f"Saved evaluation metrics to {metrics_file}")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
    else:
        logging.warning("No ground truth data available. Skipping evaluation step.")
    
    # Export canonical entities
    try:
        canonical_file = os.path.join(output_dir, 'canonical_entities.csv')
        canonical_df = resolver.export_canonical_entities(canonical_file)
        logging.info(f"Exported {len(canonical_df)} canonical entities to {canonical_file}")
    except Exception as e:
        logging.error(f"Error exporting canonical entities: {e}")
    
    # Create visualizations
    try:
        # Create cluster size distribution
        plt.figure(figsize=(10, 6))
        cluster_sizes = resolved_df.groupby('cluster_id').size()
        sns.histplot(cluster_sizes, bins=20, kde=True)
        plt.title('Cluster Size Distribution')
        plt.xlabel('Records per Cluster')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_size_distribution.png'))
        
        # Create bar chart of top domains
        if 'primary_domain' in resolved_df.columns:
            plt.figure(figsize=(12, 6))
            domain_counts = Counter()
            for domains in resolved_df['primary_domain'].dropna():
                for domain in domains.split(';'):
                    domain_counts[domain.strip()] += 1
            
            # Plot top 10 domains
            top_domains = dict(domain_counts.most_common(10))
            plt.bar(top_domains.keys(), top_domains.values())
            plt.title('Top 10 Domains in Dataset')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_domains.png'))
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
    
    # Check for disambiguation cases
    try:
        name_clusters = defaultdict(set)
        for _, row in resolved_df.iterrows():
            name = row['normalized_name']
            cluster = row['cluster_id']
            name_clusters[name].add(cluster)
        
        # Find names mapped to multiple clusters (successful disambiguation)
        disambiguated = {name: sorted(list(clusters)) for name, clusters in name_clusters.items() 
                    if len(clusters) > 1}
        
        if disambiguated:
            logging.info("\nSuccessfully disambiguated entities with same name:")
            
            # Save disambiguation details to a file
            disambig_file = os.path.join(output_dir, 'disambiguation_details.txt')
            with open(disambig_file, 'w') as f:
                f.write(f"Successfully disambiguated {len(disambiguated)} names into multiple entities\n\n")
                
                for name, clusters in disambiguated.items():
                    f.write(f"{name}: split into {len(clusters)} distinct entities\n")
                    
                    # Get example records for each cluster
                    for cluster_id in clusters:
                        cluster_records = resolved_df[resolved_df['cluster_id'] == cluster_id]
                        sample_record = cluster_records.iloc[0]
                        
                        f.write(f"  Cluster {cluster_id}: {sample_record['person']}\n")
                        
                        # Add domain information if available
                        if pd.notna(sample_record.get('primary_domain')):
                            f.write(f"    Domains: {sample_record['primary_domain']}\n")
                        elif pd.notna(sample_record.get('subjects')):
                            f.write(f"    Subjects: {sample_record['subjects']}\n")
                        
                        # Add a sample title
                        if pd.notna(sample_record['title']):
                            f.write(f"    Example title: {sample_record['title']}\n")
                        
                        # Add record count
                        f.write(f"    Records in cluster: {len(cluster_records)}\n")
                    
                    f.write("\n")
                
            logging.info(f"Saved disambiguation details to {disambig_file}")
            
            # Check for our Schubert case specifically
            if "schubert franz" in disambiguated:
                logging.info("  - Successfully disambiguated Franz Schubert entities!")
                
                # Show details of Schubert entities
                for cluster_id in disambiguated["schubert franz"]:
                    cluster_records = resolved_df[resolved_df['cluster_id'] == cluster_id]
                    sample_record = cluster_records.iloc[0]
                    
                    logging.info(f"    Cluster {cluster_id}: {sample_record['person']}")
                    if pd.notna(sample_record.get('subjects')):
                        logging.info(f"      Subjects: {sample_record['subjects']}")
    except Exception as e:
        logging.error(f"Error analyzing disambiguation cases: {e}")
    
    # Print summary
    if evaluation_metrics:
        print("\nVector-Enhanced Entity Resolution Summary:")
        print(f"Resolved {len(resolved_df)} records into {evaluation_metrics['num_clusters']} entities")
        
        if 'precision' in evaluation_metrics:
            print(f"Precision: {evaluation_metrics['precision']:.3f}")
            print(f"Recall: {evaluation_metrics['recall']:.3f}")
            print(f"F1 Score: {evaluation_metrics['f1']:.3f}")
        
        if disambiguated:
            print(f"\nSuccessfully disambiguated {len(disambiguated)} names into multiple entities")
        
        print(f"\nResults saved to '{output_dir}' directory")
    else:
        print("\nVector-Enhanced Entity Resolution Summary:")
        print(f"Resolved {len(resolved_df)} records into {len(set(resolved_df['cluster_id']))} entities")
        print(f"Results saved to '{output_dir}' directory")
    
    return resolver, resolved_df, evaluation_metrics

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run vector-enhanced entity resolution')
    parser.add_argument('--entity-file', dest='entity_file', default='entity_dataset.csv',
                       help='Path to entity data CSV file')
    parser.add_argument('--ground-truth', dest='ground_truth_file', default='data_matches.csv',
                       help='Path to ground truth matches CSV file')
    parser.add_argument('--vector-file', dest='vector_file', default='entity_dataset_vectorized.csv',
                       help='Path to vector data file (optional)')
    parser.add_argument('--output-dir', dest='output_dir', default='vector_results',
                       help='Directory to save results')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run resolution
    run_resolution(
        args.entity_file,
        args.ground_truth_file,
        args.vector_file,
        args.output_dir,
        args.verbose
    )