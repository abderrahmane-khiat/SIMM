"""
Complete Implementation of:
"A novel framework for service set recommendation in mashup creation"
Gao, W., & Wu, J. (2017). ICWS, pp. 65-72.

Service Set Recommendation (SSR) Framework
Components:
1. Service Similarity Network Construction
2. Mashup Similarity Computation  
3. Candidate Service Set Generation
4. Service Set Scoring and Ranking
5. Collaborative Filtering Enhancement
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import normalize
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
import warnings
from collections import defaultdict, Counter
import math
import random
from dataclasses import dataclass
from itertools import combinations, permutations
import heapq

warnings.filterwarnings('ignore')

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WebService:
    """Web Service/API representation"""
    service_id: str
    name: str
    description: str
    tags: List[str] = None
    category: str = ""
    provider: str = ""
    popularity: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class Mashup:
    """Mashup composition representation"""
    mashup_id: str
    name: str
    description: str
    services: List[str]  # List of service IDs
    tags: List[str] = None
    category: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ServiceSet:
    """Recommended set of services for mashup creation"""
    services: List[str]
    score: float = 0.0
    coverage_score: float = 0.0
    diversity_score: float = 0.0
    compatibility_score: float = 0.0

# ============================================================================
# Service Similarity Network Construction
# ============================================================================

class ServiceSimilarityNetwork:
    """
    Constructs and manages service similarity network
    Section III-B: Service Similarity Network Construction
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, 
                 similarity_threshold: float = 0.3):
        """
        Args:
            alpha: Weight for content-based similarity
            beta: Weight for collaborative similarity
            similarity_threshold: Minimum similarity to create edge
        """
        self.alpha = alpha
        self.beta = beta
        self.similarity_threshold = similarity_threshold
        self.graph = nx.Graph()
        self.service_features = {}
        self.service_vectors = {}
        self.similarity_matrix = None
        
    def extract_content_features(self, services: List[WebService]) -> Dict[str, np.ndarray]:
        """Extract content features from service descriptions and tags"""
        service_features = {}
        
        for service in services:
            # Combine name, description, and tags
            content_text = f"{service.name} {service.description} {' '.join(service.tags)}"
            service_features[service.service_id] = content_text
        
        return service_features
    
    def compute_content_similarity(self, services: List[WebService]) -> csr_matrix:
        """Compute content-based similarity using TF-IDF and cosine similarity"""
        service_ids = [s.service_id for s in services]
        content_features = self.extract_content_features(services)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        texts = [content_features[sid] for sid in service_ids]
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Compute cosine similarity
        content_sim = cosine_similarity(tfidf_matrix)
        
        # Store feature vectors
        for i, sid in enumerate(service_ids):
            self.service_vectors[sid] = tfidf_matrix[i].toarray().flatten()
        
        return csr_matrix(content_sim)
    
    def compute_collaborative_similarity(self, services: List[WebService], 
                                        mashups: List[Mashup]) -> csr_matrix:
        """
        Compute collaborative similarity based on co-occurrence in mashups
        (User-based collaborative filtering for services)
        """
        n_services = len(services)
        service_id_to_idx = {s.service_id: i for i, s in enumerate(services)}
        
        # Create service-mashup matrix (services as rows, mashups as columns)
        service_mashup_matrix = lil_matrix((n_services, len(mashups)))
        
        for mashup_idx, mashup in enumerate(mashups):
            for service_id in mashup.services:
                if service_id in service_id_to_idx:
                    service_idx = service_id_to_idx[service_id]
                    service_mashup_matrix[service_idx, mashup_idx] = 1
        
        service_mashup_matrix = service_mashup_matrix.tocsr()
        
        # Compute Jaccard similarity for collaborative filtering
        collaborative_sim = lil_matrix((n_services, n_services))
        
        for i in range(n_services):
            # Get mashups containing service i
            mashups_i = set(service_mashup_matrix[i].nonzero()[1])
            
            for j in range(i+1, n_services):
                # Get mashups containing service j
                mashups_j = set(service_mashup_matrix[j].nonzero()[1])
                
                if mashups_i and mashups_j:
                    intersection = len(mashups_i.intersection(mashups_j))
                    union = len(mashups_i.union(mashups_j))
                    
                    if union > 0:
                        jaccard_sim = intersection / union
                        collaborative_sim[i, j] = jaccard_sim
                        collaborative_sim[j, i] = jaccard_sim
        
        return collaborative_sim.tocsr()
    
    def construct_network(self, services: List[WebService], mashups: List[Mashup]):
        """
        Construct service similarity network
        Formula: Sim(s_i, s_j) = α * Sim_content(s_i, s_j) + β * Sim_collab(s_i, s_j)
        """
        print("Constructing service similarity network...")
        
        service_ids = [s.service_id for s in services]
        n_services = len(services)
        
        # Compute content similarity
        print("  Computing content similarity...")
        content_sim = self.compute_content_similarity(services)
        
        # Compute collaborative similarity  
        print("  Computing collaborative similarity...")
        collab_sim = self.compute_collaborative_similarity(services, mashups)
        
        # Combine similarities
        print("  Combining similarities...")
        self.similarity_matrix = lil_matrix((n_services, n_services))
        
        service_id_to_idx = {s.service_id: i for i, s in enumerate(services)}
        
        for i in range(n_services):
            for j in range(i+1, n_services):
                # Get similarity scores
                content_score = content_sim[i, j] if i < content_sim.shape[0] and j < content_sim.shape[1] else 0
                collab_score = collab_sim[i, j] if i < collab_sim.shape[0] and j < collab_sim.shape[1] else 0
                
                # Combined similarity
                combined_score = self.alpha * content_score + self.beta * collab_score
                
                if combined_score >= self.similarity_threshold:
                    self.similarity_matrix[i, j] = combined_score
                    self.similarity_matrix[j, i] = combined_score
        
        self.similarity_matrix = self.similarity_matrix.tocsr()
        
        # Build network graph
        print("  Building network graph...")
        self.graph.add_nodes_from(service_ids)
        
        edges_added = 0
        for i in range(n_services):
            for j in range(i+1, n_services):
                sim_score = self.similarity_matrix[i, j]
                if sim_score > 0:
                    service_i = service_ids[i]
                    service_j = service_ids[j]
                    self.graph.add_edge(service_i, service_j, weight=sim_score)
                    edges_added += 1
        
        print(f"  Network constructed: {n_services} nodes, {edges_added} edges")
        
        return self.graph
    
    def get_similar_services(self, service_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k similar services for a given service"""
        if service_id not in self.service_vectors:
            return []
        
        similarities = []
        target_vector = self.service_vectors[service_id]
        
        for other_id, other_vector in self.service_vectors.items():
            if other_id != service_id:
                # Compute cosine similarity
                sim = 1 - cosine(target_vector, other_vector)
                similarities.append((other_id, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]

# ============================================================================
# Mashup Similarity Computation
# ============================================================================

class MashupSimilarityCalculator:
    """
    Computes similarity between mashups and between mashups and services
    Section III-C: Mashup Similarity Computation
    """
    
    def __init__(self, use_lda: bool = True, n_topics: int = 20):
        self.use_lda = use_lda
        self.n_topics = n_topics
        self.lda_model = None
        self.vectorizer = None
        self.mashup_topic_features = {}
        
    def compute_mashup_mashup_similarity(self, mashups: List[Mashup]) -> csr_matrix:
        """
        Compute similarity between mashups based on:
        1. Content similarity (descriptions, tags)
        2. Service composition similarity
        """
        n_mashups = len(mashups)
        
        # Extract mashup content features
        mashup_texts = []
        for mashup in mashups:
            text = f"{mashup.name} {mashup.description} {' '.join(mashup.tags)}"
            mashup_texts.append(text)
        
        # Text-based similarity using TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        text_vectors = vectorizer.fit_transform(mashup_texts)
        text_similarity = cosine_similarity(text_vectors)
        
        # Service composition similarity (Jaccard)
        service_similarity = lil_matrix((n_mashups, n_mashups))
        
        for i in range(n_mashups):
            services_i = set(mashups[i].services)
            for j in range(i+1, n_mashups):
                services_j = set(mashups[j].services)
                
                if services_i and services_j:
                    intersection = len(services_i.intersection(services_j))
                    union = len(services_i.union(services_j))
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    service_similarity[i, j] = jaccard_sim
                    service_similarity[j, i] = jaccard_sim
        
        service_similarity = service_similarity.tocsr()
        
        # Combine similarities (equal weight in paper)
        combined_similarity = (text_similarity + service_similarity.toarray()) / 2
        
        return csr_matrix(combined_similarity)
    
    def compute_mashup_service_similarity(self, mashups: List[Mashup], 
                                         services: List[WebService],
                                         service_network: ServiceSimilarityNetwork) -> Dict[str, Dict[str, float]]:
        """
        Compute similarity between mashups and services
        Based on mashup requirements and service capabilities
        """
        mashup_service_sim = {}
        
        # Extract service features
        service_features = {}
        for service in services:
            text = f"{service.name} {service.description} {' '.join(service.tags)}"
            service_features[service.service_id] = text
        
        # Extract mashup features
        mashup_features = {}
        for mashup in mashups:
            text = f"{mashup.name} {mashup.description} {' '.join(mashup.tags)}"
            mashup_features[mashup.mashup_id] = text
        
        # Combine all texts for TF-IDF
        all_texts = list(service_features.values()) + list(mashup_features.values())
        all_ids = list(service_features.keys()) + list(mashup_features.keys())
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        all_vectors = vectorizer.fit_transform(all_texts)
        
        # Split back into service and mashup vectors
        n_services = len(services)
        service_vectors = all_vectors[:n_services]
        mashup_vectors = all_vectors[n_services:]
        
        # Compute cosine similarity between mashups and services
        similarity_matrix = cosine_similarity(mashup_vectors, service_vectors)
        
        # Create dictionary structure
        for i, mashup_id in enumerate(mashup_features.keys()):
            mashup_service_sim[mashup_id] = {}
            for j, service_id in enumerate(service_features.keys()):
                mashup_service_sim[mashup_id][service_id] = similarity_matrix[i, j]
        
        return mashup_service_sim
    
    def extract_mashup_topics(self, mashups: List[Mashup]):
        """Extract latent topics from mashup descriptions using LDA"""
        mashup_texts = []
        for mashup in mashups:
            text = f"{mashup.name} {mashup.description} {' '.join(mashup.tags)}"
            mashup_texts.append(text)
        
        # Create document-term matrix
        count_vectorizer = CountVectorizer(max_features=500, stop_words='english')
        doc_term_matrix = count_vectorizer.fit_transform(mashup_texts)
        
        # Apply LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='online'
        )
        
        topic_distributions = self.lda_model.fit_transform(doc_term_matrix)
        
        # Store topic features for each mashup
        for i, mashup in enumerate(mashups):
            self.mashup_topic_features[mashup.mashup_id] = topic_distributions[i]
        
        return topic_distributions

# ============================================================================
# Candidate Service Set Generation
# ============================================================================

class CandidateSetGenerator:
    """
    Generates candidate service sets for mashup creation
    Section III-D: Candidate Service Set Generation
    """
    
    def __init__(self, max_set_size: int = 5, min_set_size: int = 2):
        self.max_set_size = max_set_size
        self.min_set_size = min_set_size
        self.service_coverage = {}
        self.service_frequency = {}
        
    def compute_service_coverage(self, services: List[WebService], mashups: List[Mashup]):
        """Compute how well each service covers different mashup requirements"""
        # Count service frequency in mashups
        service_freq = Counter()
        for mashup in mashups:
            for service_id in mashup.services:
                service_freq[service_id] += 1
        
        self.service_frequency = dict(service_freq)
        
        # Normalize frequencies
        max_freq = max(service_freq.values()) if service_freq else 1
        
        for service_id, freq in service_freq.items():
            self.service_coverage[service_id] = freq / max_freq
        
        return self.service_coverage
    
    def generate_by_similar_mashups(self, target_mashup_id: str, 
                                   mashup_similarity: csr_matrix,
                                   mashup_list: List[Mashup],
                                   top_k: int = 5) -> List[List[str]]:
        """
        Generate candidate sets based on similar mashups
        Strategy 1: Direct composition from similar mashups
        """
        candidate_sets = []
        
        # Find index of target mashup
        mashup_indices = {m.mashup_id: i for i, m in enumerate(mashup_list)}
        
        if target_mashup_id not in mashup_indices:
            return candidate_sets
        
        target_idx = mashup_indices[target_mashup_id]
        
        # Get top-k similar mashups
        similarities = mashup_similarity[target_idx].toarray().flatten()
        similar_indices = np.argsort(similarities)[-top_k-1:-1][::-1]  # Exclude self
        
        for sim_idx in similar_indices:
            similar_mashup = mashup_list[sim_idx]
            candidate_sets.append(similar_mashup.services[:self.max_set_size])
        
        return candidate_sets
    
    def generate_by_service_similarity(self, target_services: List[str],
                                      service_network: ServiceSimilarityNetwork,
                                      top_k_per_service: int = 3) -> List[List[str]]:
        """
        Generate candidate sets by expanding target services with similar services
        Strategy 2: Service similarity-based expansion
        """
        candidate_sets = []
        
        if not target_services:
            return candidate_sets
        
        # For each service in target, get similar services
        similar_services_by_target = {}
        for service_id in target_services:
            similar = service_network.get_similar_services(service_id, top_k_per_service)
            similar_services_by_target[service_id] = [sid for sid, _ in similar]
        
        # Generate combinations
        # Start with original services
        candidate_sets.append(target_services)
        
        # Generate variations by replacing some services with similar ones
        for i, service_id in enumerate(target_services):
            similar_services = similar_services_by_target.get(service_id, [])
            for similar_id in similar_services[:2]:  # Top 2 similar
                new_set = target_services.copy()
                new_set[i] = similar_id
                candidate_sets.append(new_set)
        
        # Generate by combining services from different similar sets
        all_similar_services = []
        for similar_list in similar_services_by_target.values():
            all_similar_services.extend(similar_list[:2])  # Top 2 from each
        
        # Take unique services and create new sets
        unique_similar = list(set(all_similar_services))
        if len(unique_similar) >= self.min_set_size:
            # Create sets of different sizes
            for set_size in range(self.min_set_size, min(self.max_set_size, len(unique_similar)) + 1):
                # Sample some combinations (not all to avoid explosion)
                for _ in range(min(10, math.comb(len(unique_similar), set_size))):
                    candidate = random.sample(unique_similar, set_size)
                    candidate_sets.append(candidate)
        
        return candidate_sets
    
    def generate_by_coverage_diversity(self, services: List[WebService],
                                      target_requirements: str,
                                      top_n: int = 20) -> List[List[str]]:
        """
        Generate candidate sets considering coverage and diversity
        Strategy 3: Coverage-diversity trade-off
        """
        candidate_sets = []
        
        if not services:
            return candidate_sets
        
        # Sort services by coverage score
        sorted_services = sorted(services, 
                                key=lambda s: self.service_coverage.get(s.service_id, 0), 
                                reverse=True)
        
        top_services = [s.service_id for s in sorted_services[:top_n]]
        
        # Generate sets of different sizes
        for set_size in range(self.min_set_size, min(self.max_set_size, len(top_services)) + 1):
            # Generate some combinations
            n_combinations = min(10, math.comb(len(top_services), set_size))
            
            if n_combinations > 0:
                # Use random sampling for combinations
                for _ in range(n_combinations):
                    candidate = random.sample(top_services, set_size)
                    candidate_sets.append(candidate)
        
        return candidate_sets

# ============================================================================
# Service Set Scoring and Ranking
# ============================================================================

class ServiceSetScorer:
    """
    Scores and ranks candidate service sets
    Section III-E: Service Set Scoring and Ranking
    """
    
    def __init__(self, lambda1: float = 0.4, lambda2: float = 0.3, lambda3: float = 0.3):
        """
        Args:
            lambda1: Weight for coverage score
            lambda2: Weight for diversity score  
            lambda3: Weight for compatibility score
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
    def compute_coverage_score(self, service_set: List[str], 
                              target_mashup_id: str,
                              mashup_service_similarity: Dict[str, Dict[str, float]]) -> float:
        """
        Coverage Score: How well the service set covers mashup requirements
        Formula: Coverage(S) = Σ_{s∈S} Sim(m, s) / |S|
        """
        if not service_set:
            return 0.0
        
        total_similarity = 0.0
        
        if target_mashup_id in mashup_service_similarity:
            mashup_sims = mashup_service_similarity[target_mashup_id]
            
            for service_id in service_set:
                if service_id in mashup_sims:
                    total_similarity += mashup_sims[service_id]
        
        return total_similarity / len(service_set)
    
    def compute_diversity_score(self, service_set: List[str],
                               service_network: ServiceSimilarityNetwork) -> float:
        """
        Diversity Score: How diverse the services are within the set
        Formula: Diversity(S) = 1 - (2 * Σ_{i<j} Sim(s_i, s_j)) / (|S| * (|S| - 1))
        """
        if len(service_set) < 2:
            return 1.0  # Maximum diversity for single service
        
        total_similarity = 0.0
        pairs = 0
        
        # Get service indices for similarity matrix lookup
        service_indices = {}
        if hasattr(service_network, 'similarity_matrix'):
            # Build index mapping if not available
            all_services = list(service_network.service_vectors.keys())
            service_indices = {sid: i for i, sid in enumerate(all_services)}
            
            similarity_matrix = service_network.similarity_matrix
            
            for i in range(len(service_set)):
                for j in range(i+1, len(service_set)):
                    sid_i, sid_j = service_set[i], service_set[j]
                    
                    if sid_i in service_indices and sid_j in service_indices:
                        idx_i = service_indices[sid_i]
                        idx_j = service_indices[sid_j]
                        
                        if idx_i < similarity_matrix.shape[0] and idx_j < similarity_matrix.shape[1]:
                            sim = similarity_matrix[idx_i, idx_j]
                            total_similarity += sim
                            pairs += 1
        
        if pairs == 0:
            return 1.0
        
        avg_similarity = total_similarity / pairs
        diversity = 1 - avg_similarity
        
        return max(0.0, diversity)  # Ensure non-negative
    
    def compute_compatibility_score(self, service_set: List[str],
                                   historical_mashups: List[Mashup]) -> float:
        """
        Compatibility Score: How often services appear together in historical mashups
        Formula: Compatibility(S) = Σ_{M∈H} I(S ⊆ M.services) / |H|
        """
        if not service_set or not historical_mashups:
            return 0.0
        
        service_set_set = set(service_set)
        compatible_count = 0
        
        for mashup in historical_mashups:
            mashup_services_set = set(mashup.services)
            
            # Check if all services in the set appear in this mashup
            if service_set_set.issubset(mashup_services_set):
                compatible_count += 1
        
        return compatible_count / len(historical_mashups)
    
    def compute_total_score(self, service_set: List[str],
                           target_mashup_id: str,
                           mashup_service_similarity: Dict[str, Dict[str, float]],
                           service_network: ServiceSimilarityNetwork,
                           historical_mashups: List[Mashup]) -> ServiceSet:
        """
        Total Score: Weighted combination of all three scores
        Formula: Score(S) = λ1 * Coverage(S) + λ2 * Diversity(S) + λ3 * Compatibility(S)
        """
        coverage = self.compute_coverage_score(service_set, target_mashup_id, mashup_service_similarity)
        diversity = self.compute_diversity_score(service_set, service_network)
        compatibility = self.compute_compatibility_score(service_set, historical_mashups)
        
        total_score = (self.lambda1 * coverage + 
                      self.lambda2 * diversity + 
                      self.lambda3 * compatibility)
        
        return ServiceSet(
            services=service_set,
            score=total_score,
            coverage_score=coverage,
            diversity_score=diversity,
            compatibility_score=compatibility
        )

# ============================================================================
# Collaborative Filtering Enhancement
# ============================================================================

class CollaborativeFilteringEnhancer:
    """
    Enhances recommendations using collaborative filtering
    Section III-F: Collaborative Filtering Enhancement
    """
    
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        self.mashup_similarity_matrix = None
        self.service_popularity = {}
        
    def build_user_service_matrix(self, mashups: List[Mashup], services: List[WebService]):
        """Build user (mashup) - service matrix for collaborative filtering"""
        n_mashups = len(mashups)
        n_services = len(services)
        
        # Create mapping
        mashup_id_to_idx = {m.mashup_id: i for i, m in enumerate(mashups)}
        service_id_to_idx = {s.service_id: i for i, s in enumerate(services)}
        
        # Build matrix
        user_service_matrix = lil_matrix((n_mashups, n_services))
        
        for mashup in mashups:
            mashup_idx = mashup_id_to_idx[mashup.mashup_id]
            for service_id in mashup.services:
                if service_id in service_id_to_idx:
                    service_idx = service_id_to_idx[service_id]
                    user_service_matrix[mashup_idx, service_idx] = 1
        
        return user_service_matrix.tocsr(), mashup_id_to_idx, service_id_to_idx
    
    def compute_service_popularity(self, user_service_matrix: csr_matrix,
                                  service_id_to_idx: Dict[str, int]) -> Dict[str, float]:
        """Compute service popularity based on usage frequency"""
        service_popularity = {}
        
        # Sum columns to get usage counts
        usage_counts = user_service_matrix.sum(axis=0).A1
        
        for service_id, idx in service_id_to_idx.items():
            if idx < len(usage_counts):
                service_popularity[service_id] = usage_counts[idx]
        
        # Normalize
        max_count = max(service_popularity.values()) if service_popularity else 1
        
        for service_id in service_popularity:
            service_popularity[service_id] /= max_count
        
        self.service_popularity = service_popularity
        return service_popularity
    
    def find_similar_mashups(self, target_mashup_id: str,
                            mashup_similarity_matrix: csr_matrix,
                            mashup_id_to_idx: Dict[str, int]) -> List[Tuple[str, float]]:
        """Find k-nearest neighbor mashups for collaborative filtering"""
        if target_mashup_id not in mashup_id_to_idx:
            return []
        
        target_idx = mashup_id_to_idx[target_mashup_id]
        
        # Get similarities for target mashup
        similarities = mashup_similarity_matrix[target_idx].toarray().flatten()
        
        # Find top-k similar mashups (excluding self)
        similar_indices = np.argsort(similarities)[-self.k_neighbors-1:-1][::-1]
        
        # Map back to mashup IDs
        idx_to_mashup_id = {v: k for k, v in mashup_id_to_idx.items()}
        similar_mashups = []
        
        for idx in similar_indices:
            if idx in idx_to_mashup_id:
                mashup_id = idx_to_mashup_id[idx]
                similarity = similarities[idx]
                similar_mashups.append((mashup_id, similarity))
        
        return similar_mashups
    
    def predict_service_preferences(self, target_mashup_id: str,
                                   similar_mashups: List[Tuple[str, float]],
                                   user_service_matrix: csr_matrix,
                                   mashup_id_to_idx: Dict[str, int],
                                   service_id_to_idx: Dict[str, int]) -> Dict[str, float]:
        """
        Predict service preferences using user-based collaborative filtering
        Formula: P(u, s) = Σ_{v∈N(u)} sim(u, v) * r(v, s) / Σ_{v∈N(u)} |sim(u, v)|
        """
        if not similar_mashups:
            return {}
        
        target_idx = mashup_id_to_idx[target_mashup_id]
        n_services = user_service_matrix.shape[1]
        
        # Initialize preference scores
        preference_scores = np.zeros(n_services)
        total_similarity = 0.0
        
        for neighbor_mashup_id, similarity in similar_mashups:
            if neighbor_mashup_id in mashup_id_to_idx:
                neighbor_idx = mashup_id_to_idx[neighbor_mashup_id]
                
                # Get neighbor's service usage vector
                neighbor_services = user_service_matrix[neighbor_idx].toarray().flatten()
                
                # Weight by similarity
                preference_scores += similarity * neighbor_services
                total_similarity += abs(similarity)
        
        # Normalize
        if total_similarity > 0:
            preference_scores /= total_similarity
        
        # Map back to service IDs
        idx_to_service_id = {v: k for k, v in service_id_to_idx.items()}
        service_preferences = {}
        
        for idx, score in enumerate(preference_scores):
            if idx in idx_to_service_id and score > 0:
                service_id = idx_to_service_id[idx]
                service_preferences[service_id] = score
        
        return service_preferences
    
    def enhance_recommendations(self, candidate_sets: List[ServiceSet],
                               service_preferences: Dict[str, float],
                               enhancement_weight: float = 0.2) -> List[ServiceSet]:
        """
        Enhance candidate set scores using collaborative filtering predictions
        Formula: Enhanced_Score(S) = (1 - γ) * Original_Score(S) + γ * CF_Score(S)
        """
        enhanced_sets = []
        
        for service_set in candidate_sets:
            # Compute CF score for this set
            cf_score = 0.0
            for service_id in service_set.services:
                cf_score += service_preferences.get(service_id, 0)
            
            if service_set.services:
                cf_score /= len(service_set.services)
            
            # Combine with original score
            enhanced_score = ((1 - enhancement_weight) * service_set.score + 
                            enhancement_weight * cf_score)
            
            # Create enhanced service set
            enhanced_set = ServiceSet(
                services=service_set.services,
                score=enhanced_score,
                coverage_score=service_set.coverage_score,
                diversity_score=service_set.diversity_score,
                compatibility_score=service_set.compatibility_score
            )
            enhanced_sets.append(enhanced_set)
        
        return enhanced_sets

# ============================================================================
# Complete SSR Framework
# ============================================================================

class ServiceSetRecommendationFramework:
    """
    Complete Service Set Recommendation (SSR) Framework
    
    Implements the complete pipeline:
    1. Service Similarity Network Construction
    2. Mashup Similarity Computation
    3. Candidate Service Set Generation
    4. Service Set Scoring and Ranking
    5. Collaborative Filtering Enhancement
    """
    
    def __init__(self, 
                 alpha: float = 0.7, 
                 beta: float = 0.3,
                 lambda1: float = 0.4,
                 lambda2: float = 0.3,
                 lambda3: float = 0.3,
                 k_neighbors: int = 5,
                 max_set_size: int = 5):
        """
        Initialize SSR Framework with paper parameters
        """
        # Components
        self.service_network = ServiceSimilarityNetwork(alpha=alpha, beta=beta)
        self.mashup_similarity_calc = MashupSimilarityCalculator()
        self.candidate_generator = CandidateSetGenerator(max_set_size=max_set_size)
        self.scorer = ServiceSetScorer(lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
        self.cf_enhancer = CollaborativeFilteringEnhancer(k_neighbors=k_neighbors)
        
        # Data storage
        self.services = []
        self.mashups = []
        self.service_dict = {}
        self.mashup_dict = {}
        
        # Computed matrices
        self.mashup_similarity_matrix = None
        self.mashup_service_similarity = None
        self.user_service_matrix = None
        
        # Mappings
        self.mashup_id_to_idx = {}
        self.service_id_to_idx = {}
        
    def load_data(self, services_data: List[Dict], mashups_data: List[Dict]):
        """Load services and mashups data"""
        print("Loading data...")
        
        # Load services
        for service_data in services_data:
            service = WebService(
                service_id=service_data['id'],
                name=service_data.get('name', ''),
                description=service_data.get('description', ''),
                tags=service_data.get('tags', []),
                category=service_data.get('category', ''),
                provider=service_data.get('provider', ''),
                popularity=service_data.get('popularity', 0)
            )
            self.services.append(service)
            self.service_dict[service.service_id] = service
        
        # Load mashups
        for mashup_data in mashups_data:
            mashup = Mashup(
                mashup_id=mashup_data['id'],
                name=mashup_data.get('name', ''),
                description=mashup_data.get('description', ''),
                services=mashup_data.get('services', []),
                tags=mashup_data.get('tags', []),
                category=mashup_data.get('category', '')
            )
            self.mashups.append(mashup)
            self.mashup_dict[mashup.mashup_id] = mashup
        
        print(f"Loaded {len(self.services)} services and {len(self.mashups)} mashups")
        
        # Create mappings
        self.mashup_id_to_idx = {m.mashup_id: i for i, m in enumerate(self.mashups)}
        self.service_id_to_idx = {s.service_id: i for i, s in enumerate(self.services)}
    
    def build_similarity_networks(self):
        """Build service and mashup similarity networks"""
        print("\n" + "="*60)
        print("BUILDING SIMILARITY NETWORKS")
        print("="*60)
        
        # Build service similarity network
        print("\n1. Building service similarity network...")
        self.service_network.construct_network(self.services, self.mashups)
        
        # Compute mashup similarity
        print("\n2. Computing mashup similarity...")
        self.mashup_similarity_matrix = self.mashup_similarity_calc.compute_mashup_mashup_similarity(self.mashups)
        
        # Compute mashup-service similarity
        print("\n3. Computing mashup-service similarity...")
        self.mashup_service_similarity = self.mashup_similarity_calc.compute_mashup_service_similarity(
            self.mashups, self.services, self.service_network
        )
        
        # Compute service coverage
        print("\n4. Computing service coverage...")
        self.candidate_generator.compute_service_coverage(self.services, self.mashups)
        
        # Build user-service matrix for collaborative filtering
        print("\n5. Building user-service matrix for CF...")
        self.user_service_matrix, self.mashup_id_to_idx, self.service_id_to_idx = \
            self.cf_enhancer.build_user_service_matrix(self.mashups, self.services)
        
        print("\nSimilarity networks built successfully!")
    
    def recommend_service_sets(self, target_mashup_id: str, 
                              target_requirements: str = "",
                              n_recommendations: int = 10) -> List[ServiceSet]:
        """
        Recommend service sets for a target mashup
        
        Args:
            target_mashup_id: ID of the target mashup
            target_requirements: Textual requirements (optional)
            n_recommendations: Number of sets to recommend
            
        Returns:
            List of recommended service sets with scores
        """
        print("\n" + "="*60)
        print(f"RECOMMENDING SERVICE SETS FOR MASHUP: {target_mashup_id}")
        print("="*60)
        
        # Step 1: Generate candidate service sets
        print("\nStep 1: Generating candidate service sets...")
        
        # Get services from similar mashups
        candidate_sets = []
        
        # Strategy 1: From similar mashups
        similar_mashup_sets = self.candidate_generator.generate_by_similar_mashups(
            target_mashup_id, self.mashup_similarity_matrix, self.mashups, top_k=5
        )
        candidate_sets.extend(similar_mashup_sets)
        
        # Strategy 2: Get services from target mashup if it exists
        if target_mashup_id in self.mashup_dict:
            target_mashup = self.mashup_dict[target_mashup_id]
            target_services = target_mashup.services[:self.candidate_generator.max_set_size]
            
            # Expand with similar services
            similarity_sets = self.candidate_generator.generate_by_service_similarity(
                target_services, self.service_network, top_k_per_service=3
            )
            candidate_sets.extend(similarity_sets)
        
        # Strategy 3: Generate by coverage and diversity
        coverage_sets = self.candidate_generator.generate_by_coverage_diversity(
            self.services, target_requirements, top_n=20
        )
        candidate_sets.extend(coverage_sets)
        
        # Remove duplicates and invalid sets
        unique_sets = []
        seen = set()
        
        for service_set in candidate_sets:
            if not service_set:
                continue
                
            # Remove duplicates within set
            service_set = list(set(service_set))
            
            # Check size constraints
            if (len(service_set) >= self.candidate_generator.min_set_size and 
                len(service_set) <= self.candidate_generator.max_set_size):
                
                # Create tuple for deduplication
                set_tuple = tuple(sorted(service_set))
                
                if set_tuple not in seen:
                    seen.add(set_tuple)
                    unique_sets.append(service_set)
        
        print(f"  Generated {len(unique_sets)} unique candidate sets")
        
        # Step 2: Score candidate sets
        print("\nStep 2: Scoring candidate sets...")
        scored_sets = []
        
        for service_set in unique_sets:
            scored_set = self.scorer.compute_total_score(
                service_set=service_set,
                target_mashup_id=target_mashup_id,
                mashup_service_similarity=self.mashup_service_similarity,
                service_network=self.service_network,
                historical_mashups=self.mashups
            )
            scored_sets.append(scored_set)
        
        # Step 3: Collaborative filtering enhancement
        print("\nStep 3: Applying collaborative filtering enhancement...")
        
        # Find similar mashups
        similar_mashups = self.cf_enhancer.find_similar_mashups(
            target_mashup_id, self.mashup_similarity_matrix, self.mashup_id_to_idx
        )
        
        if similar_mashups:
            # Predict service preferences
            service_preferences = self.cf_enhancer.predict_service_preferences(
                target_mashup_id, similar_mashups, self.user_service_matrix,
                self.mashup_id_to_idx, self.service_id_to_idx
            )
            
            # Enhance recommendations
            enhanced_sets = self.cf_enhancer.enhance_recommendations(
                scored_sets, service_preferences, enhancement_weight=0.2
            )
        else:
            enhanced_sets = scored_sets
        
        # Step 4: Rank and select top recommendations
        print("\nStep 4: Ranking and selecting top recommendations...")
        
        # Sort by enhanced score
        enhanced_sets.sort(key=lambda x: x.score, reverse=True)
        
        # Select top N
        top_recommendations = enhanced_sets[:n_recommendations]
        
        # Display results
        print(f"\nTop {len(top_recommendations)} Recommended Service Sets:")
        print("-" * 80)
        
        for i, service_set in enumerate(top_recommendations, 1):
            service_names = []
            for service_id in service_set.services:
                if service_id in self.service_dict:
                    service_names.append(self.service_dict[service_id].name)
                else:
                    service_names.append(service_id)
            
            print(f"\n{i}. Overall Score: {service_set.score:.4f}")
            print(f"   Services: {', '.join(service_names)}")
            print(f"   Coverage: {service_set.coverage_score:.4f}, "
                  f"Diversity: {service_set.diversity_score:.4f}, "
                  f"Compatibility: {service_set.compatibility_score:.4f}")
        
        return top_recommendations
    
    def evaluate_recommendation(self, target_mashup_id: str, 
                               recommended_sets: List[ServiceSet]) -> Dict[str, float]:
        """
        Evaluate recommendation quality for a given mashup
        
        Returns precision, recall, and F1-score
        """
        if target_mashup_id not in self.mashup_dict:
            return {}
        
        target_mashup = self.mashup_dict[target_mashup_id]
        actual_services = set(target_mashup.services)
        
        # For each recommended set size
        evaluation_results = {}
        
        for k in [3, 5, 10]:
            if len(recommended_sets) >= k:
                # Get union of services in top-k sets
                recommended_services = set()
                for i in range(min(k, len(recommended_sets))):
                    recommended_services.update(recommended_sets[i].services)
                
                # Calculate metrics
                if recommended_services:
                    tp = len(actual_services.intersection(recommended_services))
                    
                    precision = tp / len(recommended_services) if recommended_services else 0
                    recall = tp / len(actual_services) if actual_services else 0
                    
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0
                    
                    evaluation_results[f'precision@{k}'] = precision
                    evaluation_results[f'recall@{k}'] = recall
                    evaluation_results[f'f1@{k}'] = f1
        
        return evaluation_results

# ============================================================================
# Data Generation and Demonstration
# ============================================================================

def generate_ssr_sample_data(n_services: int = 150, n_mashups: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """Generate sample data for SSR framework demonstration"""
    print("Generating sample data for SSR framework...")
    
    # Service categories and tags
    categories = [
        "Mapping", "Social", "Payment", "Travel", "Shopping", 
        "Video", "Audio", "Weather", "News", "Business", 
        "Education", "Health", "Finance", "Entertainment", "Communication"
    ]
    
    common_tags = {
        "Mapping": ["location", "maps", "geolocation", "navigation", "places"],
        "Social": ["social", "media", "networking", "sharing", "community"],
        "Payment": ["payment", "transaction", "money", "banking", "finance"],
        "Travel": ["travel", "hotels", "flights", "booking", "tourism"],
        "Video": ["video", "streaming", "media", "playback", "encoding"]
    }
    
    # Generate services
    services_data = []
    for i in range(n_services):
        category = random.choice(categories)
        tags = common_tags.get(category, [category.lower()])
        tags.extend(random.sample(["api", "web", "service", "rest", "cloud"], 3))
        
        services_data.append({
            'id': f'service_{i:03d}',
            'name': f'{category} Service {i}',
            'description': f'A comprehensive {category.lower()} API providing various functionalities for {category.lower()} applications. Includes robust features and easy integration.',
            'tags': tags,
            'category': category,
            'provider': random.choice(['Google', 'Amazon', 'Microsoft', 'IBM', 'Twitter', 'Facebook']),
            'popularity': random.randint(1, 1000)
        })
    
    # Generate mashups with realistic service combinations
    mashups_data = []
    
    # Define some common mashup patterns
    mashup_patterns = [
        ["Mapping", "Social"],  # Location-based social
        ["Travel", "Payment"],  # Travel booking
        ["Video", "Social"],    # Social video sharing
        ["Weather", "Mapping"], # Weather maps
        ["News", "Social"],     # Social news aggregator
    ]
    
    for i in range(n_mashups):
        # Choose a pattern or random combination
        if random.random() < 0.7 and mashup_patterns:
            pattern = random.choice(mashup_patterns)
            
            # Select services from these categories
            selected_services = []
            for category in pattern:
                category_services = [s for s in services_data if s['category'] == category]
                if category_services:
                    # Select 1-2 services from this category
                    n_select = random.randint(1, 2)
                    selected = random.sample(category_services, min(n_select, len(category_services)))
                    selected_services.extend([s['id'] for s in selected])
        else:
            # Random selection
            n_services_in_mashup = random.randint(2, 5)
            selected_services = random.sample([s['id'] for s in services_data], n_services_in_mashup)
        
        # Remove duplicates
        selected_services = list(set(selected_services))
        
        # Create mashup
        mashup_name = f"Mashup Application {i}"
        description = f"This mashup combines {len(selected_services)} services to create a powerful integrated application."
        
        # Generate tags based on services
        mashup_tags = []
        for service_id in selected_services:
            service = next((s for s in services_data if s['id'] == service_id), None)
            if service:
                mashup_tags.extend(service['tags'][:2])
        
        mashup_tags = list(set(mashup_tags))[:5]  # Limit to 5 unique tags
        
        mashups_data.append({
            'id': f'mashup_{i:03d}',
            'name': mashup_name,
            'description': description,
            'services': selected_services,
            'tags': mashup_tags,
            'category': random.choice(categories)
        })
    
    return services_data, mashups_data

def demonstrate_ssr_framework():
    """Demonstrate the complete SSR framework"""
    print("="*80)
    print("SERVICE SET RECOMMENDATION (SSR) FRAMEWORK DEMONSTRATION")
    print("Paper: 'A novel framework for service set recommendation in mashup creation'")
    print("="*80)
    
    # Generate sample data
    services_data, mashups_data = generate_ssr_sample_data(n_services=150, n_mashups=100)
    
    # Initialize SSR framework
    ssr = ServiceSetRecommendationFramework(
        alpha=0.7,      # Weight for content similarity
        beta=0.3,       # Weight for collaborative similarity
        lambda1=0.4,    # Weight for coverage score
        lambda2=0.3,    # Weight for diversity score
        lambda3=0.3,    # Weight for compatibility score
        k_neighbors=5,  # Number of neighbors for CF
        max_set_size=5  # Maximum services per set
    )
    
    # Load data
    ssr.load_data(services_data, mashups_data)
    
    # Build similarity networks
    ssr.build_similarity_networks()
    
    # Test with a sample mashup
    test_mashup_id = "mashup_050"  # Using a mashup from the middle of our data
    
    if test_mashup_id not in ssr.mashup_dict:
        # Use the first mashup if our test ID doesn't exist
        test_mashup_id = mashups_data[0]['id']
    
    print(f"\nUsing mashup for testing: {test_mashup_id}")
    
    # Get actual services used in this mashup
    actual_services = []
    if test_mashup_id in ssr.mashup_dict:
        actual_services = ssr.mashup_dict[test_mashup_id].services
        actual_service_names = []
        for service_id in actual_services:
            if service_id in ssr.service_dict:
                actual_service_names.append(ssr.service_dict[service_id].name)
        
        print(f"Actual services in this mashup: {', '.join(actual_service_names)}")
    
    # Generate recommendations
    target_requirements = "I need to create a location-based social application that integrates mapping with social media features."
    
    recommendations = ssr.recommend_service_sets(
        target_mashup_id=test_mashup_id,
        target_requirements=target_requirements,
        n_recommendations=10
    )
    
    # Evaluate recommendations
    if recommendations:
        evaluation = ssr.evaluate_recommendation(test_mashup_id, recommendations)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for metric, value in evaluation.items():
            print(f"{metric}: {value:.4f}")
    
    # Test with a new mashup (cold-start scenario)
    print("\n" + "="*80)
    print("COLD-START SCENARIO: New Mashup Creation")
    print("="*80)
    
    new_mashup_id = "new_mashup_001"
    new_requirements = "Create a travel booking application with payment integration and social sharing features."
    
    # For new mashup, we need to create a placeholder
    # In real scenario, we would add it to the system first
    print(f"\nRequirements: {new_requirements}")
    print("\nGenerating recommendations for new mashup...")
    
    # We can still use the framework by treating it as a mashup with no services
    recommendations_new = ssr.recommend_service_sets(
        target_mashup_id=new_mashup_id,
        target_requirements=new_requirements,
        n_recommendations=5
    )
    
    print("\nTop 5 recommendations for new mashup:")
    for i, service_set in enumerate(recommendations_new[:5], 1):
        service_names = []
        for service_id in service_set.services:
            if service_id in ssr.service_dict:
                service_names.append(ssr.service_dict[service_id].name)
        
        print(f"{i}. {', '.join(service_names)} (Score: {service_set.score:.4f})")
    
    return ssr, recommendations

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run demonstration
    ssr_framework, recommendations = demonstrate_ssr_framework()
    
    print("\n" + "="*80)
    print("SSR FRAMEWORK IMPLEMENTATION COMPLETE")
    print("="*80)
    
    # Display framework statistics
    print(f"\nFramework Statistics:")
    print(f"- Services: {len(ssr_framework.services)}")
    print(f"- Mashups: {len(ssr_framework.mashups)}")
    print(f"- Service network edges: {ssr_framework.service_network.graph.number_of_edges()}")
    print(f"- Mashup similarity matrix density: {(ssr_framework.mashup_similarity_matrix.nnz / (ssr_framework.mashup_similarity_matrix.shape[0] * ssr_framework.mashup_similarity_matrix.shape[1])) * 100:.2f}%")