"""
Complete Implementation of:
"Service package recommendation for mashup development based on a multi-level relational network"
Cao, J., Lu, Y., & Zhu, N. (2016). ICSOC, pp. 666-674.

Multi-level Relational Network (MLRN) Framework for Service Package Recommendation
Components:
1. Multi-level Network Construction (Service-Tag-Mashup)
2. Relation-aware Random Walk
3. Service Package Generation and Scoring
4. Quadratic Knapsack Optimization for Package Selection
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
import warnings
from collections import defaultdict, Counter
import math
import random
from dataclasses import dataclass
from itertools import combinations, product
import heapq
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WebService:
    """Web Service/API representation with tags"""
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
    """Mashup representation with services and tags"""
    mashup_id: str
    name: str
    description: str
    tags: List[str] = None
    used_services: List[str] = None
    created_time: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.used_services is None:
            self.used_services = []

@dataclass
class Tag:
    """Tag representation in multi-level network"""
    tag_id: str
    name: str
    frequency: int = 0
    type: str = ""  # service tag or mashup tag

@dataclass
class ServicePackage:
    """Recommended package of services for mashup development"""
    services: List[str]
    score: float = 0.0
    coverage_score: float = 0.0
    compatibility_score: float = 0.0
    diversity_score: float = 0.0
    popularity_score: float = 0.0

# ============================================================================
# Multi-Level Relational Network Construction
# ============================================================================

class MultiLevelNetwork:
    """
    Constructs Multi-Level Relational Network (MLRN)
    Section 3.1: Multi-level Network Construction
    
    Network levels:
    1. Service-Service relations (co-usage)
    2. Service-Tag relations
    3. Tag-Tag relations (co-occurrence)
    4. Tag-Mashup relations
    5. Mashup-Mashup relations (service/tag similarity)
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        """
        Args:
            alpha: Weight for service-service relations
            beta: Weight for service-tag relations  
            gamma: Weight for tag-tag relations
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Network graphs
        self.service_graph = nx.Graph()  # Service level
        self.tag_graph = nx.Graph()      # Tag level
        self.mashup_graph = nx.Graph()   # Mashup level
        self.bipartite_st = None         # Service-Tag bipartite
        self.bipartite_tm = None         # Tag-Mashup bipartite
        
        # Node mappings
        self.service_to_idx = {}
        self.tag_to_idx = {}
        self.mashup_to_idx = {}
        
        # Data storage
        self.services = []
        self.tags = []
        self.mashups = []
        
    def extract_tags_from_services(self, services: List[WebService]) -> List[Tag]:
        """Extract and normalize tags from services"""
        tag_counter = Counter()
        tag_to_services = defaultdict(list)
        
        for service in services:
            for tag_name in service.tags:
                # Normalize tag name
                normalized_tag = tag_name.lower().strip()
                tag_counter[normalized_tag] += 1
                tag_to_services[normalized_tag].append(service.service_id)
        
        # Create Tag objects
        tags = []
        for tag_name, freq in tag_counter.items():
            tag = Tag(
                tag_id=f"tag_{tag_name}",
                name=tag_name,
                frequency=freq,
                type="service"
            )
            tags.append(tag)
        
        return tags
    
    def extract_tags_from_mashups(self, mashups: List[Mashup]) -> List[Tag]:
        """Extract tags from mashups"""
        tag_counter = Counter()
        
        for mashup in mashups:
            for tag_name in mashup.tags:
                normalized_tag = tag_name.lower().strip()
                tag_counter[normalized_tag] += 1
        
        # Create Tag objects for mashup tags
        tags = []
        for tag_name, freq in tag_counter.items():
            tag = Tag(
                tag_id=f"mashup_tag_{tag_name}",
                name=tag_name,
                frequency=freq,
                type="mashup"
            )
            tags.append(tag)
        
        return tags
    
    def build_service_service_relations(self, services: List[WebService], 
                                       mashups: List[Mashup]) -> nx.Graph:
        """
        Build service-service relations based on co-usage in mashups
        Weighted by co-occurrence frequency
        """
        print("Building service-service relations...")
        
        service_graph = nx.Graph()
        
        # Add service nodes
        for service in services:
            service_graph.add_node(service.service_id, 
                                  type='service',
                                  popularity=service.popularity)
        
        # Count co-occurrences in mashups
        cooccurrence_counts = defaultdict(int)
        
        for mashup in mashups:
            services_in_mashup = mashup.used_services
            
            # Count all pairs of services in this mashup
            for i in range(len(services_in_mashup)):
                for j in range(i+1, len(services_in_mashup)):
                    service_i = services_in_mashup[i]
                    service_j = services_in_mashup[j]
                    
                    # Create sorted tuple for consistent key
                    pair = tuple(sorted([service_i, service_j]))
                    cooccurrence_counts[pair] += 1
        
        # Add edges with weights
        edges_added = 0
        for (service_i, service_j), count in cooccurrence_counts.items():
            if count > 0:
                # Normalize weight by total mashups
                weight = count / len(mashups)
                service_graph.add_edge(service_i, service_j, 
                                      weight=weight, 
                                      cooccurrence=count)
                edges_added += 1
        
        print(f"  Added {edges_added} service-service edges")
        return service_graph
    
    def build_service_tag_relations(self, services: List[WebService], 
                                   service_tags: List[Tag]) -> nx.Graph:
        """
        Build service-tag bipartite relations
        Weighted by TF-IDF of tag importance
        """
        print("Building service-tag relations...")
        
        bipartite_graph = nx.Graph()
        
        # Add service nodes
        for service in services:
            bipartite_graph.add_node(service.service_id, type='service')
        
        # Add tag nodes
        for tag in service_tags:
            bipartite_graph.add_node(tag.tag_id, type='tag', name=tag.name)
        
        # Count tag frequencies per service
        service_tag_counts = defaultdict(Counter)
        tag_total_counts = Counter()
        
        for service in services:
            for tag_name in service.tags:
                normalized_tag = tag_name.lower().strip()
                tag_id = f"tag_{normalized_tag}"
                service_tag_counts[service.service_id][tag_id] += 1
                tag_total_counts[tag_id] += 1
        
        # Calculate TF-IDF weights
        n_services = len(services)
        
        edges_added = 0
        for service_id, tag_counts in service_tag_counts.items():
            for tag_id, tf in tag_counts.items():
                # Term Frequency (TF)
                tf_score = tf
                
                # Inverse Document Frequency (IDF)
                df = tag_total_counts[tag_id]
                idf_score = math.log((n_services + 1) / (df + 1)) + 1
                
                # TF-IDF weight
                weight = tf_score * idf_score
                
                # Add edge with TF-IDF weight
                bipartite_graph.add_edge(service_id, tag_id, weight=weight)
                edges_added += 1
        
        print(f"  Added {edges_added} service-tag edges")
        return bipartite_graph
    
    def build_tag_tag_relations(self, service_tags: List[Tag], 
                               mashup_tags: List[Tag],
                               services: List[WebService],
                               mashups: List[Mashup]) -> nx.Graph:
        """
        Build tag-tag relations based on:
        1. Co-occurrence in services
        2. Co-occurrence in mashups
        3. Semantic similarity (optional)
        """
        print("Building tag-tag relations...")
        
        tag_graph = nx.Graph()
        
        # Combine all tags
        all_tags = service_tags + mashup_tags
        
        # Add tag nodes
        for tag in all_tags:
            tag_graph.add_node(tag.tag_id, 
                              name=tag.name,
                              type=tag.type,
                              frequency=tag.frequency)
        
        # 1. Tag co-occurrence in services
        service_tag_cooccurrence = defaultdict(Counter)
        
        for service in services:
            service_tag_ids = [f"tag_{tag.lower().strip()}" for tag in service.tags]
            
            for i in range(len(service_tag_ids)):
                for j in range(i+1, len(service_tag_ids)):
                    tag_i = service_tag_ids[i]
                    tag_j = service_tag_ids[j]
                    
                    service_tag_cooccurrence[tag_i][tag_j] += 1
                    service_tag_cooccurrence[tag_j][tag_i] += 1
        
        # 2. Tag co-occurrence in mashups
        mashup_tag_cooccurrence = defaultdict(Counter)
        
        for mashup in mashups:
            mashup_tag_ids = [f"mashup_tag_{tag.lower().strip()}" for tag in mashup.tags]
            
            for i in range(len(mashup_tag_ids)):
                for j in range(i+1, len(mashup_tag_ids)):
                    tag_i = mashup_tag_ids[i]
                    tag_j = mashup_tag_ids[j]
                    
                    mashup_tag_cooccurrence[tag_i][tag_j] += 1
                    mashup_tag_cooccurrence[tag_j][tag_i] += 1
        
        # Add edges with combined weights
        edges_added = 0
        
        # Add service tag co-occurrence edges
        for tag_i, cooccurring_tags in service_tag_cooccurrence.items():
            for tag_j, count in cooccurring_tags.items():
                if tag_i != tag_j:
                    weight = count / len(services)  # Normalize
                    tag_graph.add_edge(tag_i, tag_j, 
                                      weight=weight,
                                      type='service_cooccurrence')
                    edges_added += 1
        
        # Add mashup tag co-occurrence edges
        for tag_i, cooccurring_tags in mashup_tag_cooccurrence.items():
            for tag_j, count in cooccurring_tags.items():
                if tag_i != tag_j:
                    weight = count / len(mashups)  # Normalize
                    
                    # If edge already exists, combine weights
                    if tag_graph.has_edge(tag_i, tag_j):
                        current_weight = tag_graph[tag_i][tag_j]['weight']
                        tag_graph[tag_i][tag_j]['weight'] = (current_weight + weight) / 2
                        tag_graph[tag_i][tag_j]['type'] = 'combined'
                    else:
                        tag_graph.add_edge(tag_i, tag_j,
                                         weight=weight,
                                         type='mashup_cooccurrence')
                        edges_added += 1
        
        print(f"  Added {edges_added} tag-tag edges")
        return tag_graph
    
    def build_tag_mashup_relations(self, mashups: List[Mashup],
                                  mashup_tags: List[Tag]) -> nx.Graph:
        """
        Build tag-mashup bipartite relations
        Weighted by tag importance in mashup
        """
        print("Building tag-mashup relations...")
        
        bipartite_graph = nx.Graph()
        
        # Add mashup nodes
        for mashup in mashups:
            bipartite_graph.add_node(mashup.mashup_id, type='mashup')
        
        # Add tag nodes
        for tag in mashup_tags:
            bipartite_graph.add_node(tag.tag_id, type='tag', name=tag.name)
        
        # Create mappings for quick lookup
        tag_name_to_id = {tag.name: tag.tag_id for tag in mashup_tags}
        
        # Count tag occurrences per mashup
        mashup_tag_counts = defaultdict(Counter)
        tag_total_counts = Counter()
        
        for mashup in mashups:
            for tag_name in mashup.tags:
                normalized_tag = tag_name.lower().strip()
                tag_id = f"mashup_tag_{normalized_tag}"
                
                mashup_tag_counts[mashup.mashup_id][tag_id] += 1
                tag_total_counts[tag_id] += 1
        
        # Calculate TF-IDF weights
        n_mashups = len(mashups)
        
        edges_added = 0
        for mashup_id, tag_counts in mashup_tag_counts.items():
            for tag_id, tf in tag_counts.items():
                # Term Frequency (TF)
                tf_score = tf
                
                # Inverse Document Frequency (IDF)
                df = tag_total_counts[tag_id]
                idf_score = math.log((n_mashups + 1) / (df + 1)) + 1
                
                # TF-IDF weight
                weight = tf_score * idf_score
                
                # Add edge with TF-IDF weight
                bipartite_graph.add_edge(mashup_id, tag_id, weight=weight)
                edges_added += 1
        
        print(f"  Added {edges_added} tag-mashup edges")
        return bipartite_graph
    
    def build_mashup_mashup_relations(self, mashups: List[Mashup],
                                     services: List[WebService]) -> nx.Graph:
        """
        Build mashup-mashup relations based on:
        1. Service overlap (Jaccard similarity)
        2. Tag similarity
        3. Content similarity (optional)
        """
        print("Building mashup-mashup relations...")
        
        mashup_graph = nx.Graph()
        
        # Add mashup nodes
        for mashup in mashups:
            mashup_graph.add_node(mashup.mashup_id, 
                                 type='mashup',
                                 service_count=len(mashup.used_services))
        
        # Calculate service-based similarity
        edges_added = 0
        
        for i in range(len(mashups)):
            for j in range(i+1, len(mashups)):
                mashup_i = mashups[i]
                mashup_j = mashups[j]
                
                # Service Jaccard similarity
                services_i = set(mashup_i.used_services)
                services_j = set(mashup_j.used_services)
                
                if services_i and services_j:
                    intersection = len(services_i.intersection(services_j))
                    union = len(services_i.union(services_j))
                    
                    if union > 0:
                        service_similarity = intersection / union
                        
                        # Tag Jaccard similarity
                        tags_i = set(tag.lower().strip() for tag in mashup_i.tags)
                        tags_j = set(tag.lower().strip() for tag in mashup_j.tags)
                        
                        if tags_i and tags_j:
                            tag_intersection = len(tags_i.intersection(tags_j))
                            tag_union = len(tags_i.union(tags_j))
                            tag_similarity = tag_intersection / tag_union if tag_union > 0 else 0
                        else:
                            tag_similarity = 0
                        
                        # Combine similarities (equal weight)
                        combined_similarity = (service_similarity + tag_similarity) / 2
                        
                        if combined_similarity > 0.1:  # Threshold
                            mashup_graph.add_edge(mashup_i.mashup_id, mashup_j.mashup_id,
                                                 weight=combined_similarity,
                                                 service_sim=service_similarity,
                                                 tag_sim=tag_similarity)
                            edges_added += 1
        
        print(f"  Added {edges_added} mashup-mashup edges")
        return mashup_graph
    
    def build_complete_network(self, services: List[WebService], 
                              mashups: List[Mashup]) -> Dict[str, nx.Graph]:
        """
        Build complete multi-level relational network
        
        Returns dictionary of all network components
        """
        print("\n" + "="*60)
        print("BUILDING MULTI-LEVEL RELATIONAL NETWORK")
        print("="*60)
        
        self.services = services
        self.mashups = mashups
        
        # Extract tags
        print("\n1. Extracting tags...")
        service_tags = self.extract_tags_from_services(services)
        mashup_tags = self.extract_tags_from_mashups(mashups)
        self.tags = service_tags + mashup_tags
        
        print(f"  Service tags: {len(service_tags)}")
        print(f"  Mashup tags: {len(mashup_tags)}")
        
        # Build individual network components
        print("\n2. Building network components...")
        
        self.service_graph = self.build_service_service_relations(services, mashups)
        self.bipartite_st = self.build_service_tag_relations(services, service_tags)
        self.tag_graph = self.build_tag_tag_relations(service_tags, mashup_tags, services, mashups)
        self.bipartite_tm = self.build_tag_mashup_relations(mashups, mashup_tags)
        self.mashup_graph = self.build_mashup_mashup_relations(mashups, services)
        
        # Create node mappings
        for idx, service in enumerate(services):
            self.service_to_idx[service.service_id] = idx
        
        for idx, tag in enumerate(self.tags):
            self.tag_to_idx[tag.tag_id] = idx
        
        for idx, mashup in enumerate(mashups):
            self.mashup_to_idx[mashup.mashup_id] = idx
        
        # Build unified network (optional)
        unified_network = self.build_unified_network()
        
        print("\nMulti-level network construction complete!")
        
        return {
            'service_graph': self.service_graph,
            'tag_graph': self.tag_graph,
            'mashup_graph': self.mashup_graph,
            'bipartite_st': self.bipartite_st,
            'bipartite_tm': self.bipartite_tm,
            'unified_network': unified_network
        }
    
    def build_unified_network(self) -> nx.Graph:
        """
        Build unified network combining all levels
        Used for global random walk
        """
        print("\nBuilding unified network...")
        
        unified_graph = nx.Graph()
        
        # Add all nodes
        for service in self.services:
            unified_graph.add_node(service.service_id, type='service')
        
        for tag in self.tags:
            unified_graph.add_node(tag.tag_id, type='tag')
        
        for mashup in self.mashups:
            unified_graph.add_node(mashup.mashup_id, type='mashup')
        
        # Add edges from all components
        edges_added = 0
        
        # Service-service edges
        for u, v, data in self.service_graph.edges(data=True):
            unified_graph.add_edge(u, v, **data, edge_type='service-service')
            edges_added += 1
        
        # Service-tag edges
        for u, v, data in self.bipartite_st.edges(data=True):
            unified_graph.add_edge(u, v, **data, edge_type='service-tag')
            edges_added += 1
        
        # Tag-tag edges
        for u, v, data in self.tag_graph.edges(data=True):
            unified_graph.add_edge(u, v, **data, edge_type='tag-tag')
            edges_added += 1
        
        # Tag-mashup edges
        for u, v, data in self.bipartite_tm.edges(data=True):
            unified_graph.add_edge(u, v, **data, edge_type='tag-mashup')
            edges_added += 1
        
        # Mashup-mashup edges
        for u, v, data in self.mashup_graph.edges(data=True):
            unified_graph.add_edge(u, v, **data, edge_type='mashup-mashup')
            edges_added += 1
        
        print(f"  Unified network: {unified_graph.number_of_nodes()} nodes, "
              f"{edges_added} edges")
        
        return unified_graph

# ============================================================================
# Relation-aware Random Walk
# ============================================================================

class RelationAwareRandomWalk:
    """
    Performs relation-aware random walk on multi-level network
    Section 3.2: Relation-aware Random Walk
    
    Key features:
    1. Type-aware transition probabilities
    2. Restart mechanism for query-specific walks
    3. Multi-level path exploration
    """
    
    def __init__(self, restart_prob: float = 0.2, 
                 walk_length: int = 50,
                 num_walks: int = 100):
        """
        Args:
            restart_prob: Probability of restarting at query node
            walk_length: Length of each random walk
            num_walks: Number of walks to perform
        """
        self.restart_prob = restart_prob
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.transition_probs = {}
        self.node_visit_counts = {}
        
    def compute_transition_probabilities(self, graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Compute transition probabilities for each node
        Considering node types and edge weights
        """
        print("Computing transition probabilities...")
        
        transition_probs = {}
        
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            
            if not neighbors:
                transition_probs[node] = {}
                continue
            
            # Get edge weights
            edge_weights = []
            for neighbor in neighbors:
                edge_data = graph.get_edge_data(node, neighbor)
                weight = edge_data.get('weight', 1.0)
                edge_weights.append(weight)
            
            # Normalize weights to get probabilities
            total_weight = sum(edge_weights)
            if total_weight > 0:
                probs = [w / total_weight for w in edge_weights]
                transition_probs[node] = dict(zip(neighbors, probs))
            else:
                # Equal probability if no weights
                prob = 1.0 / len(neighbors)
                transition_probs[node] = {neighbor: prob for neighbor in neighbors}
        
        self.transition_probs = transition_probs
        return transition_probs
    
    def type_aware_transition(self, current_node: str, next_candidates: List[str],
                             graph: nx.Graph) -> Dict[str, float]:
        """
        Compute type-aware transition probabilities
        Favors transitions within same network level
        """
        current_type = graph.nodes[current_node].get('type', 'unknown')
        
        probs = {}
        type_weights = {}
        
        for candidate in next_candidates:
            candidate_type = graph.nodes[candidate].get('type', 'unknown')
            
            # Assign weight based on type compatibility
            if current_type == candidate_type:
                # Same type: higher weight
                type_weight = 1.0
            elif (current_type, candidate_type) in [('service', 'tag'), ('tag', 'service'),
                                                   ('tag', 'mashup'), ('mashup', 'tag')]:
                # Direct bipartite relations: medium weight
                type_weight = 0.7
            else:
                # Indirect relations: lower weight
                type_weight = 0.3
            
            type_weights[candidate] = type_weight
        
        # Normalize type weights
        total_type_weight = sum(type_weights.values())
        if total_type_weight > 0:
            for candidate, weight in type_weights.items():
                probs[candidate] = weight / total_type_weight
        
        return probs
    
    def perform_random_walk(self, start_node: str, graph: nx.Graph,
                           query_nodes: List[str] = None) -> List[str]:
        """
        Perform a single random walk with restart
        
        Args:
            start_node: Starting node for the walk
            graph: Network graph
            query_nodes: Nodes to restart to (if None, restart to start_node)
        """
        if query_nodes is None:
            query_nodes = [start_node]
        
        walk = [start_node]
        current_node = start_node
        
        for step in range(self.walk_length - 1):
            # Check for restart
            if random.random() < self.restart_prob:
                # Restart to one of the query nodes
                next_node = random.choice(query_nodes)
            else:
                # Get neighbors
                neighbors = list(graph.neighbors(current_node))
                
                if not neighbors:
                    # If no neighbors, restart
                    next_node = random.choice(query_nodes)
                else:
                    # Get transition probabilities
                    if current_node in self.transition_probs:
                        probs = self.transition_probs[current_node]
                        
                        # Select next node based on probabilities
                        neighbors_list = list(probs.keys())
                        prob_values = list(probs.values())
                        
                        next_node = random.choices(neighbors_list, weights=prob_values)[0]
                    else:
                        # Fallback: uniform random selection
                        next_node = random.choice(neighbors)
            
            walk.append(next_node)
            current_node = next_node
        
        return walk
    
    def perform_multiple_walks(self, start_nodes: List[str], graph: nx.Graph) -> Dict[str, float]:
        """
        Perform multiple random walks and compute node visit frequencies
        
        Returns: Dictionary of node -> visit probability
        """
        print(f"Performing {self.num_walks} random walks from {len(start_nodes)} start nodes...")
        
        # Initialize visit counts
        visit_counts = defaultdict(int)
        total_steps = 0
        
        # Compute transition probabilities if not already computed
        if not self.transition_probs:
            self.compute_transition_probabilities(graph)
        
        # Perform walks
        for walk_idx in range(self.num_walks):
            # Select random start node from start_nodes
            start_node = random.choice(start_nodes)
            
            # Perform random walk
            walk = self.perform_random_walk(start_node, graph, start_nodes)
            
            # Count visits (excluding start node to avoid bias)
            for node in walk[1:]:  # Skip start node
                visit_counts[node] += 1
            
            total_steps += len(walk) - 1  # Exclude start node
        
        # Convert counts to probabilities
        visit_probs = {}
        if total_steps > 0:
            for node, count in visit_counts.items():
                visit_probs[node] = count / total_steps
        
        # Normalize to sum to 1
        total_prob = sum(visit_probs.values())
        if total_prob > 0:
            for node in visit_probs:
                visit_probs[node] /= total_prob
        
        self.node_visit_counts = visit_probs
        return visit_probs
    
    def get_service_relevance_scores(self, start_nodes: List[str], 
                                    graph: nx.Graph) -> Dict[str, float]:
        """
        Get relevance scores for services based on random walks
        """
        visit_probs = self.perform_multiple_walks(start_nodes, graph)
        
        # Filter for service nodes only
        service_scores = {}
        for node, prob in visit_probs.items():
            if graph.nodes[node].get('type') == 'service':
                service_scores[node] = prob
        
        return service_scores

# ============================================================================
# Service Package Generation
# ============================================================================

class ServicePackageGenerator:
    """
    Generates and scores candidate service packages
    Section 3.3: Service Package Generation and Scoring
    """
    
    def __init__(self, min_package_size: int = 2, max_package_size: int = 5,
                 coverage_weight: float = 0.4, compatibility_weight: float = 0.3,
                 diversity_weight: float = 0.2, popularity_weight: float = 0.1):
        """
        Args:
            min_package_size: Minimum services in a package
            max_package_size: Maximum services in a package
            coverage_weight: Weight for requirement coverage
            compatibility_weight: Weight for service compatibility
            diversity_weight: Weight for functional diversity
            popularity_weight: Weight for service popularity
        """
        self.min_package_size = min_package_size
        self.max_package_size = max_package_size
        self.coverage_weight = coverage_weight
        self.compatibility_weight = compatibility_weight
        self.diversity_weight = diversity_weight
        self.popularity_weight = popularity_weight
        
        self.service_relevance_scores = {}
        self.service_similarity_matrix = None
        
    def compute_coverage_score(self, package: List[str], 
                              relevance_scores: Dict[str, float]) -> float:
        """
        Compute how well the package covers query requirements
        
        Formula: Coverage(P) = Σ_{s∈P} Relevance(s) / |P|
        """
        if not package:
            return 0.0
        
        total_relevance = sum(relevance_scores.get(s, 0) for s in package)
        return total_relevance / len(package)
    
    def compute_compatibility_score(self, package: List[str],
                                   service_graph: nx.Graph) -> float:
        """
        Compute compatibility among services in package
        Based on historical co-usage in mashups
        
        Formula: Compatibility(P) = Σ_{i<j} Sim(s_i, s_j) / C(|P|, 2)
        """
        if len(package) < 2:
            return 1.0  # Maximum compatibility for single service
        
        total_similarity = 0.0
        pair_count = 0
        
        for i in range(len(package)):
            for j in range(i+1, len(package)):
                service_i = package[i]
                service_j = package[j]
                
                # Get edge weight from service graph
                if service_graph.has_edge(service_i, service_j):
                    weight = service_graph[service_i][service_j].get('weight', 0)
                    total_similarity += weight
                
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        return total_similarity / pair_count
    
    def compute_diversity_score(self, package: List[str],
                               service_graph: nx.Graph) -> float:
        """
        Compute functional diversity of services in package
        Inverse of compatibility
        
        Formula: Diversity(P) = 1 - Compatibility(P)
        """
        compatibility = self.compute_compatibility_score(package, service_graph)
        diversity = 1 - compatibility
        
        # Ensure non-negative
        return max(0.0, diversity)
    
    def compute_popularity_score(self, package: List[str],
                                services: List[WebService]) -> float:
        """
        Compute popularity score based on individual service popularity
        
        Formula: Popularity(P) = Σ_{s∈P} Popularity(s) / |P|
        """
        if not package:
            return 0.0
        
        # Create mapping for quick lookup
        service_dict = {s.service_id: s for s in services}
        
        total_popularity = 0
        for service_id in package:
            if service_id in service_dict:
                total_popularity += service_dict[service_id].popularity
        
        # Normalize by package size
        return total_popularity / len(package)
    
    def generate_candidate_packages(self, candidate_services: List[str],
                                   max_candidates: int = 100) -> List[List[str]]:
        """
        Generate candidate service packages from candidate services
        
        Uses heuristic sampling to avoid combinatorial explosion
        """
        candidate_packages = []
        n_services = len(candidate_services)
        
        if n_services < self.min_package_size:
            return candidate_packages
        
        # Generate packages of different sizes
        for package_size in range(self.min_package_size, 
                                 min(self.max_package_size, n_services) + 1):
            
            # Number of combinations for this size
            total_combinations = math.comb(n_services, package_size)
            
            if total_combinations <= max_candidates:
                # Generate all combinations if manageable
                for combo in combinations(candidate_services, package_size):
                    candidate_packages.append(list(combo))
            else:
                # Sample random combinations
                n_samples = min(max_candidates, total_combinations)
                for _ in range(n_samples):
                    package = random.sample(candidate_services, package_size)
                    candidate_packages.append(sorted(package))  # Sort for consistency
        
        # Remove duplicates
        unique_packages = []
        seen = set()
        
        for package in candidate_packages:
            package_tuple = tuple(sorted(package))
            if package_tuple not in seen:
                seen.add(package_tuple)
                unique_packages.append(package)
        
        print(f"Generated {len(unique_packages)} candidate packages")
        return unique_packages
    
    def score_package(self, package: List[str],
                     relevance_scores: Dict[str, float],
                     service_graph: nx.Graph,
                     services: List[WebService]) -> ServicePackage:
        """
        Compute overall score for a service package
        
        Formula from paper:
        Score(P) = w1 * Coverage(P) + w2 * Compatibility(P) + 
                  w3 * Diversity(P) + w4 * Popularity(P)
        """
        # Compute individual scores
        coverage = self.compute_coverage_score(package, relevance_scores)
        compatibility = self.compute_compatibility_score(package, service_graph)
        diversity = self.compute_diversity_score(package, service_graph)
        popularity = self.compute_popularity_score(package, services)
        
        # Normalize scores to [0, 1] range
        coverage_norm = min(1.0, coverage)
        compatibility_norm = min(1.0, compatibility)
        diversity_norm = min(1.0, diversity)
        
        # Normalize popularity (assuming max popularity is known)
        max_popularity = max([s.popularity for s in services]) if services else 1
        popularity_norm = popularity / max_popularity if max_popularity > 0 else 0
        
        # Compute weighted score
        total_score = (self.coverage_weight * coverage_norm +
                      self.compatibility_weight * compatibility_norm +
                      self.diversity_weight * diversity_norm +
                      self.popularity_weight * popularity_norm)
        
        return ServicePackage(
            services=package,
            score=total_score,
            coverage_score=coverage_norm,
            compatibility_score=compatibility_norm,
            diversity_score=diversity_norm,
            popularity_score=popularity_norm
        )

# ============================================================================
# Quadratic Knapsack Optimization
# ============================================================================

class QuadraticKnapsackOptimizer:
    """
    Optimizes service package selection using Quadratic Knapsack Problem (QKP)
    Section 3.4: Quadratic Knapsack Optimization
    
    Maximizes: Σ_i w_i x_i + Σ_{i<j} c_{ij} x_i x_j
    Subject to: Σ_i s_i x_i ≤ B, x_i ∈ {0, 1}
    
    Where:
    - w_i: Individual score of service i
    - c_{ij}: Compatibility score between services i and j
    - s_i: Size/cost of service i
    - B: Budget/package size limit
    - x_i: Binary variable indicating if service i is selected
    """
    
    def __init__(self, budget: int = 5, max_iterations: int = 1000):
        """
        Args:
            budget: Maximum package size (B)
            max_iterations: Maximum iterations for optimization
        """
        self.budget = budget
        self.max_iterations = max_iterations
        
    def solve_qkp_greedy(self, services: List[str],
                        individual_scores: Dict[str, float],
                        compatibility_matrix: Dict[Tuple[str, str], float]) -> List[str]:
        """
        Solve QKP using greedy heuristic
        
        Returns: List of selected service IDs
        """
        n_services = len(services)
        
        # Initialize solution
        selected = []
        remaining = services.copy()
        current_budget = 0
        
        # Sort services by individual score (descending)
        sorted_services = sorted(services, 
                                key=lambda s: individual_scores.get(s, 0),
                                reverse=True)
        
        # Greedy selection
        while remaining and current_budget < self.budget:
            best_service = None
            best_marginal_gain = -float('inf')
            
            for service in remaining[:min(50, len(remaining))]:  # Consider top 50
                if current_budget + 1 > self.budget:
                    continue
                
                # Compute marginal gain
                marginal_gain = individual_scores.get(service, 0)
                
                # Add compatibility with already selected services
                for selected_service in selected:
                    pair = tuple(sorted([service, selected_service]))
                    if pair in compatibility_matrix:
                        marginal_gain += compatibility_matrix[pair]
                
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_service = service
            
            if best_service:
                selected.append(best_service)
                remaining.remove(best_service)
                current_budget += 1
            else:
                break
        
        return selected
    
    def solve_qkp_genetic(self, services: List[str],
                         individual_scores: Dict[str, float],
                         compatibility_matrix: Dict[Tuple[str, str], float],
                         population_size: int = 50,
                         generations: int = 100) -> List[str]:
        """
        Solve QKP using genetic algorithm
        """
        print(f"Solving QKP with genetic algorithm (pop={population_size}, gen={generations})...")
        
        n_services = len(services)
        
        # Fitness function
        def fitness(solution: List[int]) -> float:
            """Compute fitness of a binary solution vector"""
            total_score = 0.0
            
            # Check budget constraint
            selected_count = sum(solution)
            if selected_count > self.budget:
                return -float('inf')  # Penalize invalid solutions
            
            # Individual scores
            for i, selected in enumerate(solution):
                if selected:
                    service_id = services[i]
                    total_score += individual_scores.get(service_id, 0)
            
            # Pairwise compatibility scores
            for i in range(n_services):
                for j in range(i+1, n_services):
                    if solution[i] and solution[j]:
                        service_i = services[i]
                        service_j = services[j]
                        pair = tuple(sorted([service_i, service_j]))
                        if pair in compatibility_matrix:
                            total_score += compatibility_matrix[pair]
            
            return total_score
        
        # Initialize population
        population = []
        for _ in range(population_size):
            # Random solution with budget constraint
            solution = [0] * n_services
            n_selected = random.randint(1, self.budget)
            selected_indices = random.sample(range(n_services), n_selected)
            for idx in selected_indices:
                solution[idx] = 1
            population.append(solution)
        
        # Evolution loop
        best_solution = None
        best_fitness = -float('inf')
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for solution in population:
                score = fitness(solution)
                fitness_scores.append(score)
                
                if score > best_fitness:
                    best_fitness = score
                    best_solution = solution.copy()
            
            # Selection (tournament)
            new_population = []
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 3
                tournament = random.sample(list(zip(population, fitness_scores)), 
                                         tournament_size)
                winner = max(tournament, key=lambda x: x[1])[0]
                new_population.append(winner.copy())
            
            # Crossover and mutation
            population = []
            for i in range(0, len(new_population), 2):
                if i + 1 < len(new_population):
                    parent1 = new_population[i]
                    parent2 = new_population[i+1]
                    
                    # Single-point crossover
                    crossover_point = random.randint(1, n_services - 1)
                    child1 = parent1[:crossover_point] + parent2[crossover_point:]
                    child2 = parent2[:crossover_point] + parent1[crossover_point:]
                    
                    # Mutation
                    for child in [child1, child2]:
                        if random.random() < 0.1:  # Mutation rate
                            # Flip a random bit
                            idx = random.randint(0, n_services - 1)
                            child[idx] = 1 - child[idx]
                            
                            # Ensure budget constraint
                            if sum(child) > self.budget:
                                # Randomly deselect some services
                                selected_indices = [j for j, val in enumerate(child) if val == 1]
                                to_deselect = random.sample(selected_indices, 
                                                          sum(child) - self.budget)
                                for j in to_deselect:
                                    child[j] = 0
                        
                        population.append(child)
                else:
                    population.append(new_population[i])
            
            # Ensure population size
            population = population[:population_size]
            
            if generation % 20 == 0:
                print(f"  Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Convert best solution to service IDs
        selected_services = []
        if best_solution:
            for i, selected in enumerate(best_solution):
                if selected:
                    selected_services.append(services[i])
        
        return selected_services
    
    def optimize_package(self, candidate_services: List[str],
                        individual_scores: Dict[str, float],
                        compatibility_scores: Dict[Tuple[str, str], float],
                        method: str = 'genetic') -> List[str]:
        """
        Optimize service package selection using QKP
        
        Args:
            candidate_services: List of candidate service IDs
            individual_scores: Dictionary of service -> individual score
            compatibility_scores: Dictionary of (service_i, service_j) -> compatibility score
            method: Optimization method ('greedy' or 'genetic')
        
        Returns: Optimized list of service IDs
        """
        if method == 'greedy':
            return self.solve_qkp_greedy(candidate_services, individual_scores, 
                                        compatibility_scores)
        else:
            return self.solve_qkp_genetic(candidate_services, individual_scores,
                                         compatibility_scores)

# ============================================================================
# Complete MLRN Framework
# ============================================================================

class MLRNFramework:
    """
    Complete Multi-Level Relational Network Framework
    
    Pipeline:
    1. Build multi-level network
    2. Perform relation-aware random walk
    3. Generate candidate service packages
    4. Optimize package selection using QKP
    5. Evaluate and recommend
    """
    
    def __init__(self, 
                 min_package_size: int = 2,
                 max_package_size: int = 5,
                 restart_prob: float = 0.2,
                 num_walks: int = 100,
                 budget: int = 5):
        """
        Initialize MLRN framework
        
        Args:
            min_package_size: Minimum services in package
            max_package_size: Maximum services in package
            restart_prob: Random walk restart probability
            num_walks: Number of random walks
            budget: QKP budget (maximum package size)
        """
        # Components
        self.network_builder = MultiLevelNetwork()
        self.random_walk = RelationAwareRandomWalk(
            restart_prob=restart_prob,
            num_walks=num_walks
        )
        self.package_generator = ServicePackageGenerator(
            min_package_size=min_package_size,
            max_package_size=max_package_size
        )
        self.optimizer = QuadraticKnapsackOptimizer(budget=budget)
        
        # Data storage
        self.services = []
        self.mashups = []
        self.service_dict = {}
        self.mashup_dict = {}
        
        # Network components
        self.network_components = None
        self.unified_network = None
        
    def load_data(self, services_data: List[Dict], mashups_data: List[Dict]):
        """Load services and mashups data"""
        print("Loading data for MLRN framework...")
        
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
                tags=mashup_data.get('tags', []),
                used_services=mashup_data.get('used_services', []),
                created_time=mashup_data.get('created_time', '')
            )
            self.mashups.append(mashup)
            self.mashup_dict[mashup.mashup_id] = mashup
        
        print(f"Loaded {len(self.services)} services and {len(self.mashups)} mashups")
    
    def build_network(self):
        """Build multi-level relational network"""
        print("\n" + "="*60)
        print("BUILDING MULTI-LEVEL RELATIONAL NETWORK")
        print("="*60)
        
        self.network_components = self.network_builder.build_complete_network(
            self.services, self.mashups
        )
        
        self.unified_network = self.network_components['unified_network']
        
        print("\nNetwork statistics:")
        print(f"  Service nodes: {self.network_builder.service_graph.number_of_nodes()}")
        print(f"  Tag nodes: {self.network_builder.tag_graph.number_of_nodes()}")
        print(f"  Mashup nodes: {self.network_builder.mashup_graph.number_of_nodes()}")
        print(f"  Total edges in unified network: {self.unified_network.number_of_edges()}")
    
    def recommend_service_package(self, query_tags: List[str],
                                 query_description: str = "",
                                 n_recommendations: int = 5,
                                 optimization_method: str = 'genetic') -> List[ServicePackage]:
        """
        Recommend service package for given query
        
        Args:
            query_tags: List of tags describing requirements
            query_description: Textual description of requirements
            n_recommendations: Number of packages to recommend
            optimization_method: 'greedy' or 'genetic'
        
        Returns: List of recommended service packages
        """
        print("\n" + "="*60)
        print("RECOMMENDING SERVICE PACKAGE")
        print("="*60)
        print(f"Query tags: {query_tags}")
        print(f"Query description: {query_description[:100]}...")
        
        # Step 1: Find relevant start nodes for random walk
        print("\nStep 1: Finding relevant start nodes...")
        start_nodes = self.find_relevant_start_nodes(query_tags, query_description)
        print(f"  Selected {len(start_nodes)} start nodes")
        
        # Step 2: Perform relation-aware random walk
        print("\nStep 2: Performing relation-aware random walk...")
        service_relevance_scores = self.random_walk.get_service_relevance_scores(
            start_nodes, self.unified_network
        )
        
        # Get top candidate services
        candidate_services = self.select_candidate_services(service_relevance_scores, 
                                                          top_k=50)
        print(f"  Selected {len(candidate_services)} candidate services")
        
        # Step 3: Prepare scores for QKP
        print("\nStep 3: Preparing scores for optimization...")
        
        # Individual scores (coverage/relevance)
        individual_scores = {}
        for service_id in candidate_services:
            individual_scores[service_id] = service_relevance_scores.get(service_id, 0)
        
        # Compatibility scores
        compatibility_matrix = self.compute_compatibility_matrix(candidate_services)
        
        # Step 4: Optimize package selection using QKP
        print("\nStep 4: Optimizing package selection...")
        
        optimized_packages = []
        
        for package_idx in range(n_recommendations):
            print(f"\n  Optimizing package {package_idx + 1}...")
            
            # Run optimization
            selected_services = self.optimizer.optimize_package(
                candidate_services,
                individual_scores,
                compatibility_matrix,
                method=optimization_method
            )
            
            if selected_services:
                # Score the optimized package
                package = self.package_generator.score_package(
                    selected_services,
                    service_relevance_scores,
                    self.network_builder.service_graph,
                    self.services
                )
                
                optimized_packages.append(package)
                
                print(f"    Selected {len(selected_services)} services")
                print(f"    Package score: {package.score:.4f}")
                
                # Reduce scores of selected services for diversity
                for service_id in selected_services:
                    if service_id in individual_scores:
                        individual_scores[service_id] *= 0.5  # Reduce by half
        
        # Sort packages by score
        optimized_packages.sort(key=lambda p: p.score, reverse=True)
        
        # Display recommendations
        print("\n" + "="*60)
        print(f"TOP {len(optimized_packages)} RECOMMENDED SERVICE PACKAGES")
        print("="*60)
        
        for i, package in enumerate(optimized_packages, 1):
            service_names = []
            for service_id in package.services:
                if service_id in self.service_dict:
                    service_names.append(self.service_dict[service_id].name)
                else:
                    service_names.append(service_id)
            
            print(f"\n{i}. Overall Score: {package.score:.4f}")
            print(f"   Services: {', '.join(service_names)}")
            print(f"   Coverage: {package.coverage_score:.4f}, "
                  f"Compatibility: {package.compatibility_score:.4f}, "
                  f"Diversity: {package.diversity_score:.4f}, "
                  f"Popularity: {package.popularity_score:.4f}")
        
        return optimized_packages
    
    def find_relevant_start_nodes(self, query_tags: List[str], 
                                 query_description: str) -> List[str]:
        """
        Find relevant nodes in the network to start random walk from
        """
        start_nodes = []
        
        # 1. Find matching tags
        for tag_name in query_tags:
            normalized_tag = tag_name.lower().strip()
            
            # Look for service tags
            service_tag_id = f"tag_{normalized_tag}"
            if service_tag_id in self.unified_network:
                start_nodes.append(service_tag_id)
            
            # Look for mashup tags
            mashup_tag_id = f"mashup_tag_{normalized_tag}"
            if mashup_tag_id in self.unified_network:
                start_nodes.append(mashup_tag_id)
        
        # 2. Find services with similar tags
        for service in self.services:
            service_tags_lower = [tag.lower().strip() for tag in service.tags]
            query_tags_lower = [tag.lower().strip() for tag in query_tags]
            
            # Check for tag overlap
            overlap = set(service_tags_lower).intersection(set(query_tags_lower))
            if overlap:
                start_nodes.append(service.service_id)
        
        # 3. If no matches found, use most popular services
        if not start_nodes:
            print("  No direct tag matches found. Using popular services as start nodes.")
            
            # Sort services by popularity
            sorted_services = sorted(self.services, 
                                   key=lambda s: s.popularity, 
                                   reverse=True)
            
            for service in sorted_services[:10]:
                start_nodes.append(service.service_id)
        
        # Remove duplicates
        start_nodes = list(set(start_nodes))
        
        return start_nodes
    
    def select_candidate_services(self, relevance_scores: Dict[str, float],
                                 top_k: int = 50) -> List[str]:
        """Select top candidate services based on relevance scores"""
        sorted_services = sorted(relevance_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        candidate_services = [service_id for service_id, score in sorted_services[:top_k]]
        
        # Ensure we have enough candidates
        if len(candidate_services) < self.package_generator.min_package_size:
            # Add more services by popularity
            remaining_slots = top_k - len(candidate_services)
            popular_services = sorted(self.services, 
                                    key=lambda s: s.popularity, 
                                    reverse=True)
            
            for service in popular_services:
                if service.service_id not in candidate_services:
                    candidate_services.append(service.service_id)
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
        
        return candidate_services
    
    def compute_compatibility_matrix(self, candidate_services: List[str]) -> Dict[Tuple[str, str], float]:
        """Compute compatibility matrix for candidate services"""
        compatibility_matrix = {}
        
        service_graph = self.network_builder.service_graph
        
        for i in range(len(candidate_services)):
            for j in range(i+1, len(candidate_services)):
                service_i = candidate_services[i]
                service_j = candidate_services[j]
                
                pair = tuple(sorted([service_i, service_j]))
                
                if service_graph.has_edge(service_i, service_j):
                    weight = service_graph[service_i][service_j].get('weight', 0)
                    compatibility_matrix[pair] = weight
                else:
                    compatibility_matrix[pair] = 0
        
        return compatibility_matrix
    
    def evaluate_recommendation(self, target_mashup_id: str,
                               recommended_packages: List[ServicePackage]) -> Dict[str, float]:
        """
        Evaluate recommendation quality for a given mashup
        
        Returns precision, recall, and F1-score
        """
        if target_mashup_id not in self.mashup_dict:
            return {}
        
        target_mashup = self.mashup_dict[target_mashup_id]
        actual_services = set(target_mashup.used_services)
        
        evaluation_results = {}
        
        # For each recommended package
        for i, package in enumerate(recommended_packages, 1):
            recommended_set = set(package.services)
            
            if actual_services:
                tp = len(actual_services.intersection(recommended_set))
                precision = tp / len(recommended_set) if recommended_set else 0
                recall = tp / len(actual_services) if actual_services else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                
                evaluation_results[f'package_{i}_precision'] = precision
                evaluation_results[f'package_{i}_recall'] = recall
                evaluation_results[f'package_{i}_f1'] = f1
        
        return evaluation_results

# ============================================================================
# Data Generation and Demonstration
# ============================================================================

def generate_mlrn_sample_data(n_services: int = 200, n_mashups: int = 150) -> Tuple[List[Dict], List[Dict]]:
    """Generate sample data for MLRN framework"""
    print("Generating sample data for MLRN framework...")
    
    # Service categories and common tags
    categories = ["Mapping", "Social", "Payment", "Travel", "Shopping", 
                  "Video", "Audio", "Weather", "News", "Business"]
    
    service_tags_by_category = {
        "Mapping": ["maps", "location", "geolocation", "navigation", "places", "directions"],
        "Social": ["social", "media", "networking", "sharing", "community", "friends"],
        "Payment": ["payment", "transaction", "money", "banking", "finance", "credit"],
        "Travel": ["travel", "hotels", "flights", "booking", "tourism", "vacation"],
        "Video": ["video", "streaming", "media", "playback", "encoding", "youtube"],
        "Audio": ["audio", "music", "sound", "playlist", "streaming", "spotify"],
        "Weather": ["weather", "forecast", "temperature", "climate", "rain", "sun"],
        "News": ["news", "articles", "headlines", "rss", "media", "updates"],
        "Business": ["business", "analytics", "reports", "data", "insights", "metrics"]
    }
    
    # Generate services
    services_data = []
    for i in range(n_services):
        category = random.choice(categories)
        base_tags = service_tags_by_category.get(category, [category.lower()])
        
        # Add some random tags
        all_tags = base_tags + random.sample([
            "api", "web", "service", "rest", "cloud", "integration",
            "developer", "platform", "application", "interface"
        ], 3)
        
        services_data.append({
            'id': f'service_{i:03d}',
            'name': f'{category} Service {i}',
            'description': f'A comprehensive {category.lower()} API with advanced features and easy integration.',
            'tags': all_tags,
            'category': category,
            'provider': random.choice(['Google', 'Amazon', 'Microsoft', 'IBM', 'Twitter']),
            'popularity': random.randint(1, 1000)
        })
    
    # Generate mashups with realistic patterns
    mashups_data = []
    mashup_tag_pool = [
        "social", "maps", "travel", "shopping", "video", "music",
        "weather", "news", "business", "analytics", "mobile", "web",
        "application", "integration", "platform", "dashboard"
    ]
    
    # Common mashup patterns
    mashup_patterns = [
        (["social", "maps"], ["Social", "Mapping"]),      # Social mapping
        (["travel", "payment"], ["Travel", "Payment"]),   # Travel booking
        (["video", "social"], ["Video", "Social"]),       # Social video
        (["weather", "maps"], ["Weather", "Mapping"]),    # Weather maps
        (["news", "social"], ["News", "Social"]),         # Social news
        (["business", "analytics"], ["Business", "Social"])  # Business analytics
    ]
    
    for i in range(n_mashups):
        # Choose a pattern or random combination
        if random.random() < 0.6 and mashup_patterns:
            pattern_tags, pattern_categories = random.choice(mashup_patterns)
            
            # Select services from these categories
            selected_services = []
            for category in pattern_categories:
                category_services = [s for s in services_data if s['category'] == category]
                if category_services:
                    n_select = random.randint(1, 2)
                    selected = random.sample(category_services, min(n_select, len(category_services)))
                    selected_services.extend([s['id'] for s in selected])
            
            # Generate mashup tags
            mashup_tags = pattern_tags + random.sample(mashup_tag_pool, 2)
        else:
            # Random selection
            n_services_in_mashup = random.randint(2, 4)
            selected_services = random.sample([s['id'] for s in services_data], n_services_in_mashup)
            
            # Get tags from selected services
            service_tags = []
            for service_id in selected_services:
                service = next((s for s in services_data if s['id'] == service_id), None)
                if service:
                    service_tags.extend(service['tags'][:2])  # Take first 2 tags
            
            mashup_tags = list(set(service_tags))[:4]  # Unique tags, max 4
        
        # Remove duplicates
        selected_services = list(set(selected_services))
        mashup_tags = list(set(mashup_tags))
        
        mashups_data.append({
            'id': f'mashup_{i:03d}',
            'name': f'Mashup Application {i}',
            'description': f'Integrated application combining multiple services for enhanced functionality.',
            'tags': mashup_tags,
            'used_services': selected_services,
            'created_time': f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}'
        })
    
    return services_data, mashups_data

def demonstrate_mlrn_framework():
    """Demonstrate the complete MLRN framework"""
    print("="*80)
    print("MULTI-LEVEL RELATIONAL NETWORK (MLRN) FRAMEWORK DEMONSTRATION")
    print("Paper: 'Service package recommendation for mashup development based on a multi-level relational network'")
    print("="*80)
    
    # Generate sample data
    services_data, mashups_data = generate_mlrn_sample_data(n_services=200, n_mashups=150)
    
    # Initialize MLRN framework
    mlrn = MLRNFramework(
        min_package_size=2,
        max_package_size=5,
        restart_prob=0.2,
        num_walks=100,
        budget=4
    )
    
    # Load data
    mlrn.load_data(services_data, mashups_data)
    
    # Build network
    mlrn.build_network()
    
    # Test with sample query
    query_tags = ["social", "maps", "location"]
    query_description = "I want to create a location-based social application that shows friends on a map and allows social interactions based on location."
    
    print(f"\nQuery: {query_description}")
    print(f"Tags: {query_tags}")
    
    # Generate recommendations
    recommendations = mlrn.recommend_service_package(
        query_tags=query_tags,
        query_description=query_description,
        n_recommendations=3,
        optimization_method='genetic'
    )
    
    # Test evaluation with an existing mashup
    print("\n" + "="*60)
    print("EVALUATION ON EXISTING MASHUP")
    print("="*60)
    
    # Find a mashup that uses social and mapping services
    test_mashup = None
    for mashup in mashups_data:
        mashup_tags_lower = [tag.lower() for tag in mashup['tags']]
        if 'social' in mashup_tags_lower and 'maps' in mashup_tags_lower:
            test_mashup = mashup
            break
    
    if test_mashup:
        print(f"Testing with mashup: {test_mashup['name']}")
        print(f"Actual services: {test_mashup['used_services']}")
        
        # Create query based on mashup tags
        test_query_tags = test_mashup['tags'][:3]  # First 3 tags
        
        # Generate recommendations for this query
        test_recommendations = mlrn.recommend_service_package(
            query_tags=test_query_tags,
            query_description=test_mashup['description'],
            n_recommendations=2,
            optimization_method='greedy'
        )
        
        # Evaluate
        evaluation = mlrn.evaluate_recommendation(test_mashup['id'], test_recommendations)
        
        print("\nEvaluation results:")
        for metric, value in evaluation.items():
            print(f"{metric}: {value:.4f}")
    
    # Demonstrate cold-start scenario
    print("\n" + "="*60)
    print("COLD-START SCENARIO: New Domain")
    print("="*60)
    
    new_query_tags = ["iot", "analytics", "dashboard"]
    new_description = "Create an IoT analytics dashboard to monitor sensor data and generate insights."
    
    print(f"\nNew domain query: {new_description}")
    print(f"Tags: {new_query_tags}")
    
    # Since these tags might not exist in our network, framework should handle it
    new_recommendations = mlrn.recommend_service_package(
        query_tags=new_query_tags,
        query_description=new_description,
        n_recommendations=2,
        optimization_method='genetic'
    )
    
    return mlrn, recommendations

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run demonstration
    mlrn_framework, recommendations = demonstrate_mlrn_framework()
    
    print("\n" + "="*80)
    print("MLRN FRAMEWORK IMPLEMENTATION COMPLETE")
    print("="*80)
    
    # Display framework statistics
    print(f"\nFramework Statistics:")
    print(f"- Services: {len(mlrn_framework.services)}")
    print(f"- Mashups: {len(mlrn_framework.mashups)}")
    print(f"- Tags in network: {mlrn_framework.network_builder.tag_graph.number_of_nodes()}")
    print(f"- Service-service edges: {mlrn_framework.network_builder.service_graph.number_of_edges()}")
    print(f"- Tag-tag edges: {mlrn_framework.network_builder.tag_graph.number_of_edges()}")
    print(f"- Unified network size: {mlrn_framework.unified_network.number_of_nodes()} nodes, "
          f"{mlrn_framework.unified_network.number_of_edges()} edges")