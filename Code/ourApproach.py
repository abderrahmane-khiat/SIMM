"""
Complete Implementation of: 
"Sustainable Intelligent Media Mashups: A Knowledge-Centric and History-Aware Approach"


This is a single-file implementation of the history-aware semantic framework
for generating reliable media-intelligent mashups.
"""

import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Any, Optional
import itertools
from dataclasses import dataclass, field
import pickle

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class API:
    """Represents a Web/IoT API with semantic and historical properties"""
    id: str
    name: str
    description: str
    category: str = ""
    functions: List[str] = field(default_factory=list)
    popularity: float = 0.0
    # Cached values
    similarity_values: Dict[str, float] = field(default_factory=dict)  # {other_api_id: similarity}
    cooccurrence_values: Dict[str, float] = field(default_factory=dict)  # {other_api_id: cooccurrence}
    embedding: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

@dataclass
class Mashup:
    """Represents a composition of APIs"""
    apis: List[API]
    id: str = ""
    reliability_score: float = 0.0
    historical_usage_count: int = 0
    average_rating: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            # Generate ID based on sorted API IDs
            api_ids = sorted([api.id for api in self.apis])
            self.id = "_".join(api_ids)
    
    def get_api_ids(self) -> Set[str]:
        return {api.id for api in self.apis}
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

# ============================================================================
# SEMANTIC SIMILARITY MODULE
# ============================================================================

class SemanticSimilarityCalculator:
    """Calculates semantic similarity between APIs using embeddings"""
    
    def __init__(self, model_type='tfidf'):
        """
        Args:
            model_type: 'tfidf', 'sbert', or 'custom'
        """
        self.model_type = model_type
        self.embeddings = {}  # api_id -> embedding
        self.vocabulary = {}
        
        if model_type == 'sbert':
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_sbert = True
            except ImportError:
                print("Warning: sentence-transformers not installed. Using TF-IDF instead.")
                self.model_type = 'tfidf'
                self.use_sbert = False
        else:
            self.use_sbert = False
    
    def compute_embeddings(self, apis: Dict[str, API]) -> None:
        """Compute embeddings for all APIs"""
        if self.model_type == 'sbert' and self.use_sbert:
            texts = [f"{api.name} {api.description} {' '.join(api.functions)}" 
                    for api in apis.values()]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            for (api_id, api), embedding in zip(apis.items(), embeddings):
                api.embedding = embedding
                self.embeddings[api_id] = embedding
        else:
            # TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [f"{api.name} {api.description} {' '.join(api.functions)}" 
                    for api in apis.values()]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            self.vocabulary = vectorizer.vocabulary_
            
            # Convert to dense arrays for easier computation
            for (api_id, api), idx in zip(apis.items(), range(len(apis))):
                embedding = tfidf_matrix[idx].toarray().flatten()
                api.embedding = embedding
                self.embeddings[api_id] = embedding
    
    def calculate_similarity(self, api1: API, api2: API) -> float:
        """Calculate cosine similarity between two APIs"""
        if api1.id == api2.id:
            return 1.0
        
        # Check cache
        if api2.id in api1.similarity_values:
            return api1.similarity_values[api2.id]
        
        # Calculate similarity
        if api1.embedding is not None and api2.embedding is not None:
            # Cosine similarity
            dot_product = np.dot(api1.embedding, api2.embedding)
            norm1 = np.linalg.norm(api1.embedding)
            norm2 = np.linalg.norm(api2.embedding)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = 0.0
        else:
            # Fallback: Jaccard similarity on functions
            func1 = set(api1.functions)
            func2 = set(api2.functions)
            if len(func1.union(func2)) > 0:
                similarity = len(func1.intersection(func2)) / len(func1.union(func2))
            else:
                similarity = 0.0
        
        # Cache results
        api1.similarity_values[api2.id] = similarity
        api2.similarity_values[api1.id] = similarity
        
        return similarity

# ============================================================================
# HISTORICAL METRICS MODULE
# ============================================================================

class HistoricalMetricsCalculator:
    """Calculates co-occurrence and popularity metrics from historical data"""
    
    def __init__(self, historical_data: pd.DataFrame = None):
        """
        Args:
            historical_data: DataFrame with columns ['mashup_id', 'api_id', 'rating', 'timestamp']
        """
        self.historical_data = historical_data
        self.api_usage_stats = {}
        self.cooccurrence_matrix = defaultdict(Counter)
        self.popularity_scores = {}
        
        if historical_data is not None:
            self._preprocess_data()
    
    def _preprocess_data(self) -> None:
        """Preprocess historical data to compute metrics"""
        # Group by mashup
        mashup_groups = self.historical_data.groupby('mashup_id')
        
        # Calculate API usage statistics
        api_usage = self.historical_data['api_id'].value_counts()
        total_mashups = len(mashup_groups)
        
        for api_id, count in api_usage.items():
            usage_freq = count / total_mashups if total_mashups > 0 else 0
            
            # Get average rating for this API
            api_ratings = self.historical_data[self.historical_data['api_id'] == api_id]
            avg_rating = api_ratings['rating'].mean() if 'rating' in api_ratings.columns else 0.5
            
            # Normalize popularity (0-1)
            popularity = 0.6 * min(usage_freq * 10, 1.0) + 0.4 * (avg_rating / 5.0)
            self.popularity_scores[api_id] = min(1.0, popularity)
        
        # Calculate co-occurrence
        for mashup_id, group in mashup_groups:
            api_ids = group['api_id'].tolist()
            
            # Count all pairs
            for i in range(len(api_ids)):
                for j in range(i + 1, len(api_ids)):
                    self.cooccurrence_matrix[api_ids[i]][api_ids[j]] += 1
                    self.cooccurrence_matrix[api_ids[j]][api_ids[i]] += 1
        
        # Normalize co-occurrence
        for api1 in self.cooccurrence_matrix:
            total_pairs = sum(self.cooccurrence_matrix[api1].values())
            if total_pairs > 0:
                for api2 in self.cooccurrence_matrix[api1]:
                    self.cooccurrence_matrix[api1][api2] /= total_pairs
    
    def get_cooccurrence(self, api1_id: str, api2_id: str) -> float:
        """Get normalized co-occurrence between two APIs"""
        if api1_id in self.cooccurrence_matrix and api2_id in self.cooccurrence_matrix[api1_id]:
            return self.cooccurrence_matrix[api1_id][api2_id]
        return 0.0
    
    def get_popularity(self, api_id: str) -> float:
        """Get popularity score for an API"""
        return self.popularity_scores.get(api_id, 0.0)
    
    def update_metrics(self, new_mashup: Mashup, rating: float = 4.0) -> None:
        """Update metrics with a new mashup (for online learning)"""
        api_ids = [api.id for api in new_mashup.apis]
        
        # Update popularity
        for api_id in api_ids:
            current_pop = self.popularity_scores.get(api_id, 0.0)
            # Simple moving average update
            self.popularity_scores[api_id] = 0.9 * current_pop + 0.1 * (rating / 5.0)
        
        # Update co-occurrence
        for i in range(len(api_ids)):
            for j in range(i + 1, len(api_ids)):
                api1, api2 = api_ids[i], api_ids[j]
                
                # Increment counts
                self.cooccurrence_matrix[api1][api2] = self.cooccurrence_matrix[api1].get(api2, 0) + 1
                self.cooccurrence_matrix[api2][api1] = self.cooccurrence_matrix[api2].get(api1, 0) + 1
                
                # Normalize
                total1 = sum(self.cooccurrence_matrix[api1].values())
                total2 = sum(self.cooccurrence_matrix[api2].values())
                
                if total1 > 0:
                    for key in self.cooccurrence_matrix[api1]:
                        self.cooccurrence_matrix[api1][key] /= total1
                if total2 > 0:
                    for key in self.cooccurrence_matrix[api2]:
                        self.cooccurrence_matrix[api2][key] /= total2

# ============================================================================
# GENETIC ALGORITHM OPTIMIZATION
# ============================================================================

class GeneticAlgorithmOptimizer:
    """Genetic Algorithm for dynamic weight optimization"""
    
    def __init__(self, n_weights: int = 6, pop_size: int = 100, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.05):
        """
        Args:
            n_weights: Number of weights to optimize (wf, wc, wp, wfm, wcm, wpm)
        """
        self.n_weights = n_weights
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
    def initialize_population(self) -> List[np.ndarray]:
        """Initialize random population of weight vectors"""
        population = []
        for _ in range(self.pop_size):
            # Initialize weights between 0 and 1
            weights = np.random.rand(self.n_weights)
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            population.append(weights)
        return population
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Blend crossover (BLX-alpha)"""
        if random.random() < self.crossover_rate:
            alpha = 0.5
            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)
            
            for i in range(len(parent1)):
                d = abs(parent1[i] - parent2[i])
                lower = min(parent1[i], parent2[i]) - alpha * d
                upper = max(parent1[i], parent2[i]) + alpha * d
                
                child1[i] = random.uniform(lower, upper)
                child2[i] = random.uniform(lower, upper)
            
            # Normalize
            child1 = np.maximum(0, child1)
            child2 = np.maximum(0, child2)
            child1 = child1 / np.sum(child1) if np.sum(child1) > 0 else child1
            child2 = child2 / np.sum(child2) if np.sum(child2) > 0 else child2
            
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Add small Gaussian noise
                mutated[i] += random.gauss(0, 0.1)
                mutated[i] = max(0, min(1, mutated[i]))  # Clamp to [0, 1]
        
        # Renormalize
        if np.sum(mutated) > 0:
            mutated = mutated / np.sum(mutated)
        return mutated
    
    def select_parents(self, population: List[np.ndarray], 
                      fitness: List[float]) -> List[np.ndarray]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def evaluate_fitness(self, weights: np.ndarray, candidate_mashups: List[Mashup],
                        ground_truth: Set[str], reliability_calculator) -> float:
        """
        Evaluate fitness of weight vector using F-measure
        
        Args:
            weights: Weight vector [wf, wc, wp, wfm, wcm, wpm]
            candidate_mashups: List of candidate mashups
            ground_truth: Set of ground truth mashup IDs
            reliability_calculator: Function to calculate reliability
        """
        # Calculate reliability scores for all candidate mashups
        mashup_scores = []
        for mashup in candidate_mashups:
            score = reliability_calculator(mashup, weights)
            mashup_scores.append((mashup.id, score))
        
        # Select Top-K (K = min(10, len(ground_truth)))
        k = min(10, len(ground_truth))
        top_k_ids = [mid for mid, _ in sorted(mashup_scores, key=lambda x: x[1], reverse=True)[:k]]
        top_k_set = set(top_k_ids)
        
        # Calculate precision, recall, F-measure
        tp = len(top_k_set.intersection(ground_truth))
        fp = len(top_k_set - ground_truth)
        fn = len(ground_truth - top_k_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f_measure = 2 * precision * recall / (precision + recall)
        else:
            f_measure = 0
        
        return f_measure
    
    def optimize(self, candidate_mashups: List[Mashup], ground_truth: Set[str],
                reliability_calculator, max_generations: int = 50,
                fitness_threshold: float = 0.75) -> Tuple[np.ndarray, List[float]]:
        """
        Main optimization loop
        
        Returns:
            Tuple of (best_weights, fitness_history)
        """
        # Initialize population
        population = self.initialize_population()
        fitness_history = []
        best_fitness = 0
        best_weights = None
        stagnation_counter = 0
        
        for generation in range(max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for weights in population:
                fitness = self.evaluate_fitness(weights, candidate_mashups, 
                                               ground_truth, reliability_calculator)
                fitness_scores.append(fitness)
            
            # Update best solution
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_weights = population[current_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            fitness_history.append(best_fitness)
            
            # Check termination criteria
            if best_fitness >= fitness_threshold or stagnation_counter >= 20:
                break
            
            # Selection
            selected = self.select_parents(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i + 1]
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(self.mutate(child1))
                    new_population.append(self.mutate(child2))
                else:
                    new_population.append(self.mutate(selected[i]))
            
            # Elitism: keep best solution
            if best_weights is not None:
                new_population[0] = best_weights.copy()
            
            population = new_population
        
        return best_weights, fitness_history

# ============================================================================
# MAIN FRAMEWORK CLASS
# ============================================================================

class MediaIntelligentMashupFramework:
    """
    Main framework implementing the history-aware semantic approach
    for Top-K reliable mashup composition
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.apis: Dict[str, API] = {}
        self.mashup_history: Dict[str, Mashup] = {}
        self.semantic_calculator = SemanticSimilarityCalculator(
            model_type=self.config.get('similarity_model', 'tfidf')
        )
        self.historical_calculator = None
        self.ga_optimizer = GeneticAlgorithmOptimizer(
            pop_size=self.config.get('ga_pop_size', 100),
            crossover_rate=self.config.get('ga_crossover_rate', 0.8),
            mutation_rate=self.config.get('ga_mutation_rate', 0.05)
        )
        
        # Caches
        self.api_embeddings_computed = False
        self.candidate_cache = {}
        
    def load_apis_from_json(self, json_file: str) -> None:
        """Load APIs from JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for api_data in data.get('apis', []):
            api = API(
                id=api_data['id'],
                name=api_data.get('name', ''),
                description=api_data.get('description', ''),
                category=api_data.get('category', ''),
                functions=api_data.get('functions', []),
                popularity=api_data.get('popularity', 0.0)
            )
            self.apis[api.id] = api
        
        print(f"Loaded {len(self.apis)} APIs")
    
    def load_historical_data(self, historical_file: str) -> None:
        """Load historical mashup data"""
        try:
            df = pd.read_csv(historical_file)
            self.historical_calculator = HistoricalMetricsCalculator(df)
            
            # Also create Mashup objects from historical data
            mashup_groups = df.groupby('mashup_id')
            for mashup_id, group in mashup_groups:
                api_ids = group['api_id'].tolist()
                apis = [self.apis[api_id] for api_id in api_ids if api_id in self.apis]
                
                if len(apis) >= 2:  # At least 2 APIs for a mashup
                    mashup = Mashup(apis)
                    mashup.historical_usage_count = len(group)
                    mashup.average_rating = group['rating'].mean() if 'rating' in group.columns else 0.0
                    self.mashup_history[mashup.id] = mashup
            
            print(f"Loaded {len(self.mashup_history)} historical mashups")
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            self.historical_calculator = HistoricalMetricsCalculator()
    
    def compute_semantic_embeddings(self) -> None:
        """Compute semantic embeddings for all APIs"""
        if not self.api_embeddings_computed:
            self.semantic_calculator.compute_embeddings(self.apis)
            self.api_embeddings_computed = True
            print("Semantic embeddings computed")
    
    def _calculate_reliability(self, mashup: Mashup, weights: np.ndarray) -> float:
        """
        Calculate reliability score for a mashup using Equation 2 from paper
        
        Args:
            mashup: Mashup to evaluate
            weights: Weight vector [wf, wc, wp, wfm, wcm, wpm]
        """
        apis = mashup.apis
        n = len(apis)
        
        if n < 2:
            return 0.0
        
        # API-level metrics
        fsim_sum = 0.0
        coocc_sum = 0.0
        pop_sum = 0.0
        
        # Calculate pairwise metrics
        for i in range(n):
            api_i = apis[i]
            
            # Popularity
            if self.historical_calculator:
                pop_sum += self.historical_calculator.get_popularity(api_i.id)
            else:
                pop_sum += api_i.popularity
            
            for j in range(i + 1, n):
                api_j = apis[j]
                
                # Functional similarity
                fsim = self.semantic_calculator.calculate_similarity(api_i, api_j)
                fsim_sum += fsim
                
                # Co-occurrence
                if self.historical_calculator:
                    coocc = self.historical_calculator.get_cooccurrence(api_i.id, api_j.id)
                else:
                    # Default co-occurrence based on category similarity
                    coocc = 1.0 if api_i.category == api_j.category else 0.2
                coocc_sum += coocc
        
        # Averages
        num_pairs = n * (n - 1) / 2
        avg_fsim = fsim_sum / num_pairs if num_pairs > 0 else 0
        avg_coocc = coocc_sum / num_pairs if num_pairs > 0 else 0
        avg_pop = pop_sum / n if n > 0 else 0
        
        # Mashup-level metrics (simplified - use historical data if available)
        mfsim = 0.0
        mcoocc = 0.0
        mpop = 0.0
        
        if mashup.id in self.mashup_history:
            historical_mashup = self.mashup_history[mashup.id]
            mpop = historical_mashup.average_rating / 5.0  # Normalize to [0,1]
            mfsim = 0.8  # Assume high similarity for historically successful mashups
            mcoocc = 1.0  # Perfect co-occurrence for existing mashups
        
        # Apply weights (Equation 2)
        reliability = (
            weights[0] * avg_fsim +      # wf * FSim
            weights[1] * avg_coocc +     # wc * CoOcc
            weights[2] * avg_pop +       # wp * Pop
            weights[3] * mfsim +         # wfm * MFSim
            weights[4] * mcoocc +        # wcm * MCoOcc
            weights[5] * mpop            # wpm * MPop
        )
        
        return reliability
    
    def _find_relevant_apis(self, requirements: str, top_n: int = 20) -> List[API]:
        """
        Find APIs relevant to user requirements using semantic search
        
        Args:
            requirements: User requirements in natural language
            top_n: Number of relevant APIs to return
        """
        # Simple keyword-based filtering
        keywords = requirements.lower().split()
        relevant_apis = []
        
        for api in self.apis.values():
            score = 0
            api_text = f"{api.name} {api.description} {' '.join(api.functions)}".lower()
            
            # Count keyword matches
            for keyword in keywords:
                if keyword in api_text:
                    score += 1
            
            # Boost score if category matches
            for keyword in keywords:
                if keyword in api.category.lower():
                    score += 2
            
            if score > 0:
                relevant_apis.append((api, score))
        
        # Sort by score and return top N
        relevant_apis.sort(key=lambda x: x[1], reverse=True)
        return [api for api, _ in relevant_apis[:top_n]]
    
    def _generate_candidate_mashups(self, apis: List[API], 
                                   max_apis_per_mashup: int = 3,
                                   max_candidates: int = 1000) -> List[Mashup]:
        """
        Generate candidate mashup combinations from relevant APIs
        
        Args:
            apis: List of relevant APIs
            max_apis_per_mashup: Maximum APIs per mashup
            max_candidates: Maximum number of candidates to generate
        """
        if len(apis) < 2:
            return []
        
        candidate_mashups = []
        
        # Generate combinations of different sizes
        for r in range(2, min(max_apis_per_mashup + 1, len(apis) + 1)):
            combinations = list(itertools.combinations(apis, r))
            
            for combo in combinations:
                mashup = Mashup(list(combo))
                candidate_mashups.append(mashup)
                
                # Limit total candidates
                if len(candidate_mashups) >= max_candidates:
                    return candidate_mashups
        
        return candidate_mashups
    
    def _get_ground_truth(self, requirements: str) -> Set[str]:
        """
        Get ground truth mashups for the given requirements
        (In real implementation, this would come from user validation)
        
        Args:
            requirements: User requirements
        """
        ground_truth = set()
        
        # Simple implementation: use historical mashups that match keywords
        keywords = set(requirements.lower().split())
        
        for mashup_id, mashup in self.mashup_history.items():
            # Check if mashup description/name contains any keyword
            mashup_text = " ".join([api.name.lower() for api in mashup.apis])
            for keyword in keywords:
                if keyword in mashup_text:
                    ground_truth.add(mashup_id)
                    break
        
        # If no historical matches, create synthetic ground truth
        if not ground_truth:
            # Take top 5 historical mashups by rating
            sorted_mashups = sorted(self.mashup_history.values(), 
                                   key=lambda m: m.average_rating, 
                                   reverse=True)[:5]
            ground_truth = {m.id for m in sorted_mashups}
        
        return ground_truth
    
    def get_top_k_mashups(self, requirements: str, k: int = 10,
                         use_ga_optimization: bool = True) -> Tuple[List[Mashup], np.ndarray]:
        """
        Main method: Get Top-K reliable mashups for given requirements
        
        Args:
            requirements: User requirements in natural language
            k: Number of mashups to return (Top-K)
            use_ga_optimization: Whether to use GA for weight optimization
            
        Returns:
            Tuple of (list of top-k mashups, optimized weights)
        """
        print(f"Processing requirements: {requirements}")
        
        # Step 1: Ensure semantic embeddings are computed
        self.compute_semantic_embeddings()
        
        # Step 2: Find relevant APIs
        relevant_apis = self._find_relevant_apis(requirements)
        print(f"Found {len(relevant_apis)} relevant APIs")
        
        if len(relevant_apis) < 2:
            print("Not enough relevant APIs to create mashups")
            return [], np.zeros(6)
        
        # Step 3: Generate candidate mashups
        candidate_mashups = self._generate_candidate_mashups(relevant_apis)
        print(f"Generated {len(candidate_mashups)} candidate mashups")
        
        if not candidate_mashups:
            return [], np.zeros(6)
        
        # Step 4: Get ground truth for optimization
        ground_truth = self._get_ground_truth(requirements)
        print(f"Using {len(ground_truth)} ground truth mashups for optimization")
        
        # Step 5: Optimize weights
        if use_ga_optimization and len(ground_truth) >= 3:
            print("Optimizing weights with Genetic Algorithm...")
            
            best_weights, fitness_history = self.ga_optimizer.optimize(
                candidate_mashups,
                ground_truth,
                self._calculate_reliability,
                max_generations=self.config.get('max_generations', 50)
            )
            
            print(f"GA optimization completed. Best fitness: {max(fitness_history):.4f}")
            print(f"Optimized weights: {best_weights}")
        else:
            # Use equal weights as fallback
            best_weights = np.ones(6) / 6
            print("Using equal weights (no GA optimization)")
        
        # Step 6: Calculate final reliability scores with optimized weights
        mashup_scores = []
        for mashup in candidate_mashups:
            score = self._calculate_reliability(mashup, best_weights)
            mashup.reliability_score = score
            mashup_scores.append((mashup, score))
        
        # Step 7: Sort and return Top-K
        mashup_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_mashups = [mashup for mashup, _ in mashup_scores[:k]]
        
        print(f"Returning top {len(top_k_mashups)} mashups")
        return top_k_mashups, best_weights
    
    def evaluate_performance(self, test_requirements: List[str], 
                           test_ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
        """
        Evaluate framework performance on test data
        
        Args:
            test_requirements: List of test requirements
            test_ground_truth: Dict mapping requirement to set of ground truth mashup IDs
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_predictions = []
        all_ground_truth = []
        
        for req in test_requirements:
            # Get predictions
            top_mashups, _ = self.get_top_k_mashups(req, k=10)
            predicted_ids = {m.id for m in top_mashups}
            
            # Get ground truth
            true_ids = test_ground_truth.get(req, set())
            
            # Store for overall evaluation
            all_predictions.append(predicted_ids)
            all_ground_truth.append(true_ids)
        
        # Calculate overall metrics
        total_tp = total_fp = total_fn = 0
        
        for pred_set, true_set in zip(all_predictions, all_ground_truth):
            tp = len(pred_set.intersection(true_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate precision, recall, F-measure
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        if precision + recall > 0:
            f_measure = 2 * precision * recall / (precision + recall)
        else:
            f_measure = 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f_measure': f_measure,
            'total_predictions': len(all_predictions),
            'average_prediction_size': np.mean([len(p) for p in all_predictions])
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the framework state to a file"""
        state = {
            'apis': self.apis,
            'mashup_history': self.mashup_history,
            'config': self.config,
            'api_embeddings_computed': self.api_embeddings_computed
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the framework state from a file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.apis = state['apis']
        self.mashup_history = state['mashup_history']
        self.config = state.get('config', {})
        self.api_embeddings_computed = state.get('api_embeddings_computed', False)
        
        # Reinitialize components
        self.semantic_calculator = SemanticSimilarityCalculator(
            model_type=self.config.get('similarity_model', 'tfidf')
        )
        self.ga_optimizer = GeneticAlgorithmOptimizer(
            pop_size=self.config.get('ga_pop_size', 100)
        )
        
        print(f"Model loaded from {filepath}")
        print(f"Loaded {len(self.apis)} APIs and {len(self.mashup_history)} historical mashups")

# ============================================================================
# UTILITY FUNCTIONS AND DEMO
# ============================================================================

def create_sample_dataset(num_apis: int = 100, num_mashups: int = 500) -> Tuple[Dict, pd.DataFrame]:
    """Create sample dataset for demonstration"""
    # Create sample APIs
    categories = ['video', 'audio', 'social', 'maps', 'weather', 'news', 'iot', 'analytics']
    apis = []
    
    for i in range(num_apis):
        category = random.choice(categories)
        api = {
            'id': f'api_{i:03d}',
            'name': f'{category.capitalize()} API {i}',
            'description': f'A {category} API for processing {category} data',
            'category': category,
            'functions': [f'{category}_function_{j}' for j in range(1, 4)],
            'popularity': random.uniform(0, 1)
        }
        apis.append(api)
    
    # Create sample historical mashup data
    historical_records = []
    for mashup_id in range(num_mashups):
        # Randomly select 2-4 APIs
        num_apis_in_mashup = random.randint(2, 4)
        selected_api_indices = random.sample(range(num_apis), num_apis_in_mashup)
        
        for api_idx in selected_api_indices:
            record = {
                'mashup_id': f'mashup_{mashup_id:04d}',
                'api_id': f'api_{api_idx:03d}',
                'rating': random.uniform(3, 5),
                'timestamp': f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
            }
            historical_records.append(record)
    
    df = pd.DataFrame(historical_records)
    
    return {'apis': apis}, df

def demo():
    """Demonstrate the framework with sample data"""
    print("=" * 60)
    print("Media-Intelligent Mashup Composition Framework Demo")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n1. Creating sample dataset...")
    apis_data, historical_df = create_sample_dataset(num_apis=50, num_mashups=200)
    
    # Save sample data
    with open('sample_apis.json', 'w') as f:
        json.dump(apis_data, f, indent=2)
    
    historical_df.to_csv('sample_historical.csv', index=False)
    
    # Step 2: Initialize framework
    print("\n2. Initializing framework...")
    config = {
        'similarity_model': 'tfidf',
        'ga_pop_size': 50,
        'max_generations': 30
    }
    
    framework = MediaIntelligentMashupFramework(config)
    
    # Step 3: Load data
    print("\n3. Loading data...")
    framework.load_apis_from_json('sample_apis.json')
    framework.load_historical_data('sample_historical.csv')
    
    # Step 4: Test with sample requirements
    print("\n4. Testing with sample requirements...")
    test_requirements = [
        "video processing with audio transcription",
        "social media analytics with sentiment analysis",
        "weather data with map visualization"
    ]
    
    for i, req in enumerate(test_requirements, 1):
        print(f"\n--- Test {i}: '{req}' ---")
        top_mashups, weights = framework.get_top_k_mashups(req, k=5)
        
        if top_mashups:
            print(f"Top {len(top_mashups)} mashups:")
            for j, mashup in enumerate(top_mashups, 1):
                api_names = [api.name for api in mashup.apis]
                print(f"  {j}. {api_names} (score: {mashup.reliability_score:.3f})")
        else:
            print("No mashups generated.")
    
    # Step 5: Performance evaluation (simplified)
    print("\n5. Evaluating performance...")
    
    # Create test ground truth (simplified)
    test_ground_truth = {}
    for req in test_requirements:
        # Use first 3 historical mashups as ground truth
        test_ground_truth[req] = set(list(framework.mashup_history.keys())[:3])
    
    metrics = framework.evaluate_performance(test_requirements, test_ground_truth)
    
    print(f"Evaluation results:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F-measure: {metrics['f_measure']:.3f}")
    
    # Step 6: Save model
    print("\n6. Saving model...")
    framework.save_model('mashup_framework_model.pkl')
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Install required packages if not available
    required_packages = ['numpy', 'pandas', 'scikit-learn']
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Run demo
    demo()