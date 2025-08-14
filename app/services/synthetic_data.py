import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sdv.tabular import CTGAN, TVAE, CopulaGAN, GaussianCopula
    from sdv.evaluation import evaluate
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    logging.warning("SDV library not available. Using fallback synthetic data generation.")

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Synthetic Data Generation using advanced deep learning models"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        
    async def generate_synthetic_data(
        self,
        data: pd.DataFrame,
        method: str = "ctgan",
        num_samples: Optional[int] = None,
        epochs: int = 300,
        batch_size: int = 500,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data using specified method
        
        Args:
            data: Original dataset
            method: Generation method ('ctgan', 'tvae', 'copula', 'gaussian_copula')
            num_samples: Number of synthetic samples to generate
            epochs: Training epochs for deep learning models
            batch_size: Batch size for training
            
        Returns:
            Synthetic dataset
        """
        
        logger.info(f"Generating synthetic data using {method}")
        
        if num_samples is None:
            num_samples = len(data)
        
        try:
            if SDV_AVAILABLE:
                synthetic_data = await self._generate_with_sdv(
                    data, method, num_samples, epochs, batch_size, **kwargs
                )
            else:
                synthetic_data = await self._generate_fallback(
                    data, num_samples, **kwargs
                )
            
            # Validate synthetic data
            await self._validate_synthetic_data(data, synthetic_data)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {str(e)}")
            # Fallback to simple generation
            return await self._generate_fallback(data, num_samples, **kwargs)
    
    async def _generate_with_sdv(
        self,
        data: pd.DataFrame,
        method: str,
        num_samples: int,
        epochs: int,
        batch_size: int,
        **kwargs
    ) -> pd.DataFrame:
        """Generate synthetic data using SDV library"""
        
        # Prepare data
        prepared_data = await self._prepare_data_for_sdv(data)
        
        # Initialize model based on method
        if method.lower() == "ctgan":
            model = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                discriminator_steps=1,
                generator_steps=1,
                discriminator_decay=1e-6,
                generator_decay=1e-6,
                **kwargs
            )
        elif method.lower() == "tvae":
            model = TVAE(
                epochs=epochs,
                batch_size=batch_size,
                compress_dims=(128, 128),
                decompress_dims=(128, 128),
                **kwargs
            )
        elif method.lower() == "copula":
            model = CopulaGAN(
                epochs=epochs,
                batch_size=batch_size,
                **kwargs
            )
        elif method.lower() == "gaussian_copula":
            model = GaussianCopula(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Train model
        logger.info(f"Training {method} model...")
        model.fit(prepared_data)
        
        # Generate synthetic data
        logger.info(f"Generating {num_samples} synthetic samples...")
        synthetic_data = model.sample(num_samples)
        
        # Store model for future use
        model_id = f"{method}_{datetime.now().timestamp()}"
        self.models[model_id] = model
        self.model_metadata[model_id] = {
            "method": method,
            "original_shape": data.shape,
            "synthetic_shape": synthetic_data.shape,
            "training_time": datetime.now(),
            "parameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                **kwargs
            }
        }
        
        return synthetic_data
    
    async def _prepare_data_for_sdv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for SDV models"""
        
        prepared_data = data.copy()
        
        # Handle missing values
        for col in prepared_data.columns:
            if prepared_data[col].dtype == 'object':
                # Fill categorical missing values with mode
                mode_value = prepared_data[col].mode()
                if not mode_value.empty:
                    prepared_data[col].fillna(mode_value[0], inplace=True)
                else:
                    prepared_data[col].fillna('Unknown', inplace=True)
            else:
                # Fill numeric missing values with median
                prepared_data[col].fillna(prepared_data[col].median(), inplace=True)
        
        # Convert datetime columns to numeric
        for col in prepared_data.columns:
            if pd.api.types.is_datetime64_any_dtype(prepared_data[col]):
                prepared_data[col] = prepared_data[col].astype('int64') // 10**9
        
        # Ensure all categorical columns are string type
        for col in prepared_data.select_dtypes(include=['object']).columns:
            prepared_data[col] = prepared_data[col].astype(str)
        
        return prepared_data
    
    async def _generate_fallback(
        self,
        data: pd.DataFrame,
        num_samples: int,
        **kwargs
    ) -> pd.DataFrame:
        """Fallback synthetic data generation without SDV"""
        
        logger.info("Using fallback synthetic data generation")
        
        synthetic_data = pd.DataFrame()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Generate numeric data using normal distribution
                mean = data[col].mean()
                std = data[col].std()
                synthetic_values = np.random.normal(mean, std, num_samples)
                
                # Ensure values are within original range
                min_val, max_val = data[col].min(), data[col].max()
                synthetic_values = np.clip(synthetic_values, min_val, max_val)
                
                synthetic_data[col] = synthetic_values
                
            else:
                # Generate categorical data based on distribution
                value_counts = data[col].value_counts(normalize=True)
                synthetic_values = np.random.choice(
                    value_counts.index,
                    size=num_samples,
                    p=value_counts.values
                )
                synthetic_data[col] = synthetic_values
        
        return synthetic_data
    
    async def _validate_synthetic_data(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ):
        """Validate generated synthetic data"""
        
        # Check shape consistency
        if synthetic_data.shape[1] != original_data.shape[1]:
            raise ValueError("Synthetic data has different number of columns")
        
        # Check column names
        if list(synthetic_data.columns) != list(original_data.columns):
            raise ValueError("Synthetic data has different column names")
        
        # Check data types consistency
        for col in original_data.columns:
            orig_type = original_data[col].dtype
            synth_type = synthetic_data[col].dtype
            
            # Allow some flexibility in numeric types
            if pd.api.types.is_numeric_dtype(orig_type) and pd.api.types.is_numeric_dtype(synth_type):
                continue
            elif orig_type != synth_type:
                logger.warning(f"Data type mismatch for column {col}: {orig_type} vs {synth_type}")
        
        logger.info("Synthetic data validation passed")
    
    async def evaluate_synthetic_quality(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate quality of synthetic data"""
        
        quality_metrics = {}
        
        try:
            if SDV_AVAILABLE:
                # Use SDV evaluation
                evaluation_result = evaluate(synthetic_data, original_data)
                quality_metrics["sdv_score"] = evaluation_result
            
            # Custom quality metrics
            quality_metrics.update({
                "column_correlation": await self._evaluate_correlations(original_data, synthetic_data),
                "distribution_similarity": await self._evaluate_distributions(original_data, synthetic_data),
                "statistical_similarity": await self._evaluate_statistics(original_data, synthetic_data),
                "privacy_metrics": await self._evaluate_privacy_preservation(original_data, synthetic_data)
            })
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {str(e)}")
            quality_metrics["error"] = str(e)
        
        return quality_metrics
    
    async def _evaluate_correlations(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Evaluate correlation preservation"""
        
        # Select numeric columns
        orig_numeric = original_data.select_dtypes(include=[np.number])
        synth_numeric = synthetic_data.select_dtypes(include=[np.number])
        
        if orig_numeric.empty or synth_numeric.empty:
            return 1.0
        
        # Calculate correlation matrices
        orig_corr = orig_numeric.corr()
        synth_corr = synth_numeric.corr()
        
        # Calculate correlation between correlation matrices
        corr_values_orig = orig_corr.values[np.triu_indices_from(orig_corr.values, k=1)]
        corr_values_synth = synth_corr.values[np.triu_indices_from(synth_corr.values, k=1)]
        
        if len(corr_values_orig) == 0:
            return 1.0
        
        correlation = np.corrcoef(corr_values_orig, corr_values_synth)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    async def _evaluate_distributions(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Evaluate distribution similarity using Kolmogorov-Smirnov test"""
        
        from scipy import stats
        
        similarities = []
        
        for col in original_data.columns:
            if col in synthetic_data.columns:
                if pd.api.types.is_numeric_dtype(original_data[col]):
                    try:
                        ks_stat, _ = stats.ks_2samp(
                            original_data[col].dropna(),
                            synthetic_data[col].dropna()
                        )
                        similarity = 1 - ks_stat
                        similarities.append(similarity)
                    except:
                        similarities.append(0.5)
                else:
                    # For categorical data, compare distributions
                    orig_dist = original_data[col].value_counts(normalize=True)
                    synth_dist = synthetic_data[col].value_counts(normalize=True)
                    
                    # Calculate Jensen-Shannon divergence
                    js_div = await self._jensen_shannon_divergence(orig_dist, synth_dist)
                    similarity = 1 - js_div
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def _jensen_shannon_divergence(
        self,
        dist1: pd.Series,
        dist2: pd.Series
    ) -> float:
        """Calculate Jensen-Shannon divergence between two distributions"""
        
        # Align distributions
        all_values = set(dist1.index) | set(dist2.index)
        p = np.array([dist1.get(v, 0) for v in all_values])
        q = np.array([dist2.get(v, 0) for v in all_values])
        
        # Normalize
        p = p / np.sum(p) if np.sum(p) > 0 else p
        q = q / np.sum(q) if np.sum(q) > 0 else q
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Calculate JS divergence
        m = 0.5 * (p + q)
        js_div = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        
        return js_div
    
    async def _evaluate_statistics(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Evaluate statistical similarity"""
        
        similarities = []
        
        for col in original_data.columns:
            if col in synthetic_data.columns:
                if pd.api.types.is_numeric_dtype(original_data[col]):
                    # Compare mean and std
                    orig_mean, orig_std = original_data[col].mean(), original_data[col].std()
                    synth_mean, synth_std = synthetic_data[col].mean(), synthetic_data[col].std()
                    
                    mean_sim = 1 - abs(orig_mean - synth_mean) / (abs(orig_mean) + 1e-8)
                    std_sim = 1 - abs(orig_std - synth_std) / (abs(orig_std) + 1e-8)
                    
                    similarities.append((mean_sim + std_sim) / 2)
                else:
                    # Compare mode and entropy
                    orig_mode = original_data[col].mode()
                    synth_mode = synthetic_data[col].mode()
                    
                    mode_match = 1.0 if (not orig_mode.empty and not synth_mode.empty and 
                                       orig_mode.iloc[0] == synth_mode.iloc[0]) else 0.0
                    similarities.append(mode_match)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def _evaluate_privacy_preservation(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate privacy preservation of synthetic data"""
        
        # Distance to closest record (DCR)
        dcr_score = await self._calculate_dcr(original_data, synthetic_data)
        
        # Nearest neighbor distance ratio (NNDR)
        nndr_score = await self._calculate_nndr(original_data, synthetic_data)
        
        return {
            "distance_to_closest_record": dcr_score,
            "nearest_neighbor_distance_ratio": nndr_score,
            "privacy_score": (dcr_score + nndr_score) / 2
        }
    
    async def _calculate_dcr(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Calculate Distance to Closest Record"""
        
        # Use only numeric columns for distance calculation
        orig_numeric = original_data.select_dtypes(include=[np.number])
        synth_numeric = synthetic_data.select_dtypes(include=[np.number])
        
        if orig_numeric.empty or synth_numeric.empty:
            return 1.0
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(orig_numeric.fillna(0))
        synth_scaled = scaler.transform(synth_numeric.fillna(0))
        
        # Calculate minimum distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(synth_scaled, orig_scaled)
        min_distances = np.min(distances, axis=1)
        
        # Return normalized average minimum distance
        return np.mean(min_distances) / np.sqrt(orig_scaled.shape[1])
    
    async def _calculate_nndr(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame
    ) -> float:
        """Calculate Nearest Neighbor Distance Ratio"""
        
        # Use only numeric columns
        orig_numeric = original_data.select_dtypes(include=[np.number])
        synth_numeric = synthetic_data.select_dtypes(include=[np.number])
        
        if orig_numeric.empty or synth_numeric.empty:
            return 1.0
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        orig_scaled = scaler.fit_transform(orig_numeric.fillna(0))
        synth_scaled = scaler.transform(synth_numeric.fillna(0))
        
        # Calculate distances within each dataset
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Sample for efficiency
        sample_size = min(1000, len(synth_scaled))
        synth_sample = synth_scaled[:sample_size]
        
        # Distances to original data
        dist_to_orig = euclidean_distances(synth_sample, orig_scaled)
        nearest_orig = np.min(dist_to_orig, axis=1)
        
        # Distances within synthetic data
        dist_within_synth = euclidean_distances(synth_sample, synth_scaled)
        np.fill_diagonal(dist_within_synth[:sample_size, :sample_size], np.inf)
        nearest_synth = np.min(dist_within_synth, axis=1)
        
        # Calculate ratio
        ratios = nearest_orig / (nearest_synth + 1e-8)
        return np.mean(ratios)
