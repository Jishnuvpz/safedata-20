import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from scipy import stats
import tensorflow as tf
import tensorflow_privacy as tfp

logger = logging.getLogger(__name__)

class DifferentialPrivacyEngine:
    """Differential Privacy implementation for SafeData 2.0"""
    
    def __init__(self):
        self.privacy_spent = {}  # Track privacy budget usage
        
    async def apply_dp_noise(
        self,
        data: pd.DataFrame,
        epsilon: float,
        delta: float,
        mechanism: str = "gaussian",
        sensitivity: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply differential privacy noise to dataset
        
        Args:
            data: Input dataframe
            epsilon: Privacy budget parameter
            delta: Privacy parameter for (epsilon, delta)-DP
            mechanism: DP mechanism to use ('gaussian', 'laplace', 'exponential')
            sensitivity: Global sensitivity (auto-calculated if None)
            
        Returns:
            DataFrame with DP noise applied
        """
        logger.info(f"Applying {mechanism} DP mechanism with ε={epsilon}, δ={delta}")
        
        try:
            noisy_data = data.copy()
            
            # Calculate sensitivity if not provided
            if sensitivity is None:
                sensitivity = await self._calculate_global_sensitivity(data)
            
            # Apply DP noise based on mechanism
            if mechanism.lower() == "gaussian":
                noisy_data = await self._apply_gaussian_mechanism(noisy_data, epsilon, delta, sensitivity)
            elif mechanism.lower() == "laplace":
                noisy_data = await self._apply_laplace_mechanism(noisy_data, epsilon, sensitivity)
            elif mechanism.lower() == "exponential":
                noisy_data = await self._apply_exponential_mechanism(noisy_data, epsilon, sensitivity)
            else:
                raise ValueError(f"Unknown DP mechanism: {mechanism}")
            
            # Track privacy budget usage
            await self._track_privacy_budget(epsilon, delta)
            
            return noisy_data
            
        except Exception as e:
            logger.error(f"DP noise application failed: {str(e)}")
            raise e
    
    async def _calculate_global_sensitivity(self, data: pd.DataFrame) -> float:
        """Calculate global sensitivity of the dataset"""
        sensitivities = []
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # For numeric columns, sensitivity is the range
                col_range = data[col].max() - data[col].min()
                sensitivities.append(col_range)
            else:
                # For categorical columns, sensitivity is 1 (binary encoding)
                sensitivities.append(1.0)
        
        # Global sensitivity is the maximum column sensitivity
        return max(sensitivities) if sensitivities else 1.0
    
    async def _apply_gaussian_mechanism(
        self,
        data: pd.DataFrame,
        epsilon: float,
        delta: float,
        sensitivity: float
    ) -> pd.DataFrame:
        """Apply Gaussian mechanism for (ε, δ)-differential privacy"""
        
        # Calculate noise scale
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        noisy_data = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Add Gaussian noise to numeric columns
                noise = np.random.normal(0, sigma, size=len(data))
                noisy_data[col] = data[col] + noise
            else:
                # For categorical data, apply randomized response
                noisy_data[col] = await self._randomized_response(data[col], epsilon)
        
        return noisy_data
    
    async def _apply_laplace_mechanism(
        self,
        data: pd.DataFrame,
        epsilon: float,
        sensitivity: float
    ) -> pd.DataFrame:
        """Apply Laplace mechanism for ε-differential privacy"""
        
        # Calculate noise scale
        scale = sensitivity / epsilon
        
        noisy_data = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Add Laplace noise to numeric columns
                noise = np.random.laplace(0, scale, size=len(data))
                noisy_data[col] = data[col] + noise
            else:
                # For categorical data, apply randomized response
                noisy_data[col] = await self._randomized_response(data[col], epsilon)
        
        return noisy_data
    
    async def _apply_exponential_mechanism(
        self,
        data: pd.DataFrame,
        epsilon: float,
        sensitivity: float
    ) -> pd.DataFrame:
        """Apply Exponential mechanism for categorical data"""
        
        noisy_data = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # For numeric data, apply discretization then exponential mechanism
                discretized = pd.cut(data[col], bins=10, labels=False)
                noisy_data[col] = await self._exponential_mechanism_categorical(
                    discretized, epsilon, sensitivity
                )
            else:
                # Apply exponential mechanism to categorical data
                noisy_data[col] = await self._exponential_mechanism_categorical(
                    data[col], epsilon, sensitivity
                )
        
        return noisy_data
    
    async def _randomized_response(self, column: pd.Series, epsilon: float) -> pd.Series:
        """Apply randomized response to categorical data"""
        
        unique_values = column.unique()
        k = len(unique_values)
        
        if k <= 1:
            return column.copy()
        
        # Probability of keeping true value
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        
        # Probability of choosing any other value
        q = 1 / (np.exp(epsilon) + k - 1)
        
        noisy_column = column.copy()
        
        for i in range(len(column)):
            if np.random.random() < p:
                # Keep true value
                continue
            else:
                # Choose random alternative value
                other_values = [v for v in unique_values if v != column.iloc[i]]
                if other_values:
                    noisy_column.iloc[i] = np.random.choice(other_values)
        
        return noisy_column
    
    async def _exponential_mechanism_categorical(
        self,
        column: pd.Series,
        epsilon: float,
        sensitivity: float
    ) -> pd.Series:
        """Apply exponential mechanism to categorical column"""
        
        unique_values = column.unique()
        value_counts = column.value_counts()
        
        noisy_column = column.copy()
        
        for i in range(len(column)):
            # Calculate utility scores (frequency-based)
            utilities = [value_counts.get(val, 0) for val in unique_values]
            
            # Apply exponential mechanism
            weights = np.exp(epsilon * np.array(utilities) / (2 * sensitivity))
            probabilities = weights / np.sum(weights)
            
            # Sample new value
            noisy_column.iloc[i] = np.random.choice(unique_values, p=probabilities)
        
        return noisy_column
    
    async def _track_privacy_budget(self, epsilon: float, delta: float):
        """Track privacy budget usage"""
        
        session_id = "default"  # In practice, this would be session-specific
        
        if session_id not in self.privacy_spent:
            self.privacy_spent[session_id] = {"epsilon": 0.0, "delta": 0.0}
        
        # Accumulate privacy budget (composition)
        self.privacy_spent[session_id]["epsilon"] += epsilon
        self.privacy_spent[session_id]["delta"] += delta
        
        logger.info(f"Privacy budget used: ε={self.privacy_spent[session_id]['epsilon']:.3f}, "
                   f"δ={self.privacy_spent[session_id]['delta']:.6f}")
    
    async def get_privacy_budget_remaining(self, session_id: str = "default") -> Dict[str, float]:
        """Get remaining privacy budget for a session"""
        
        from app.core.config import settings
        
        spent = self.privacy_spent.get(session_id, {"epsilon": 0.0, "delta": 0.0})
        
        return {
            "epsilon_remaining": max(0, settings.max_epsilon - spent["epsilon"]),
            "delta_remaining": max(0, 1e-3 - spent["delta"]),  # Reasonable delta limit
            "epsilon_used": spent["epsilon"],
            "delta_used": spent["delta"]
        }
    
    async def apply_dp_sgd_training(
        self,
        model: tf.keras.Model,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        epsilon: float,
        delta: float,
        **kwargs
    ) -> tf.keras.Model:
        """
        Apply Differentially Private Stochastic Gradient Descent
        
        Args:
            model: TensorFlow model to train
            train_data: Training data
            train_labels: Training labels
            epsilon: Privacy budget
            delta: Privacy parameter
            
        Returns:
            Trained model with DP guarantees
        """
        
        logger.info(f"Starting DP-SGD training with ε={epsilon}, δ={delta}")
        
        # Training parameters
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 10)
        learning_rate = kwargs.get('learning_rate', 0.01)
        
        # Calculate noise multiplier
        noise_multiplier = tfp.compute_noise_from_budget_lib.compute_noise(
            n=len(train_data),
            batch_size=batch_size,
            target_epsilon=epsilon,
            epochs=epochs,
            delta=delta,
            noise_lbd=0.1
        )
        
        # Create DP optimizer
        optimizer = tfp.DPKerasSGDOptimizer(
            l2_norm_clip=1.0,
            noise_multiplier=noise_multiplier,
            num_microbatches=batch_size,
            learning_rate=learning_rate
        )
        
        # Compile model with DP optimizer
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with DP guarantees
        history = model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Track privacy budget
        await self._track_privacy_budget(epsilon, delta)
        
        return model
    
    async def calculate_privacy_loss(
        self,
        mechanism: str,
        epsilon: float,
        delta: float,
        queries: int = 1
    ) -> Dict[str, float]:
        """Calculate actual privacy loss for given parameters"""
        
        if mechanism.lower() == "gaussian":
            # Advanced composition for Gaussian mechanism
            composed_epsilon = epsilon * np.sqrt(2 * queries * np.log(1/delta))
            composed_delta = delta * queries
        elif mechanism.lower() == "laplace":
            # Basic composition for Laplace mechanism
            composed_epsilon = epsilon * queries
            composed_delta = 0.0
        else:
            # Conservative estimate
            composed_epsilon = epsilon * queries
            composed_delta = delta * queries
        
        return {
            "epsilon_loss": composed_epsilon,
            "delta_loss": composed_delta,
            "queries": queries,
            "mechanism": mechanism
        }
