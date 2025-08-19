import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import hashlib
import json

from app.services.synthetic_data import SyntheticDataGenerator
from app.services.differential_privacy import DifferentialPrivacyEngine
from app.services.attack_simulation import AttackSimulator
from app.core.config import settings

logger = logging.getLogger(__name__)

class AnonymizationEngine:
    """Core anonymization engine implementing multiple privacy techniques"""
    
    def __init__(self):
        self.sdg = SyntheticDataGenerator()
        self.dp_engine = DifferentialPrivacyEngine()
        self.attack_simulator = AttackSimulator()
        
    async def anonymize_data(
        self,
        data: pd.DataFrame,
        method: str = "full",
        epsilon: float = 1.0,
        delta: float = 1e-5,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main anonymization function implementing the SafeData 2.0 pipeline
        
        Args:
            data: Original dataset
            method: Anonymization method (sdg, dp, sdc, sdg_dp, sdg_sdc, dp_sdc, full)
            epsilon: Privacy budget for differential privacy
            delta: Privacy parameter for differential privacy
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive attribute columns
            
        Returns:
            Dictionary containing anonymized data and metrics
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting anonymization with method: {method}")
            
            # Step 1: Data profiling and risk assessment
            data_profile = await self._profile_data(data, quasi_identifiers, sensitive_attributes)
            
            # Step 2: Apply anonymization based on method
            if method == "sdg":
                anonymized_data = await self._apply_synthetic_data_generation(data, **kwargs)
            elif method == "dp":
                anonymized_data = await self._apply_differential_privacy(data, epsilon, delta, **kwargs)
            elif method == "sdc":
                anonymized_data = await self._apply_statistical_disclosure_control(
                    data, quasi_identifiers, sensitive_attributes, **kwargs
                )
            elif method == "sdg_dp":
                # Combined SDG + DP
                synthetic_data = await self._apply_synthetic_data_generation(data, **kwargs)
                anonymized_data = await self._apply_differential_privacy(synthetic_data, epsilon, delta, **kwargs)
            elif method == "sdg_sdc":
                # Combined SDG + SDC
                synthetic_data = await self._apply_synthetic_data_generation(data, **kwargs)
                anonymized_data = await self._apply_statistical_disclosure_control(
                    synthetic_data, quasi_identifiers, sensitive_attributes, **kwargs
                )
            elif method == "dp_sdc":
                # Combined DP + SDC
                dp_data = await self._apply_differential_privacy(data, epsilon, delta, **kwargs)
                anonymized_data = await self._apply_statistical_disclosure_control(
                    dp_data, quasi_identifiers, sensitive_attributes, **kwargs
                )
            elif method == "full":
                # Full SafeData 2.0 pipeline: SDG + DP + SDC
                synthetic_data = await self._apply_synthetic_data_generation(data, **kwargs)
                dp_data = await self._apply_differential_privacy(synthetic_data, epsilon, delta, **kwargs)
                anonymized_data = await self._apply_statistical_disclosure_control(
                    dp_data, quasi_identifiers, sensitive_attributes, **kwargs
                )
            else:
                raise ValueError(f"Unknown anonymization method: {method}")
            
            # Step 3: Privacy and utility assessment
            privacy_metrics = await self._assess_privacy(data, anonymized_data, epsilon, delta)
            utility_metrics = await self._assess_utility(data, anonymized_data)
            
            # Step 4: Attack simulation
            attack_results = await self.attack_simulator.simulate_attacks(
                original_data=data,
                anonymized_data=anonymized_data,
                attack_types=["linkage", "membership", "attribute"]
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "anonymized_data": anonymized_data,
                "data_profile": data_profile,
                "privacy_metrics": privacy_metrics,
                "utility_metrics": utility_metrics,
                "attack_results": attack_results,
                "execution_time": execution_time,
                "method_used": method,
                "parameters": {
                    "epsilon": epsilon,
                    "delta": delta,
                    "quasi_identifiers": quasi_identifiers,
                    "sensitive_attributes": sensitive_attributes
                }
            }
            
        except Exception as e:
            logger.error(f"Anonymization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _profile_data(
        self,
        data: pd.DataFrame,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Profile the dataset to identify privacy risks"""
        
        profile = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "column_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "unique_values": {col: data[col].nunique() for col in data.columns},
            "quasi_identifiers": quasi_identifiers or [],
            "sensitive_attributes": sensitive_attributes or []
        }
        
        # Auto-detect potential quasi-identifiers if not provided
        if not quasi_identifiers:
            profile["auto_detected_qi"] = await self._detect_quasi_identifiers(data)
        
        # Calculate uniqueness and k-anonymity risks
        if quasi_identifiers:
            profile["k_anonymity"] = await self._calculate_k_anonymity(data, quasi_identifiers)
            profile["uniqueness_risk"] = await self._calculate_uniqueness_risk(data, quasi_identifiers)
        
        return profile
    
    async def _detect_quasi_identifiers(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect potential quasi-identifiers based on data characteristics"""
        potential_qi = []
        
        for col in data.columns:
            # High cardinality but not unique
            cardinality_ratio = data[col].nunique() / len(data)
            if 0.1 < cardinality_ratio < 0.9:
                potential_qi.append(col)
            
            # Common QI patterns
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['age', 'zip', 'postal', 'income', 'education', 'occupation']):
                potential_qi.append(col)
        
        return list(set(potential_qi))
    
    async def _calculate_k_anonymity(self, data: pd.DataFrame, quasi_identifiers: List[str]) -> Dict[str, Any]:
        """Calculate k-anonymity metrics"""
        if not quasi_identifiers:
            return {"k_value": float('inf'), "groups": 0}
        
        # Group by quasi-identifiers
        grouped = data.groupby(quasi_identifiers).size()
        k_value = grouped.min()
        
        return {
            "k_value": int(k_value),
            "total_groups": len(grouped),
            "avg_group_size": float(grouped.mean()),
            "groups_below_5": int(sum(grouped < 5))
        }
    
    async def _calculate_uniqueness_risk(self, data: pd.DataFrame, quasi_identifiers: List[str]) -> float:
        """Calculate uniqueness risk based on quasi-identifiers"""
        if not quasi_identifiers:
            return 0.0
        
        grouped = data.groupby(quasi_identifiers).size()
        unique_records = sum(grouped == 1)
        return unique_records / len(data)
    
    async def _apply_synthetic_data_generation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply synthetic data generation"""
        logger.info("Applying Synthetic Data Generation")
        return await self.sdg.generate_synthetic_data(data, **kwargs)
    
    async def _apply_differential_privacy(
        self,
        data: pd.DataFrame,
        epsilon: float,
        delta: float,
        **kwargs
    ) -> pd.DataFrame:
        """Apply differential privacy mechanisms"""
        logger.info(f"Applying Differential Privacy (ε={epsilon}, δ={delta})")
        return await self.dp_engine.apply_dp_noise(data, epsilon, delta, **kwargs)
    
    async def _apply_statistical_disclosure_control(
        self,
        data: pd.DataFrame,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Apply Statistical Disclosure Control techniques"""
        logger.info("Applying Statistical Disclosure Control")
        
        result_data = data.copy()
        
        # Generalization for quasi-identifiers
        if quasi_identifiers:
            for qi in quasi_identifiers:
                if qi in result_data.columns:
                    result_data[qi] = await self._generalize_column(result_data[qi])
        
        # Suppression for rare combinations
        result_data = await self._suppress_rare_combinations(result_data, quasi_identifiers)
        
        # Microaggregation for sensitive attributes
        if sensitive_attributes:
            for sa in sensitive_attributes:
                if sa in result_data.columns and pd.api.types.is_numeric_dtype(result_data[sa]):
                    result_data[sa] = await self._microaggregate_column(result_data[sa])
        
        return result_data
    
    async def _generalize_column(self, column: pd.Series) -> pd.Series:
        """Apply generalization to a column"""
        if pd.api.types.is_numeric_dtype(column):
            # Numeric generalization: binning
            return pd.cut(column, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        else:
            # Categorical generalization: reduce categories
            value_counts = column.value_counts()
            rare_values = value_counts[value_counts < len(column) * 0.05].index
            generalized = column.copy()
            generalized[generalized.isin(rare_values)] = 'Other'
            return generalized
    
    async def _suppress_rare_combinations(
        self,
        data: pd.DataFrame,
        quasi_identifiers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Suppress records with rare quasi-identifier combinations"""
        if not quasi_identifiers:
            return data
        
        # Find rare combinations (appearing less than k=3 times)
        grouped = data.groupby(quasi_identifiers).size()
        rare_combinations = grouped[grouped < 3].index
        
        # Create mask for rare combinations
        mask = pd.Series([True] * len(data), index=data.index)
        for combo in rare_combinations:
            if isinstance(combo, tuple):
                combo_mask = (data[quasi_identifiers] == combo).all(axis=1)
            else:
                combo_mask = data[quasi_identifiers[0]] == combo
            mask &= ~combo_mask
        
        return data[mask].copy()
    
    async def _microaggregate_column(self, column: pd.Series, k: int = 3) -> pd.Series:
        """Apply microaggregation to a numeric column"""
        # Sort values and group into clusters of size k
        sorted_data = column.sort_values()
        result = column.copy()
        
        for i in range(0, len(sorted_data), k):
            cluster = sorted_data.iloc[i:i+k]
            cluster_mean = cluster.mean()
            result.loc[cluster.index] = cluster_mean
        
        return result
    
    async def _assess_privacy(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame,
        epsilon: float,
        delta: float
    ) -> Dict[str, Any]:
        """Assess privacy protection of anonymized data"""
        
        # Calculate basic privacy metrics
        privacy_metrics = {
            "epsilon_used": epsilon,
            "delta_used": delta,
            "privacy_budget_remaining": max(0, settings.max_epsilon - epsilon),
            "data_reduction_ratio": len(anonymized_data) / len(original_data),
            "column_preservation_ratio": len(anonymized_data.columns) / len(original_data.columns)
        }
        
        # Calculate re-identification risk
        privacy_metrics["re_identification_risk"] = await self._calculate_reidentification_risk(
            original_data, anonymized_data
        )
        
        return privacy_metrics
    
    async def _calculate_reidentification_risk(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame
    ) -> float:
        """Calculate re-identification risk"""
        # Simplified risk calculation based on unique value preservation
        original_unique_ratio = original_data.nunique().mean() / len(original_data)
        anonymized_unique_ratio = anonymized_data.nunique().mean() / len(anonymized_data)
        
        risk_score = anonymized_unique_ratio / original_unique_ratio if original_unique_ratio > 0 else 0
        return min(1.0, risk_score)
    
    async def _assess_utility(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Assess utility preservation of anonymized data"""
        
        utility_metrics = {}
        
        # Statistical similarity
        utility_metrics["statistical_similarity"] = await self._calculate_statistical_similarity(
            original_data, anonymized_data
        )
        
        # Correlation preservation
        utility_metrics["correlation_preservation"] = await self._calculate_correlation_preservation(
            original_data, anonymized_data
        )
        
        # Distribution similarity
        utility_metrics["distribution_similarity"] = await self._calculate_distribution_similarity(
            original_data, anonymized_data
        )
        
        # Data completeness
        utility_metrics["data_completeness"] = 1.0 - (anonymized_data.isnull().sum().sum() / anonymized_data.size)
        
        # Overall utility score - ensure all values are in [0, 1] range
        metric_values = [max(0.0, min(1.0, v)) for v in utility_metrics.values() if isinstance(v, (int, float))]
        utility_metrics["overall_utility_score"] = max(0.0, min(1.0, np.mean(metric_values) if metric_values else 0.5))
        
        return utility_metrics
    
    async def _calculate_statistical_similarity(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame
    ) -> float:
        """Calculate statistical similarity between original and anonymized data"""
        similarities = []
        
        for col in original_data.columns:
            if col in anonymized_data.columns:
                if pd.api.types.is_numeric_dtype(original_data[col]):
                    # For numeric columns, compare means and standard deviations
                    orig_mean, orig_std = original_data[col].mean(), original_data[col].std()
                    anon_mean, anon_std = anonymized_data[col].mean(), anonymized_data[col].std()
                    
                    # Calculate relative differences and clamp to [0, 1] range
                    mean_diff = abs(orig_mean - anon_mean) / (abs(orig_mean) + 1e-8)
                    std_diff = abs(orig_std - anon_std) / (abs(orig_std) + 1e-8)
                    
                    mean_similarity = max(0.0, min(1.0, 1.0 - mean_diff))
                    std_similarity = max(0.0, min(1.0, 1.0 - std_diff))
                    similarities.append((mean_similarity + std_similarity) / 2)
                else:
                    # For categorical columns, compare value distributions
                    orig_dist = original_data[col].value_counts(normalize=True)
                    anon_dist = anonymized_data[col].value_counts(normalize=True)
                    
                    # Calculate overlap
                    common_values = set(orig_dist.index) & set(anon_dist.index)
                    if common_values:
                        overlap = sum(min(orig_dist.get(val, 0), anon_dist.get(val, 0)) for val in common_values)
                        similarities.append(overlap)
                    else:
                        similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def _calculate_correlation_preservation(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame
    ) -> float:
        """Calculate correlation preservation between datasets"""
        # Select only numeric columns
        orig_numeric = original_data.select_dtypes(include=[np.number])
        anon_numeric = anonymized_data.select_dtypes(include=[np.number])
        
        if orig_numeric.empty or anon_numeric.empty:
            return 1.0
        
        # Calculate correlation matrices
        orig_corr = orig_numeric.corr()
        anon_corr = anon_numeric.corr()
        
        # Calculate similarity between correlation matrices
        common_cols = list(set(orig_corr.columns) & set(anon_corr.columns))
        if len(common_cols) < 2:
            return 1.0
        
        orig_corr_subset = orig_corr.loc[common_cols, common_cols]
        anon_corr_subset = anon_corr.loc[common_cols, common_cols]
        
        # Calculate Frobenius norm of difference
        diff_norm = np.linalg.norm(orig_corr_subset.values - anon_corr_subset.values, 'fro')
        max_norm = np.linalg.norm(orig_corr_subset.values, 'fro')
        
        return max(0, 1 - diff_norm / (max_norm + 1e-8))
    
    async def _calculate_distribution_similarity(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame
    ) -> float:
        """Calculate distribution similarity using KS test for numeric columns"""
        from scipy import stats
        
        similarities = []
        
        for col in original_data.columns:
            if col in anonymized_data.columns and pd.api.types.is_numeric_dtype(original_data[col]):
                try:
                    # Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.ks_2samp(
                        original_data[col].dropna(),
                        anonymized_data[col].dropna()
                    )
                    # Convert to similarity score (lower KS statistic = higher similarity)
                    similarity = 1 - ks_stat
                    similarities.append(similarity)
                except:
                    similarities.append(0.5)  # Default similarity for failed tests
        
        return np.mean(similarities) if similarities else 1.0
