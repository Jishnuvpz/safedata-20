import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AttackSimulator:
    """Simulate various privacy attacks to evaluate anonymization effectiveness"""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        
    async def simulate_attacks(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame,
        attack_types: List[str] = ["linkage", "membership", "attribute"],
        auxiliary_data_ratio: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate multiple privacy attacks
        
        Args:
            original_data: Original dataset
            anonymized_data: Anonymized dataset
            attack_types: List of attack types to simulate
            auxiliary_data_ratio: Fraction of data available to attacker
            
        Returns:
            Attack simulation results
        """
        
        logger.info(f"Simulating attacks: {attack_types}")
        
        results = {}
        
        try:
            for attack_type in attack_types:
                if attack_type.lower() == "linkage":
                    results["linkage"] = await self._simulate_linkage_attack(
                        original_data, anonymized_data, auxiliary_data_ratio, **kwargs
                    )
                elif attack_type.lower() == "membership":
                    results["membership"] = await self._simulate_membership_inference_attack(
                        original_data, anonymized_data, **kwargs
                    )
                elif attack_type.lower() == "attribute":
                    results["attribute"] = await self._simulate_attribute_inference_attack(
                        original_data, anonymized_data, **kwargs
                    )
                else:
                    logger.warning(f"Unknown attack type: {attack_type}")
            
            # Calculate overall risk score
            overall_risk = await self._calculate_overall_risk(results)
            results["overall_risk_score"] = overall_risk
            
            # Generate recommendations
            results["recommendations"] = await self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Attack simulation failed: {str(e)}")
            return {
                "error": str(e),
                "overall_risk_score": 0.5,  # Default moderate risk
                "recommendations": ["Error in attack simulation - consider increasing privacy protection"]
            }
    
    async def _simulate_linkage_attack(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame,
        auxiliary_data_ratio: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate record linkage attack
        
        The attacker tries to link records between original and anonymized datasets
        using auxiliary information.
        """
        
        logger.info("Simulating linkage attack")
        
        try:
            # Create auxiliary dataset (subset of original data known to attacker)
            aux_size = int(len(original_data) * auxiliary_data_ratio)
            aux_indices = np.random.choice(len(original_data), aux_size, replace=False)
            auxiliary_data = original_data.iloc[aux_indices].copy()
            
            # Prepare data for linkage
            orig_prepared = await self._prepare_data_for_attack(original_data, "linkage")
            anon_prepared = await self._prepare_data_for_attack(anonymized_data, "linkage")
            aux_prepared = await self._prepare_data_for_attack(auxiliary_data, "linkage")
            
            # Calculate similarity scores between auxiliary and anonymized records
            linkage_scores = await self._calculate_linkage_scores(aux_prepared, anon_prepared)
            
            # Evaluate linkage success
            true_links = 0
            total_attempts = min(len(auxiliary_data), len(anonymized_data))
            
            for i in range(total_attempts):
                # Find best match in anonymized data for each auxiliary record
                best_match_idx = np.argmax(linkage_scores[i])
                
                # Check if this is a correct link (simplified evaluation)
                if await self._is_correct_link(auxiliary_data.iloc[i], anonymized_data.iloc[best_match_idx]):
                    true_links += 1
            
            success_rate = true_links / total_attempts if total_attempts > 0 else 0
            
            return {
                "attack_type": "linkage",
                "success_rate": success_rate,
                "attempted_links": total_attempts,
                "successful_links": true_links,
                "risk_level": "high" if success_rate > 0.3 else "medium" if success_rate > 0.1 else "low",
                "auxiliary_data_size": aux_size,
                "max_similarity_score": float(np.max(linkage_scores)) if linkage_scores.size > 0 else 0.0,
                "avg_similarity_score": float(np.mean(linkage_scores)) if linkage_scores.size > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Linkage attack simulation failed: {str(e)}")
            return {
                "attack_type": "linkage",
                "success_rate": 0.0,
                "error": str(e),
                "risk_level": "unknown"
            }
    
    async def _simulate_membership_inference_attack(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate membership inference attack
        
        The attacker tries to determine if a specific record was in the original dataset
        by analyzing the anonymized dataset.
        """
        
        logger.info("Simulating membership inference attack")
        
        try:
            # Prepare datasets
            orig_prepared = await self._prepare_data_for_attack(original_data, "membership")
            anon_prepared = await self._prepare_data_for_attack(anonymized_data, "membership")
            
            # Create training data for membership classifier
            # Positive samples: records that were in original dataset
            # Negative samples: synthetic records not in original dataset
            
            n_samples = min(1000, len(original_data))
            
            # Positive samples (members)
            member_indices = np.random.choice(len(original_data), n_samples // 2, replace=False)
            member_features = await self._extract_membership_features(
                original_data.iloc[member_indices], anonymized_data
            )
            member_labels = np.ones(len(member_features))
            
            # Negative samples (non-members) - generate synthetic non-member records
            non_member_data = await self._generate_non_member_records(original_data, n_samples // 2)
            non_member_features = await self._extract_membership_features(non_member_data, anonymized_data)
            non_member_labels = np.zeros(len(non_member_features))
            
            # Combine training data
            X = np.vstack([member_features, non_member_features])
            y = np.hstack([member_labels, non_member_labels])
            
            # Train membership inference classifier
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train, y_train)
            
            # Evaluate attack performance
            y_pred = classifier.predict(X_test)
            y_pred_proba = classifier.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Baseline accuracy for random guessing
            baseline_accuracy = 0.5
            attack_advantage = accuracy - baseline_accuracy
            
            return {
                "attack_type": "membership_inference",
                "accuracy": accuracy,
                "auc_score": auc_score,
                "attack_advantage": attack_advantage,
                "baseline_accuracy": baseline_accuracy,
                "risk_level": "high" if attack_advantage > 0.2 else "medium" if attack_advantage > 0.1 else "low",
                "sample_size": n_samples,
                "feature_importance": dict(zip(
                    [f"feature_{i}" for i in range(X.shape[1])],
                    classifier.feature_importances_.tolist()
                ))
            }
            
        except Exception as e:
            logger.error(f"Membership inference attack simulation failed: {str(e)}")
            return {
                "attack_type": "membership_inference",
                "accuracy": 0.5,
                "attack_advantage": 0.0,
                "error": str(e),
                "risk_level": "unknown"
            }
    
    async def _simulate_attribute_inference_attack(
        self,
        original_data: pd.DataFrame,
        anonymized_data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate attribute inference attack
        
        The attacker tries to infer sensitive attributes from non-sensitive ones
        in the anonymized dataset.
        """
        
        logger.info("Simulating attribute inference attack")
        
        try:
            # Select target attributes (assume last column is sensitive)
            target_columns = kwargs.get('target_columns', [original_data.columns[-1]])
            
            results = {}
            
            for target_col in target_columns:
                if target_col not in anonymized_data.columns:
                    continue
                
                # Prepare features (all columns except target)
                feature_cols = [col for col in anonymized_data.columns if col != target_col]
                
                if not feature_cols:
                    continue
                
                # Prepare data
                X = await self._prepare_features_for_inference(anonymized_data[feature_cols])
                y = await self._prepare_target_for_inference(anonymized_data[target_col])
                
                # Train inference model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                if len(np.unique(y)) > 2:
                    # Multi-class classification
                    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    baseline_accuracy = 1.0 / len(np.unique(y))
                else:
                    # Binary classification
                    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    baseline_accuracy = 0.5
                
                attack_advantage = accuracy - baseline_accuracy
                
                results[target_col] = {
                    "accuracy": accuracy,
                    "baseline_accuracy": baseline_accuracy,
                    "attack_advantage": attack_advantage,
                    "auc_score": auc_score if len(np.unique(y)) == 2 else None,
                    "risk_level": "high" if attack_advantage > 0.2 else "medium" if attack_advantage > 0.1 else "low",
                    "feature_importance": dict(zip(
                        feature_cols,
                        classifier.feature_importances_.tolist()
                    )) if hasattr(classifier, 'feature_importances_') else {}
                }
            
            # Calculate overall attribute inference risk
            if results:
                avg_advantage = np.mean([res["attack_advantage"] for res in results.values()])
                overall_risk = "high" if avg_advantage > 0.2 else "medium" if avg_advantage > 0.1 else "low"
            else:
                avg_advantage = 0.0
                overall_risk = "low"
            
            return {
                "attack_type": "attribute_inference",
                "target_attributes": results,
                "overall_attack_advantage": avg_advantage,
                "overall_risk_level": overall_risk,
                "attributes_analyzed": len(results)
            }
            
        except Exception as e:
            logger.error(f"Attribute inference attack simulation failed: {str(e)}")
            return {
                "attack_type": "attribute_inference",
                "overall_attack_advantage": 0.0,
                "error": str(e),
                "overall_risk_level": "unknown"
            }
    
    async def _prepare_data_for_attack(self, data: pd.DataFrame, attack_type: str) -> pd.DataFrame:
        """Prepare data for specific attack type"""
        
        prepared_data = data.copy()
        
        # Handle missing values
        for col in prepared_data.columns:
            if prepared_data[col].dtype == 'object':
                prepared_data[col].fillna('missing', inplace=True)
            else:
                prepared_data[col].fillna(prepared_data[col].median(), inplace=True)
        
        # Encode categorical variables
        for col in prepared_data.select_dtypes(include=['object']).columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                prepared_data[col] = self.encoders[col].fit_transform(prepared_data[col].astype(str))
            else:
                # Handle unseen categories
                le = self.encoders[col]
                mask = prepared_data[col].astype(str).isin(le.classes_)
                prepared_data.loc[mask, col] = le.transform(prepared_data.loc[mask, col].astype(str))
                prepared_data.loc[~mask, col] = -1  # Assign -1 to unseen categories
        
        return prepared_data
    
    async def _calculate_linkage_scores(
        self,
        auxiliary_data: pd.DataFrame,
        anonymized_data: pd.DataFrame
    ) -> np.ndarray:
        """Calculate similarity scores for record linkage"""
        
        # Normalize data
        scaler = StandardScaler()
        aux_scaled = scaler.fit_transform(auxiliary_data)
        anon_scaled = scaler.transform(anonymized_data)
        
        # Calculate cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(aux_scaled, anon_scaled)
        
        return similarity_matrix
    
    async def _is_correct_link(
        self,
        aux_record: pd.Series,
        anon_record: pd.Series,
        threshold: float = 0.8
    ) -> bool:
        """Determine if linkage is correct (simplified heuristic)"""
        
        # Calculate similarity between records
        similarities = []
        
        for col in aux_record.index:
            if col in anon_record.index:
                aux_val = aux_record[col]
                anon_val = anon_record[col]
                
                if pd.api.types.is_numeric_dtype(type(aux_val)) and pd.api.types.is_numeric_dtype(type(anon_val)):
                    # Numeric similarity
                    max_val = max(abs(aux_val), abs(anon_val), 1e-8)
                    similarity = 1 - abs(aux_val - anon_val) / max_val
                else:
                    # Categorical similarity
                    similarity = 1.0 if aux_val == anon_val else 0.0
                
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return avg_similarity >= threshold
    
    async def _extract_membership_features(
        self,
        query_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> np.ndarray:
        """Extract features for membership inference attack"""
        
        features = []
        
        for _, query_record in query_data.iterrows():
            record_features = []
            
            # Distance-based features
            distances = []
            for _, ref_record in reference_data.iterrows():
                distance = await self._calculate_record_distance(query_record, ref_record)
                distances.append(distance)
            
            distances = np.array(distances)
            
            # Statistical features
            record_features.extend([
                np.min(distances),      # Minimum distance
                np.mean(distances),     # Average distance
                np.std(distances),      # Standard deviation of distances
                np.percentile(distances, 25),  # 25th percentile
                np.percentile(distances, 75),  # 75th percentile
            ])
            
            features.append(record_features)
        
        return np.array(features)
    
    async def _calculate_record_distance(
        self,
        record1: pd.Series,
        record2: pd.Series
    ) -> float:
        """Calculate distance between two records"""
        
        distances = []
        
        for col in record1.index:
            if col in record2.index:
                val1, val2 = record1[col], record2[col]
                
                if pd.api.types.is_numeric_dtype(type(val1)) and pd.api.types.is_numeric_dtype(type(val2)):
                    # Numeric distance (normalized)
                    max_val = max(abs(val1), abs(val2), 1e-8)
                    distance = abs(val1 - val2) / max_val
                else:
                    # Categorical distance
                    distance = 0.0 if val1 == val2 else 1.0
                
                distances.append(distance)
        
        return np.mean(distances) if distances else 1.0
    
    async def _generate_non_member_records(
        self,
        original_data: pd.DataFrame,
        n_samples: int
    ) -> pd.DataFrame:
        """Generate synthetic non-member records"""
        
        non_members = pd.DataFrame()
        
        for col in original_data.columns:
            if pd.api.types.is_numeric_dtype(original_data[col]):
                # Generate numeric values outside the original range
                col_min, col_max = original_data[col].min(), original_data[col].max()
                col_range = col_max - col_min
                
                # Generate values with some offset from original range
                offset = col_range * 0.1
                synthetic_values = np.random.normal(
                    loc=col_min - offset,
                    scale=col_range * 0.5,
                    size=n_samples
                )
                non_members[col] = synthetic_values
            else:
                # For categorical, use rare or synthetic categories
                original_values = original_data[col].unique()
                synthetic_values = [f"synthetic_{col}_{i}" for i in range(n_samples)]
                non_members[col] = synthetic_values
        
        return non_members
    
    async def _prepare_features_for_inference(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for attribute inference"""
        
        prepared_features = features.copy()
        
        # Encode categorical variables
        for col in prepared_features.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            prepared_features[col] = le.fit_transform(prepared_features[col].astype(str))
        
        # Handle missing values
        prepared_features.fillna(0, inplace=True)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(prepared_features)
        
        return scaled_features
    
    async def _prepare_target_for_inference(self, target: pd.Series) -> np.ndarray:
        """Prepare target variable for attribute inference"""
        
        if target.dtype == 'object':
            le = LabelEncoder()
            return le.fit_transform(target.astype(str))
        else:
            # For numeric targets, discretize into bins
            return pd.cut(target, bins=5, labels=False)
    
    async def _calculate_overall_risk(self, attack_results: Dict[str, Any]) -> float:
        """Calculate overall privacy risk score"""
        
        risk_scores = []
        
        for attack_type, results in attack_results.items():
            if isinstance(results, dict) and "risk_level" in results:
                risk_level = results["risk_level"]
                if risk_level == "high":
                    risk_scores.append(0.8)
                elif risk_level == "medium":
                    risk_scores.append(0.5)
                elif risk_level == "low":
                    risk_scores.append(0.2)
                else:
                    risk_scores.append(0.5)  # Default for unknown
        
        return np.mean(risk_scores) if risk_scores else 0.5
    
    async def _generate_recommendations(self, attack_results: Dict[str, Any]) -> List[str]:
        """Generate privacy protection recommendations based on attack results"""
        
        recommendations = []
        
        # Check linkage attack results
        if "linkage" in attack_results:
            linkage_risk = attack_results["linkage"].get("risk_level", "unknown")
            if linkage_risk == "high":
                recommendations.append("Consider increasing generalization of quasi-identifiers")
                recommendations.append("Apply stronger statistical disclosure control")
            elif linkage_risk == "medium":
                recommendations.append("Review quasi-identifier selection and apply targeted generalization")
        
        # Check membership inference results
        if "membership_inference" in attack_results:
            membership_risk = attack_results["membership_inference"].get("risk_level", "unknown")
            if membership_risk == "high":
                recommendations.append("Increase differential privacy epsilon budget")
                recommendations.append("Apply stronger noise injection mechanisms")
            elif membership_risk == "medium":
                recommendations.append("Consider adjusting differential privacy parameters")
        
        # Check attribute inference results
        if "attribute_inference" in attack_results:
            attribute_risk = attack_results["attribute_inference"].get("overall_risk_level", "unknown")
            if attribute_risk == "high":
                recommendations.append("Apply stronger suppression of sensitive attributes")
                recommendations.append("Increase correlation disruption between features")
            elif attribute_risk == "medium":
                recommendations.append("Review correlation patterns and apply targeted perturbation")
        
        # General recommendations
        overall_risk = attack_results.get("overall_risk_score", 0.5)
        if overall_risk > 0.7:
            recommendations.append("Consider using stronger anonymization methods")
            recommendations.append("Reduce data granularity and increase aggregation")
        elif overall_risk > 0.4:
            recommendations.append("Fine-tune privacy parameters for better protection")
        
        if not recommendations:
            recommendations.append("Current privacy protection appears adequate")
            recommendations.append("Continue monitoring with periodic re-evaluation")
        
        return recommendations
