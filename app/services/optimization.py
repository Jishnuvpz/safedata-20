import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import asyncio

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available. Using fallback optimization.")

from app.services.anonymization import AnonymizationEngine
from app.core.config import settings

logger = logging.getLogger(__name__)

class PrivacyUtilityOptimizer:
    """Bayesian optimization for privacy-utility trade-off"""
    
    def __init__(self):
        self.anonymization_engine = AnonymizationEngine()
        self.optimization_history = []
        
    async def optimize_parameters(
        self,
        data: pd.DataFrame,
        target_privacy_level: str = "medium",
        utility_weight: float = 0.5,
        privacy_weight: float = 0.5,
        max_iterations: int = 50,
        method: str = "full",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize privacy-utility trade-off using Bayesian optimization
        
        Args:
            data: Input dataset
            target_privacy_level: Target privacy level (low, medium, high, maximum)
            utility_weight: Weight for utility in objective function
            privacy_weight: Weight for privacy in objective function
            max_iterations: Maximum optimization iterations
            method: Anonymization method to optimize
            
        Returns:
            Optimization results with optimal parameters
        """
        
        logger.info(f"Starting privacy-utility optimization with target level: {target_privacy_level}")
        
        try:
            if SKOPT_AVAILABLE:
                results = await self._optimize_with_skopt(
                    data, target_privacy_level, utility_weight, privacy_weight,
                    max_iterations, method, **kwargs
                )
            else:
                results = await self._optimize_fallback(
                    data, target_privacy_level, utility_weight, privacy_weight,
                    max_iterations, method, **kwargs
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "optimal_parameters": await self._get_default_parameters(target_privacy_level)
            }
    
    async def _optimize_with_skopt(
        self,
        data: pd.DataFrame,
        target_privacy_level: str,
        utility_weight: float,
        privacy_weight: float,
        max_iterations: int,
        method: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using scikit-optimize Bayesian optimization"""
        
        # Define search space
        search_space = await self._define_search_space(method)
        
        # Define objective function
        objective_func = await self._create_objective_function(
            data, utility_weight, privacy_weight, method, **kwargs
        )
        
        # Perform optimization
        logger.info(f"Running Bayesian optimization with {max_iterations} iterations")
        
        optimization_result = gp_minimize(
            func=objective_func,
            dimensions=search_space,
            n_calls=max_iterations,
            random_state=42,
            acq_func='EI',  # Expected Improvement
            n_initial_points=10
        )
        
        # Extract optimal parameters
        optimal_params = await self._extract_optimal_parameters(
            optimization_result, method
        )
        
        # Evaluate optimal solution
        optimal_evaluation = await self._evaluate_solution(
            data, optimal_params, method, **kwargs
        )
        
        return {
            "success": True,
            "optimal_parameters": optimal_params,
            "expected_privacy_score": optimal_evaluation["privacy_score"],
            "expected_utility_score": optimal_evaluation["utility_score"],
            "objective_value": optimization_result.fun,
            "optimization_history": self.optimization_history,
            "convergence_info": {
                "n_calls": len(optimization_result.x_iters),
                "best_objective": optimization_result.fun,
                "function_evaluations": optimization_result.func_vals.tolist()
            }
        }
    
    async def _optimize_fallback(
        self,
        data: pd.DataFrame,
        target_privacy_level: str,
        utility_weight: float,
        privacy_weight: float,
        max_iterations: int,
        method: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback optimization using grid search"""
        
        logger.info("Using fallback grid search optimization")
        
        # Define parameter grid
        param_grid = await self._define_parameter_grid(method, target_privacy_level)
        
        best_score = float('inf')
        best_params = None
        best_evaluation = None
        
        iteration = 0
        for params in param_grid:
            if iteration >= max_iterations:
                break
            
            try:
                # Evaluate parameters
                evaluation = await self._evaluate_solution(data, params, method, **kwargs)
                
                # Calculate objective score
                objective_score = await self._calculate_objective_score(
                    evaluation, utility_weight, privacy_weight
                )
                
                # Track optimization history
                self.optimization_history.append({
                    "iteration": iteration,
                    "parameters": params,
                    "privacy_score": evaluation["privacy_score"],
                    "utility_score": evaluation["utility_score"],
                    "objective_score": objective_score
                })
                
                # Update best solution
                if objective_score < best_score:
                    best_score = objective_score
                    best_params = params
                    best_evaluation = evaluation
                
                iteration += 1
                
            except Exception as e:
                logger.warning(f"Evaluation failed for params {params}: {str(e)}")
                continue
        
        return {
            "success": True,
            "optimal_parameters": best_params or await self._get_default_parameters(target_privacy_level),
            "expected_privacy_score": best_evaluation["privacy_score"] if best_evaluation else 0.5,
            "expected_utility_score": best_evaluation["utility_score"] if best_evaluation else 0.5,
            "objective_value": best_score,
            "optimization_history": self.optimization_history
        }
    
    async def _define_search_space(self, method: str) -> List:
        """Define Bayesian optimization search space"""
        
        search_space = [
            Real(settings.min_epsilon, settings.max_epsilon, name='epsilon'),
            Real(1e-8, 1e-3, prior='log-uniform', name='delta'),
        ]
        
        if method in ["sdg", "sdg_dp", "sdg_sdc", "full"]:
            search_space.extend([
                Integer(100, 500, name='synthetic_epochs'),
                Integer(100, 1000, name='synthetic_batch_size'),
            ])
        
        if method in ["sdc", "sdg_sdc", "dp_sdc", "full"]:
            search_space.extend([
                Integer(3, 10, name='k_anonymity'),
                Real(0.01, 0.1, name='suppression_threshold'),
            ])
        
        return search_space
    
    async def _create_objective_function(
        self,
        data: pd.DataFrame,
        utility_weight: float,
        privacy_weight: float,
        method: str,
        **kwargs
    ) -> Callable:
        """Create objective function for optimization"""
        
        @use_named_args(await self._define_search_space(method))
        async def objective(**params):
            try:
                evaluation = await self._evaluate_solution(data, params, method, **kwargs)
                objective_score = await self._calculate_objective_score(
                    evaluation, utility_weight, privacy_weight
                )
                
                # Track optimization history
                self.optimization_history.append({
                    "iteration": len(self.optimization_history),
                    "parameters": params,
                    "privacy_score": evaluation["privacy_score"],
                    "utility_score": evaluation["utility_score"],
                    "objective_score": objective_score
                })
                
                return objective_score
                
            except Exception as e:
                logger.warning(f"Objective evaluation failed: {str(e)}")
                return 1.0  # Return high penalty for failed evaluations
        
        # Wrap async function for skopt
        def sync_objective(**params):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(objective(**params))
            finally:
                loop.close()
        
        return sync_objective
    
    async def _define_parameter_grid(
        self,
        method: str,
        target_privacy_level: str
    ) -> List[Dict[str, Any]]:
        """Define parameter grid for fallback optimization"""
        
        # Base epsilon values based on target privacy level
        if target_privacy_level == "low":
            epsilon_values = [5.0, 7.0, 10.0]
        elif target_privacy_level == "medium":
            epsilon_values = [1.0, 2.0, 3.0]
        elif target_privacy_level == "high":
            epsilon_values = [0.1, 0.5, 1.0]
        else:  # maximum
            epsilon_values = [0.01, 0.05, 0.1]
        
        delta_values = [1e-5, 1e-6, 1e-7]
        
        param_grid = []
        
        for epsilon in epsilon_values:
            for delta in delta_values:
                params = {
                    "epsilon": epsilon,
                    "delta": delta
                }
                
                if method in ["sdg", "sdg_dp", "sdg_sdc", "full"]:
                    params.update({
                        "synthetic_epochs": 300,
                        "synthetic_batch_size": 500
                    })
                
                if method in ["sdc", "sdg_sdc", "dp_sdc", "full"]:
                    params.update({
                        "k_anonymity": 5,
                        "suppression_threshold": 0.05
                    })
                
                param_grid.append(params)
        
        return param_grid
    
    async def _evaluate_solution(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        method: str,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate a parameter configuration"""
        
        # Apply anonymization with given parameters
        result = await self.anonymization_engine.anonymize_data(
            data=data,
            method=method,
            **parameters,
            **kwargs
        )
        
        if not result["success"]:
            raise ValueError(f"Anonymization failed: {result.get('error', 'Unknown error')}")
        
        # Extract privacy and utility scores
        privacy_metrics = result["privacy_metrics"]
        utility_metrics = result["utility_metrics"]
        attack_results = result["attack_results"]
        
        # Calculate privacy score (lower is better privacy protection)
        privacy_score = await self._calculate_privacy_score(privacy_metrics, attack_results)
        
        # Calculate utility score (higher is better utility preservation)
        utility_score = await self._calculate_utility_score(utility_metrics)
        
        return {
            "privacy_score": privacy_score,
            "utility_score": utility_score,
            "privacy_metrics": privacy_metrics,
            "utility_metrics": utility_metrics,
            "attack_results": attack_results
        }
    
    async def _calculate_privacy_score(
        self,
        privacy_metrics: Dict[str, Any],
        attack_results: Dict[str, Any]
    ) -> float:
        """Calculate privacy protection score (0 = best privacy, 1 = worst privacy)"""
        
        scores = []
        
        # Re-identification risk
        reident_risk = privacy_metrics.get("re_identification_risk", 0.5)
        scores.append(reident_risk)
        
        # Attack success rates
        if "linkage" in attack_results:
            linkage_success = attack_results["linkage"].get("success_rate", 0.5)
            scores.append(linkage_success)
        
        if "membership_inference" in attack_results:
            membership_advantage = attack_results["membership_inference"].get("attack_advantage", 0.0)
            scores.append(membership_advantage)
        
        if "attribute_inference" in attack_results:
            attribute_advantage = attack_results["attribute_inference"].get("overall_attack_advantage", 0.0)
            scores.append(attribute_advantage)
        
        # Overall risk score
        overall_risk = attack_results.get("overall_risk_score", 0.5)
        scores.append(overall_risk)
        
        # Return average score (normalized to [0, 1])
        privacy_score = np.mean(scores) if scores else 0.5
        return min(1.0, max(0.0, privacy_score))
    
    async def _calculate_utility_score(self, utility_metrics: Dict[str, Any]) -> float:
        """Calculate utility preservation score (1 = best utility, 0 = worst utility)"""
        
        scores = []
        
        # Statistical similarity
        stat_sim = utility_metrics.get("statistical_similarity", 0.5)
        scores.append(stat_sim)
        
        # Correlation preservation
        corr_pres = utility_metrics.get("correlation_preservation", 0.5)
        scores.append(corr_pres)
        
        # Distribution similarity
        dist_sim = utility_metrics.get("distribution_similarity", 0.5)
        scores.append(dist_sim)
        
        # Data completeness
        completeness = utility_metrics.get("data_completeness", 0.5)
        scores.append(completeness)
        
        # Overall utility score
        overall_utility = utility_metrics.get("overall_utility_score", 0.5)
        scores.append(overall_utility)
        
        # Return average score (normalized to [0, 1])
        utility_score = np.mean(scores) if scores else 0.5
        return min(1.0, max(0.0, utility_score))
    
    async def _calculate_objective_score(
        self,
        evaluation: Dict[str, float],
        utility_weight: float,
        privacy_weight: float
    ) -> float:
        """Calculate objective function score for optimization"""
        
        privacy_score = evaluation["privacy_score"]  # Lower is better
        utility_score = evaluation["utility_score"]  # Higher is better
        
        # Convert utility score to loss (lower is better)
        utility_loss = 1.0 - utility_score
        
        # Weighted combination
        objective_score = privacy_weight * privacy_score + utility_weight * utility_loss
        
        return objective_score
    
    async def _extract_optimal_parameters(
        self,
        optimization_result,
        method: str
    ) -> Dict[str, Any]:
        """Extract optimal parameters from optimization result"""
        
        search_space = await self._define_search_space(method)
        optimal_values = optimization_result.x
        
        optimal_params = {}
        for i, dimension in enumerate(search_space):
            optimal_params[dimension.name] = optimal_values[i]
        
        return optimal_params
    
    async def _get_default_parameters(self, target_privacy_level: str) -> Dict[str, Any]:
        """Get default parameters based on target privacy level"""
        
        if target_privacy_level == "low":
            return {"epsilon": 5.0, "delta": 1e-5}
        elif target_privacy_level == "medium":
            return {"epsilon": 1.0, "delta": 1e-5}
        elif target_privacy_level == "high":
            return {"epsilon": 0.5, "delta": 1e-6}
        else:  # maximum
            return {"epsilon": 0.1, "delta": 1e-7}
    
    async def get_optimization_report(
        self,
        optimization_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate optimization report"""
        
        if not optimization_history:
            return {"error": "No optimization history available"}
        
        # Extract metrics
        iterations = [entry["iteration"] for entry in optimization_history]
        privacy_scores = [entry["privacy_score"] for entry in optimization_history]
        utility_scores = [entry["utility_score"] for entry in optimization_history]
        objective_scores = [entry["objective_score"] for entry in optimization_history]
        
        # Find best solution
        best_idx = np.argmin(objective_scores)
        best_solution = optimization_history[best_idx]
        
        # Calculate convergence metrics
        convergence_rate = await self._calculate_convergence_rate(objective_scores)
        
        return {
            "total_iterations": len(optimization_history),
            "best_iteration": best_solution["iteration"],
            "best_objective_score": best_solution["objective_score"],
            "best_privacy_score": best_solution["privacy_score"],
            "best_utility_score": best_solution["utility_score"],
            "convergence_rate": convergence_rate,
            "final_improvement": objective_scores[0] - min(objective_scores),
            "parameter_sensitivity": await self._analyze_parameter_sensitivity(optimization_history),
            "recommendations": await self._generate_optimization_recommendations(optimization_history)
        }
    
    async def _calculate_convergence_rate(self, objective_scores: List[float]) -> float:
        """Calculate optimization convergence rate"""
        
        if len(objective_scores) < 2:
            return 0.0
        
        # Calculate improvement rate over iterations
        improvements = []
        best_so_far = objective_scores[0]
        
        for score in objective_scores[1:]:
            if score < best_so_far:
                improvement = (best_so_far - score) / best_so_far
                improvements.append(improvement)
                best_so_far = score
            else:
                improvements.append(0.0)
        
        return np.mean(improvements) if improvements else 0.0
    
    async def _analyze_parameter_sensitivity(
        self,
        optimization_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze parameter sensitivity"""
        
        if len(optimization_history) < 10:
            return {}
        
        # Extract parameter values and objective scores
        param_names = list(optimization_history[0]["parameters"].keys())
        param_data = {name: [] for name in param_names}
        objective_scores = []
        
        for entry in optimization_history:
            for name in param_names:
                param_data[name].append(entry["parameters"][name])
            objective_scores.append(entry["objective_score"])
        
        # Calculate correlation between each parameter and objective
        sensitivities = {}
        for name in param_names:
            try:
                correlation = np.corrcoef(param_data[name], objective_scores)[0, 1]
                sensitivities[name] = abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                sensitivities[name] = 0.0
        
        return sensitivities
    
    async def _generate_optimization_recommendations(
        self,
        optimization_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        if not optimization_history:
            return ["No optimization data available"]
        
        # Analyze convergence
        objective_scores = [entry["objective_score"] for entry in optimization_history]
        final_improvement = objective_scores[0] - min(objective_scores)
        
        if final_improvement < 0.01:
            recommendations.append("Consider expanding search space or increasing iterations")
        
        # Analyze parameter sensitivity
        sensitivity = await self._analyze_parameter_sensitivity(optimization_history)
        
        if sensitivity:
            most_sensitive = max(sensitivity.items(), key=lambda x: x[1])
            recommendations.append(f"Parameter '{most_sensitive[0]}' has highest impact on results")
        
        # Analyze privacy-utility trade-off
        privacy_scores = [entry["privacy_score"] for entry in optimization_history]
        utility_scores = [entry["utility_score"] for entry in optimization_history]
        
        avg_privacy = np.mean(privacy_scores)
        avg_utility = np.mean(utility_scores)
        
        if avg_privacy > 0.7:
            recommendations.append("Consider stronger privacy protection methods")
        elif avg_privacy < 0.3:
            recommendations.append("Current privacy protection is strong - consider relaxing for better utility")
        
        if avg_utility < 0.5:
            recommendations.append("Utility preservation is low - consider adjusting anonymization parameters")
        
        return recommendations if recommendations else ["Optimization results look good"]
