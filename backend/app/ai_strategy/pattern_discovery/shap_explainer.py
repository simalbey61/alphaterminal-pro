"""
AlphaTerminal Pro - SHAP Explainer
==================================

SHAP (SHapley Additive exPlanations) tabanlı feature importance ve nedensellik analizi.

Author: AlphaTerminal Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature önem skoru."""
    feature: str
    importance: float
    direction: str  # positive, negative, mixed
    mean_shap: float
    std_shap: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "importance": self.importance,
            "direction": self.direction,
            "mean_shap": self.mean_shap,
            "std_shap": self.std_shap,
        }


@dataclass
class FeatureInteraction:
    """Feature etkileşimi."""
    feature_1: str
    feature_2: str
    interaction_strength: float
    synergy_type: str  # synergistic, antagonistic, neutral
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_1": self.feature_1,
            "feature_2": self.feature_2,
            "interaction_strength": self.interaction_strength,
            "synergy_type": self.synergy_type,
        }


@dataclass
class SHAPExplanation:
    """SHAP açıklama sonucu."""
    feature_importances: List[FeatureImportance]
    interactions: List[FeatureInteraction]
    base_value: float
    model_type: str
    sample_size: int
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_importances": [f.to_dict() for f in self.feature_importances],
            "interactions": [i.to_dict() for i in self.interactions],
            "base_value": self.base_value,
            "model_type": self.model_type,
            "sample_size": self.sample_size,
            "calculated_at": self.calculated_at.isoformat(),
        }
    
    def get_top_features(self, n: int = 10) -> List[FeatureImportance]:
        """En önemli N feature."""
        sorted_features = sorted(
            self.feature_importances,
            key=lambda x: abs(x.importance),
            reverse=True
        )
        return sorted_features[:n]
    
    def get_positive_features(self) -> List[FeatureImportance]:
        """Pozitif etkili feature'lar."""
        return [f for f in self.feature_importances if f.direction == "positive"]
    
    def get_negative_features(self) -> List[FeatureImportance]:
        """Negatif etkili feature'lar."""
        return [f for f in self.feature_importances if f.direction == "negative"]


class SHAPExplainer:
    """
    SHAP tabanlı model açıklama ve feature importance analizi.
    
    Özellikler:
    - Feature importance hesaplama
    - Feature etkileşim analizi
    - Counterfactual senaryolar
    - Strateji başarı açıklaması
    
    Example:
        ```python
        explainer = SHAPExplainer()
        
        # Model açıkla
        explanation = explainer.explain(model, X, feature_names)
        
        # Top features
        top = explanation.get_top_features(10)
        
        # Strateji açıkla
        strategy_exp = explainer.explain_strategy(strategy, trades)
        ```
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        interaction_depth: int = 2,
    ):
        """
        Initialize SHAP Explainer.
        
        Args:
            n_samples: Background sample sayısı
            interaction_depth: Etkileşim analiz derinliği
        """
        self.n_samples = n_samples
        self.interaction_depth = interaction_depth
    
    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        model_type: str = "tree",
    ) -> SHAPExplanation:
        """
        Model için SHAP açıklama hesapla.
        
        Args:
            model: Scikit-learn uyumlu model
            X: Feature matrix
            feature_names: Feature adları
            model_type: Model tipi (tree, linear, kernel)
            
        Returns:
            SHAPExplanation: SHAP açıklama sonuçları
        """
        try:
            # SHAP kütüphanesi varsa kullan
            import shap
            
            if model_type == "tree":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            else:
                # Kernel SHAP (yavaş ama genel)
                background = X[np.random.choice(len(X), min(self.n_samples, len(X)), replace=False)]
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X[:min(100, len(X))])
            
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
        except ImportError:
            logger.warning("SHAP library not available, using permutation importance")
            shap_values, base_value = self._permutation_importance(model, X)
        except Exception as e:
            logger.error(f"SHAP calculation error: {e}")
            shap_values, base_value = self._permutation_importance(model, X)
        
        # Feature importances hesapla
        feature_importances = self._calculate_feature_importances(
            shap_values, feature_names
        )
        
        # Interactions hesapla
        interactions = self._calculate_interactions(
            shap_values, feature_names
        )
        
        return SHAPExplanation(
            feature_importances=feature_importances,
            interactions=interactions,
            base_value=float(base_value) if not isinstance(base_value, float) else base_value,
            model_type=model_type,
            sample_size=len(X),
        )
    
    def _permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Permutation importance (SHAP alternatifi)."""
        n_samples, n_features = X.shape
        
        # Base prediction
        try:
            base_pred = model.predict_proba(X)[:, 1]
        except:
            base_pred = model.predict(X)
        
        base_value = np.mean(base_pred)
        
        # Her feature için importance
        importances = np.zeros((n_samples, n_features))
        
        for feat_idx in range(n_features):
            X_permuted = X.copy()
            X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
            
            try:
                permuted_pred = model.predict_proba(X_permuted)[:, 1]
            except:
                permuted_pred = model.predict(X_permuted)
            
            importances[:, feat_idx] = base_pred - permuted_pred
        
        return importances, base_value
    
    def _calculate_feature_importances(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureImportance]:
        """Feature importance hesapla."""
        importances = []
        
        # Binary classification için
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        for i, name in enumerate(feature_names):
            values = shap_values[:, i]
            mean_shap = np.mean(values)
            std_shap = np.std(values)
            importance = np.mean(np.abs(values))
            
            # Direction belirleme
            if mean_shap > std_shap * 0.5:
                direction = "positive"
            elif mean_shap < -std_shap * 0.5:
                direction = "negative"
            else:
                direction = "mixed"
            
            importances.append(FeatureImportance(
                feature=name,
                importance=importance,
                direction=direction,
                mean_shap=mean_shap,
                std_shap=std_shap,
            ))
        
        return importances
    
    def _calculate_interactions(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> List[FeatureInteraction]:
        """Feature etkileşimlerini hesapla."""
        interactions = []
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        n_features = len(feature_names)
        
        # Top features arasındaki etkileşimleri hesapla
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-10:]
        
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                idx_i, idx_j = top_indices[i], top_indices[j]
                
                # Correlation of SHAP values
                corr = np.corrcoef(shap_values[:, idx_i], shap_values[:, idx_j])[0, 1]
                
                if np.isnan(corr):
                    continue
                
                interaction_strength = abs(corr)
                
                if interaction_strength < 0.3:
                    continue
                
                if corr > 0.3:
                    synergy_type = "synergistic"
                elif corr < -0.3:
                    synergy_type = "antagonistic"
                else:
                    synergy_type = "neutral"
                
                interactions.append(FeatureInteraction(
                    feature_1=feature_names[idx_i],
                    feature_2=feature_names[idx_j],
                    interaction_strength=interaction_strength,
                    synergy_type=synergy_type,
                ))
        
        return sorted(interactions, key=lambda x: x.interaction_strength, reverse=True)
    
    def explain_strategy(
        self,
        strategy_trades: List[Dict[str, Any]],
        feature_values: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Strateji başarısını açıkla.
        
        Args:
            strategy_trades: Strateji trade'leri
            feature_values: Trade feature değerleri
            feature_names: Feature adları
            
        Returns:
            Dict: Strateji açıklama sonuçları
        """
        if len(strategy_trades) == 0:
            return {"error": "No trades to analyze"}
        
        # Win/Loss ayır
        wins = [t for t in strategy_trades if t.get("pnl", 0) > 0]
        losses = [t for t in strategy_trades if t.get("pnl", 0) <= 0]
        
        win_indices = [i for i, t in enumerate(strategy_trades) if t.get("pnl", 0) > 0]
        loss_indices = [i for i, t in enumerate(strategy_trades) if t.get("pnl", 0) <= 0]
        
        # Feature means
        if len(win_indices) > 0 and len(loss_indices) > 0:
            win_features = feature_values[win_indices].mean(axis=0)
            loss_features = feature_values[loss_indices].mean(axis=0)
            
            # Fark analizi
            diff = win_features - loss_features
            
            # En önemli farklar
            important_diffs = []
            for i, name in enumerate(feature_names):
                if abs(diff[i]) > 0.1:  # Threshold
                    important_diffs.append({
                        "feature": name,
                        "win_mean": float(win_features[i]),
                        "loss_mean": float(loss_features[i]),
                        "difference": float(diff[i]),
                        "impact": "positive" if diff[i] > 0 else "negative",
                    })
            
            important_diffs.sort(key=lambda x: abs(x["difference"]), reverse=True)
        else:
            important_diffs = []
        
        return {
            "total_trades": len(strategy_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(strategy_trades) if strategy_trades else 0,
            "key_differentiators": important_diffs[:10],
            "analysis_type": "feature_comparison",
        }
    
    def counterfactual_analysis(
        self,
        sample: np.ndarray,
        model: Any,
        feature_names: List[str],
        target_prediction: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Counterfactual analiz: "Ne olsaydı farklı olurdu?"
        
        Args:
            sample: Analiz edilecek örnek
            model: Tahmin modeli
            feature_names: Feature adları
            target_prediction: Hedef tahmin değeri
            
        Returns:
            List[Dict]: Counterfactual senaryolar
        """
        try:
            current_pred = model.predict_proba(sample.reshape(1, -1))[0, 1]
        except:
            current_pred = model.predict(sample.reshape(1, -1))[0]
        
        counterfactuals = []
        
        for i, name in enumerate(feature_names):
            # Feature'ı %10 artır/azalt
            for change in [-0.1, -0.05, 0.05, 0.1]:
                modified = sample.copy()
                original_value = modified[i]
                modified[i] = original_value * (1 + change)
                
                try:
                    new_pred = model.predict_proba(modified.reshape(1, -1))[0, 1]
                except:
                    new_pred = model.predict(modified.reshape(1, -1))[0]
                
                pred_change = new_pred - current_pred
                
                if abs(pred_change) > 0.01:
                    counterfactuals.append({
                        "feature": name,
                        "original_value": float(original_value),
                        "modified_value": float(modified[i]),
                        "change_pct": change * 100,
                        "original_prediction": float(current_pred),
                        "new_prediction": float(new_pred),
                        "prediction_change": float(pred_change),
                    })
        
        return sorted(counterfactuals, key=lambda x: abs(x["prediction_change"]), reverse=True)[:20]
