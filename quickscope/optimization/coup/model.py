"""Surrogate models for COUP optimization.

Contains model classes used as surrogates in Bayesian optimization:
- SurrogateModel: Protocol defining the expected interface
- ConfigEncoder: Encodes ConfigSpace params to numeric vectors
- XGBoostModel: XGBoost-based surrogate (original COUP model)
- RFModel: Random Forest-based surrogate (simpler alternative)
"""

from typing import Any, Dict, Optional, Protocol, Sequence #, Protocol, runtime_checkable

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)


# @runtime_checkable
class SurrogateModel(Protocol):
    """Protocol defining the interface for COUP surrogate models.

    Any class with train() and predict() methods matching these signatures
    can be used as a surrogate model (duck typing).
    """

    def train(
        self, configs: dict, targets_by_index: Optional[Dict[int, float]] = None
    ) -> None:
        """Train the model on configs using observed utilities or provided targets."""
        ...

    def predict(self, candidates: list) -> np.ndarray:
        """Predict utility values for candidate configurations."""
        ...
    
    def is_fitted(self) -> bool:
        """Check if the model has been fitted/trained."""
        ...

    def compute_proximity_matrix(
        self, cfgs_a: list[dict[str, Any]], cfgs_b: list[dict[str, Any]]
    ) -> np.ndarray:
        """Compute proximity between two sets of configs."""
        ...


class ConfigEncoder:
    """Encode ConfigSpace samples (dict) into fixed-length numeric vectors."""

    def __init__(self, cs: ConfigurationSpace):
        self.cs = cs
        self.hp_defs: list[tuple[str, str, Any]] = []
        feature_names: list[str] = []

        for hp in self.cs.values():
            if isinstance(
                hp, (CategoricalHyperparameter, OrdinalHyperparameter, Constant)
            ):
                choices = list(
                    hp.sequence if isinstance(hp, OrdinalHyperparameter) else hp.choices
                )
                self.hp_defs.append((hp.name, "cat", choices))
                feature_names.extend([f"{hp.name}={c}" for c in choices])
            elif isinstance(
                hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)
            ):
                self.hp_defs.append((hp.name, "num", (hp.lower, hp.upper)))
                feature_names.append(hp.name)
            else:
                # Fallback: treat as categorical string
                self.hp_defs.append((hp.name, "cat", []))
                feature_names.append(f"{hp.name}=unk")

        self.feature_names = feature_names

    def encode_one(self, cfg: dict[str, Any]) -> np.ndarray:
        vals: list[float] = []
        for name, kind, meta in self.hp_defs:
            val = cfg.get(name)
            if kind == "cat":
                choices: Sequence[Any] = meta
                if not choices:
                    vals.append(0.0)
                    continue
                one_hot = [0.0] * len(choices)
                if val in choices:
                    idx = choices.index(val)
                    one_hot[idx] = 1.0
                vals.extend(one_hot)
            elif kind == "num":
                lower, upper = meta
                try:
                    v = float(val)
                    if upper > lower:
                        v = (v - lower) / (upper - lower)
                    vals.append(float(np.clip(v, 0.0, 1.0)))
                except Exception:
                    vals.append(0.0)
            else:
                vals.append(0.0)
        return np.array(vals, dtype=float)

    def encode_batch(self, cfgs: list[dict[str, Any]]) -> np.ndarray:
        if not cfgs:
            return np.zeros((0, len(self.feature_names)))
        return np.stack([self.encode_one(cfg) for cfg in cfgs], axis=0)


class XGBoostModel(SurrogateModel):
    """XGBoost surrogate model for COUP.

    Uses bootstrap sampling to train an ensemble of XGBoost models.
    """

    def __init__(
        self,
        u,
        feature_names,
        num_models=10,
        num_bootstrap_samples=1000,
        num_trees=2000,
        max_depth=2,
        learning_rate=0.01,
        validation_configs=None,
        seed=1,
    ):
        self.u = u
        self.u_vect = np.vectorize(u, otypes=[float])
        self.feature_names = feature_names
        self.num_models = num_models
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_trees = num_trees
        self.model_params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
        }
        self.validation_configs = validation_configs
        np.random.seed(seed)

    def train(self, configs, targets_by_index: Optional[Dict[int, float]] = None):
        print("Training model ... ")
        ordered_indices = list(configs.keys())
        if len(ordered_indices) < 2:
            return
        if targets_by_index is not None:
            ordered_indices = [
                i
                for i in ordered_indices
                if i in targets_by_index and np.isfinite(float(targets_by_index[i]))
            ]
            if len(ordered_indices) < 2:
                return
            X = np.stack([configs[i]["config"].get_array() for i in ordered_indices])
            y = np.array([float(targets_by_index[i]) for i in ordered_indices])
        else:
            X = np.stack([configs[i]["config"].get_array() for i in ordered_indices])
            first_idx = ordered_indices[0]
            if type(configs[first_idx]["runtimes"]) is np.ndarray:
                y = np.array(
                    [np.mean(self.u_vect(configs[i]["runtimes"])) for i in ordered_indices]
                )
            else:
                y = np.array(
                    [
                        np.mean([self.u(t) for t in configs[i]["runtimes"].values()])
                        for i in ordered_indices
                    ]
                )

        # Take bootstrap samples and train models:
        self.models = []
        for i in range(self.num_models):
            sample = np.random.choice(
                y.shape[0], size=self.num_bootstrap_samples, replace=True
            )
            X_boot = X[sample, :]
            y_boot = y[sample]
            dboot = xgb.DMatrix(X_boot, y_boot, feature_names=self.feature_names)
            model = xgb.train(self.model_params, dboot, num_boost_round=self.num_trees)

            self.models.append(model)

    def predict(self, configs, average=True):
        if isinstance(configs, list):
            X_test = np.stack([c.get_array() for c in configs])
        elif isinstance(configs, dict):
            X_test = np.stack([configs[i]["config"].get_array() for i in configs])
        else:  # just a single config
            X_test = [configs.get_array()]

        dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)

        y_hats = np.zeros((len(self.models), len(configs)))
        for m, model in enumerate(self.models):
            y_hat = model.predict(dtest)
            y_hats[m, :] = y_hat
        if average:
            y_hats = np.mean(y_hats, axis=0)

        if isinstance(configs, list) or isinstance(configs, dict):
            return y_hats
        else:
            return y_hats[0]
    
    def is_fitted(self) -> bool:
        return hasattr(self, "models") and len(self.models) > 0


class RFModel(SurrogateModel):
    """Random Forest surrogate model for COUP.

    Simpler alternative to XGBoostModel, uses ConfigEncoder for feature extraction.
    """

    def __init__(
        self,
        encoder: ConfigEncoder,
        n_estimators: int = 200,
        random_state: int = 0,
    ):
        self.encoder = encoder
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def train(
        self,
        configs: Dict[int, Dict[str, Any]],
        targets_by_index: Optional[Dict[int, float]] = None,
    ) -> None:
        if self.model is None:
            return
        cfgs = []
        ys = []
        for idx, cfg in configs.items():
            target = None
            if targets_by_index is not None and idx in targets_by_index:
                target_candidate = float(targets_by_index[idx])
                if np.isfinite(target_candidate):
                    target = target_candidate

            if target is None:
                utilities = cfg.get("utilities", {})
                if not utilities:
                    continue
                target = float(np.mean(list(utilities.values())))

            cfgs.append(cfg["config"])
            ys.append(target)
        if len(cfgs) < 2:
            return
        X = self.encoder.encode_batch(cfgs)
        try:
            self.model.fit(X, ys)
            self._fitted = True
        except Exception:
            self._fitted = False

    def predict(self, candidates: list[dict[str, Any]]) -> np.ndarray:
        if (
            self.model is None
            or not candidates
            or not self._fitted
            or not hasattr(self.model, "estimators_")
        ):
            return np.zeros(len(candidates))
        X = self.encoder.encode_batch(candidates)
        try:
            return self.model.predict(X)
        except Exception:
            return np.zeros(len(candidates))

    def is_fitted(self) -> bool:
        return self._fitted

    def compute_proximity_matrix(
        self, cfgs_a: list[dict[str, Any]], cfgs_b: list[dict[str, Any]]
    ) -> np.ndarray:
        """Compute RF leaf co-occurrence proximity between two sets of configs.

        Proximity(a, b) = fraction of trees where a and b land in same leaf.
        Distance = 1 - proximity.

        This handles categorical variables naturally since the RF learned
        which splits matter for prediction.

        Parameters
        ----------
        cfgs_a : list[dict]
            First set of configurations.
        cfgs_b : list[dict]
            Second set of configurations.

        Returns
        -------
        np.ndarray
            (len(cfgs_a), len(cfgs_b)) array of proximities in [0, 1].
        """
        if not self._fitted or not cfgs_a or not cfgs_b:
            return np.zeros((len(cfgs_a), len(cfgs_b)))

        X_a = self.encoder.encode_batch(cfgs_a)
        X_b = self.encoder.encode_batch(cfgs_b)

        # Get leaf indices for each sample in each tree
        leaves_a = self.model.apply(X_a)  # (n_a, n_trees)
        leaves_b = self.model.apply(X_b)  # (n_b, n_trees)

        # Broadcast comparison: same_leaf[i, j, t] = 1 if cfgs_a[i] and cfgs_b[j]
        # land in same leaf in tree t
        same_leaf = leaves_a[:, None, :] == leaves_b[None, :, :]  # (n_a, n_b, n_trees)

        # Proximity = fraction of trees with same leaf
        proximity = same_leaf.mean(axis=2)  # (n_a, n_b)

        return proximity
