"""
OP 7F: Strategy Plugin Architecture
====================================
Standard interface for all AI strategy/model plugins.

Any model that wants to participate in the ensemble MUST implement
the StrategyPlugin protocol. The StrategyRegistry discovers, loads,
and manages plugin lifecycle.

Usage:
    # Define a plugin
    class MyModelAgent(BaseAgent):
        name = "my_model"
        version = "1.0.0"
        model_type = "tree"
        def predict(self, symbol: str, features: dict) -> StrategyPrediction: ...
        def get_required_features(self) -> list[str]: ...
        def health_check(self) -> bool: ...
        def get_metadata(self) -> dict: ...

    # Register it
    registry = StrategyRegistry()
    registry.register("my_model", MyModelAgent)

    # Or auto-discover from a package
    registry.discover("ai_engine.agents")
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class StrategyPrediction:
    """Standardised prediction returned by every strategy plugin."""
    symbol: str
    action: str              # "BUY" | "SELL" | "HOLD"
    confidence: float        # 0.0 – 1.0
    confidence_std: float    # uncertainty estimate
    version: str             # model version string
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Legacy compat — return the dict format EnsembleManager expects."""
        d = {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "confidence_std": self.confidence_std,
            "version": self.version,
        }
        d.update(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyPrediction":
        """Create a StrategyPrediction from a legacy dict."""
        known = {"symbol", "action", "confidence", "confidence_std", "version"}
        meta = {k: v for k, v in d.items() if k not in known}
        return cls(
            symbol=d.get("symbol", ""),
            action=d.get("action", "HOLD"),
            confidence=float(d.get("confidence", 0.0)),
            confidence_std=float(d.get("confidence_std", 0.1)),
            version=d.get("version", "unknown"),
            metadata=meta,
        )


@runtime_checkable
class StrategyPlugin(Protocol):
    """
    Protocol that every ensemble strategy plugin must satisfy.

    Implementations can be plain classes — no inheritance required.
    The StrategyRegistry validates conformance at registration time
    via isinstance(obj, StrategyPlugin).
    """

    name: str               # short key, e.g. "xgb", "nhits"
    version: str            # model version
    model_type: str         # "tree", "neural", "rl", "rule"

    def predict(self, symbol: str, features: dict) -> dict:
        """Return a prediction dict (or StrategyPrediction.to_dict())."""
        ...

    def get_required_features(self) -> list[str]:
        """Feature names this model expects (empty = accepts any)."""
        ...

    def health_check(self) -> bool:
        """Return True if model is loaded and ready."""
        ...

    def get_metadata(self) -> dict:
        """Arbitrary metadata (model path, params, etc.)."""
        ...


class StrategyRegistry:
    """
    Central registry for strategy plugins.

    Supports:
      - Explicit registration: registry.register("xgb", XGBoostAgent)
      - Instance registration: registry.register_instance("xgb", agent_obj)
      - Package discovery:     registry.discover("ai_engine.agents")
      - Lifecycle:             registry.get / list / unregister / health
    """

    # Class-level attribute marking discoverable plugins
    PLUGIN_MARKER = "__strategy_plugin__"

    def __init__(self):
        self._factories: Dict[str, type] = {}
        self._instances: Dict[str, Any] = {}  # lazy or pre-built
        self._weights: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, cls: type, weight: float = 0.0) -> None:
        """Register a plugin class (instantiated lazily on first .get())."""
        if name in self._factories:
            logger.warning(f"[Registry] Overwriting factory for '{name}'")
        self._factories[name] = cls
        if weight > 0:
            self._weights[name] = weight
        logger.info(f"[Registry] Registered factory: {name} -> {cls.__name__}")

    def register_instance(self, name: str, instance: Any, weight: float = 0.0) -> None:
        """Register a pre-built plugin instance."""
        self._instances[name] = instance
        if weight > 0:
            self._weights[name] = weight
        logger.info(f"[Registry] Registered instance: {name} ({type(instance).__name__})")

    def unregister(self, name: str) -> None:
        self._factories.pop(name, None)
        self._instances.pop(name, None)
        self._weights.pop(name, None)
        logger.info(f"[Registry] Unregistered: {name}")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, package_name: str) -> int:
        """
        Import every module in *package_name* and register classes
        that are marked with ``__strategy_plugin__ = True``.

        Returns the number of newly registered plugins.
        """
        count = 0
        try:
            pkg = importlib.import_module(package_name)
        except ImportError:
            logger.warning(f"[Registry] Cannot import package '{package_name}'")
            return 0

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            # single module, not a package
            count += self._scan_module(pkg)
            return count

        for importer, modname, ispkg in pkgutil.walk_packages(
            pkg_path, prefix=package_name + "."
        ):
            try:
                mod = importlib.import_module(modname)
                count += self._scan_module(mod)
            except Exception as exc:
                logger.debug(f"[Registry] Skipping {modname}: {exc}")
        return count

    def _scan_module(self, mod) -> int:
        found = 0
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                inspect.isclass(obj)
                and getattr(obj, self.PLUGIN_MARKER, False) is True
            ):
                key = getattr(obj, "name", attr_name.lower())
                if key not in self._factories:
                    self.register(key, obj)
                    found += 1
        return found

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str, **kwargs) -> Optional[Any]:
        """
        Return the plugin instance for *name*.

        If only a factory was registered, instantiate it now (with **kwargs).
        Returns None if the plugin is unknown or instantiation fails.
        """
        if name in self._instances:
            return self._instances[name]

        cls = self._factories.get(name)
        if cls is None:
            return None

        try:
            instance = cls(**kwargs) if kwargs else cls()
            self._instances[name] = instance
            return instance
        except Exception as exc:
            logger.error(f"[Registry] Failed to instantiate '{name}': {exc}")
            return None

    def get_all(self) -> Dict[str, Any]:
        """Return all registered instances (instantiating factories as needed)."""
        all_names = set(self._factories) | set(self._instances)
        return {n: self.get(n) for n in all_names if self.get(n) is not None}

    def list_names(self) -> List[str]:
        return sorted(set(self._factories) | set(self._instances))

    def get_weight(self, name: str) -> float:
        return self._weights.get(name, 0.0)

    def set_weight(self, name: str, w: float) -> None:
        self._weights[name] = w

    def get_weights(self) -> Dict[str, float]:
        """Return weights for all registered plugins."""
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, bool]:
        """Run health_check() on every instantiated plugin."""
        result: Dict[str, bool] = {}
        for name, inst in self._instances.items():
            try:
                hc = getattr(inst, "health_check", None)
                result[name] = hc() if callable(hc) else True
            except Exception:
                result[name] = False
        return result

    def __len__(self) -> int:
        return len(set(self._factories) | set(self._instances))

    def __contains__(self, name: str) -> bool:
        return name in self._factories or name in self._instances

    def __repr__(self) -> str:
        names = self.list_names()
        return f"StrategyRegistry({len(names)} plugins: {names})"
