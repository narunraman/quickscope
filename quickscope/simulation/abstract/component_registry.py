"""
Component Registry for Quickscope Simulation
-----------------------------------------
Registry for scenario-specific components (engines, generators, handlers).
Components are registered via decorators and looked up by scenario name.

Fallback behavior:
- Engine: falls back to DefaultEngine if not registered
- Handler: falls back to DefaultResponseHandler if not registered  
- Generator: None if not registered (signals file-based scenario)

Auto-discovery:
- When get() is called for an unregistered scenario, the registry attempts
  to import quickscope.simulation.<scenario> to trigger decorator registration.
"""

import importlib
from typing import Type, Callable, Any


class ComponentRegistry:
    """
    Registry for simulation components by scenario name.
    
    Usage:
        @ComponentRegistry.engine("custom")
        class CustomEngine: ...
        
        @ComponentRegistry.generator("dyval")
        class DyValGenerator: ...
        
        @ComponentRegistry.handler("custom")  # Optional
        class CustomHandler: ...
        
        components = ComponentRegistry.get("dyval")
        engine = components["engine"]()  # DefaultEngine (fallback)
        generator = components["generator"]  # DyValGenerator
    """
    
    _engines: dict[str, Type] = {}
    _generators: dict[str, Type] = {}
    _handlers: dict[str, Type] = {}
    _metrics: dict[str, Callable] = {}
    
    
    @classmethod
    def engine(cls, name: str) -> Callable[[Type], Type]:
        """Decorator to register an engine class."""
        def decorator(klass: Type) -> Type:
            cls._engines[name] = klass
            return klass
        return decorator
    
    @classmethod
    def generator(cls, name: str) -> Callable[[Type], Type]:
        """Decorator to register a generator class."""
        def decorator(klass: Type) -> Type:
            cls._generators[name] = klass
            return klass
        return decorator
    
    @classmethod
    def handler(cls, name: str) -> Callable[[Type], Type]:
        """Decorator to register a response handler class."""
        def decorator(klass: Type) -> Type:
            cls._handlers[name] = klass
            return klass
        return decorator
    

    @classmethod
    def metrics(cls, name: str) -> Callable[[Callable], Callable]:
        """Decorator to register a metrics function."""
        def decorator(fn: Callable) -> Callable:
            cls._metrics[name] = fn
            return fn
        return decorator
    
    @classmethod
    def get(cls, scenario: str) -> dict[str, Any]:
        """
        Get components for a scenario with fallbacks.
        
        Auto-discovers scenario modules by attempting to import:
        - quickscope.simulation.<scenario> (package with __init__.py)
        - quickscope.simulation.<scenario>.<scenario>_generator (generator module)
        
        Returns dict with:
            - engine: Engine class (falls back to DefaultEngine)
            - generator: Generator class (None if not registered = file-based)
            - handler: Handler class (falls back to DefaultResponseHandler)
        """
        # Auto-discover: try importing scenario module if not already registered
        cls._try_import_scenario(scenario)
        
        # Engine: use registered or fallback to default
        engine = cls._engines.get(scenario)
        if engine is None:
            if "default" not in cls._engines:
                raise ValueError(
                    f"No engine for '{scenario}' and no 'default' fallback registered."
                )
            engine = cls._engines["default"]
        
        # Handler: use registered or fallback to DefaultResponseHandler
        handler = cls._handlers.get(scenario)
        if handler is None:
            if "default" not in cls._handlers:
                raise ValueError(
                    f"No handler for '{scenario}' and no 'default' fallback registered."
                )
            handler = cls._handlers["default"]
        
        return {
            "engine": engine,
            "generator": cls._generators.get(scenario),
            "handler": handler,
        }
    
    @classmethod
    def _try_import_scenario(cls, scenario: str) -> None:
        """Try to import scenario module to trigger decorator registration.
        
        Attempts to import in order:
        1. quickscope.simulation.<scenario> (package __init__.py)
        2. quickscope.simulation.<scenario>.<scenario>_generator
        3. quickscope.simulation.<scenario>.generator
        4. quickscope.simulation.<scenario>.engine
        """
        # Skip if already registered
        if scenario in cls._generators or scenario in cls._engines:
            return
        
        # Skip special names
        if scenario in ("default", "abstract", "base"):
            return
        
        # Try importing various module patterns
        module_patterns = [
            f"quickscope.simulation.{scenario}",
            f"quickscope.simulation.{scenario}.{scenario}_generator",
            f"quickscope.simulation.{scenario}.generator",
            f"quickscope.simulation.{scenario}.engine",
        ]
        
        last_error = None
        for module_name in module_patterns:
            try:
                importlib.import_module(module_name)
                if scenario in cls._generators:
                    return
            except ImportError as e:
                last_error = e
                continue
            except Exception as e:
                raise ImportError(
                    f"Failed to import scenario '{scenario}' "
                    f"(module {module_name}): {e}"
                ) from e
    
    @classmethod
    def get_engine(cls, name: str) -> Any:
        """Get an engine instance with appropriate handler injected."""
        # Get engine class (with fallback to default)
        engine_cls = cls._engines.get(name)
        if engine_cls is None:
            engine_cls = cls._engines.get("default")
            if engine_cls is None:
                raise ValueError(f"No engine for '{name}' and no 'default' fallback")
        
        # Get handler class (with fallback to default)
        handler_cls = cls.get_handler(name)
        handler = handler_cls()
        
        # Instantiate engine with handler
        return engine_cls(handler=handler)

    @classmethod
    def get_handler(cls, scenario: str) -> Type:
        """Get handler class for scenario, falling back to default."""
        handler = cls._handlers.get(scenario)
        if handler is None:
            handler = cls._handlers.get("default")
            if handler is None:
                raise ValueError(f"No handler for '{scenario}' and no 'default' fallback")
        return handler
    

    @classmethod
    def get_metrics(cls, scenario: str) -> Callable:
        """Get metrics function, falling back to default."""
        # Try to import scenario module to trigger decorator registration
        cls._try_import_scenario(scenario)
        
        fn = cls._metrics.get(scenario)
        if fn is None:
            fn = cls._metrics.get("default")
            if fn is None:
                raise ValueError(f"No metrics for '{scenario}' and no 'default' fallback")
        return fn
    
    @classmethod
    def list_scenarios(cls) -> list[str]:
        """List all registered scenario names (engines + generators)."""
        return list(set(cls._engines.keys()) | set(cls._generators.keys()))
    
    @classmethod
    def has_engine(cls, name: str) -> bool:
        """Check if an engine is registered for a scenario."""
        return name in cls._engines
    
    @classmethod
    def has_generator(cls, name: str) -> bool:
        """Check if a generator is registered for a scenario."""
        return name in cls._generators
