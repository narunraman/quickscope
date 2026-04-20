"""
Response Cache for LLM evaluations.

Caches LLM responses for cacheable scenarios to avoid redundant API calls
across runs. Uses snapshot-based semantics: a run reads from cache entries
that existed at start, writes new entries for future runs.
"""

import json
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from collections import defaultdict

from .logging_config import get_logger

if TYPE_CHECKING:
    from quickscope.simulation.abstract.schemas import PromptRequest

logger = get_logger("response_cache")


class ResponseCache:
    """Cache for LLM responses with cross-run semantics.
    
    At initialization, snapshots existing cache entries. During the run:
    - get() returns cached responses from the snapshot (sequentially)
    - put() writes new entries to file (for future runs)
    
    When ``run_id`` is provided, entries tagged with that run_id are excluded
    from the snapshot.  This prevents a resumed run from reading its own
    earlier writes while still benefiting from entries written by other
    concurrent or past runs.
    """
    
    def __init__(self, cache_path: Path, run_id: str | None = None):
        """Initialize cache for a specific scenario/model.
        
        Args:
            cache_path: Path to the JSONL cache file.
            run_id: If provided, cache entries tagged with this run_id are
                    excluded from the snapshot (used during resume to avoid
                    reading the current run's own previous writes).
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        
        # Snapshot existing entries at start (excluding own run_id if set)
        self._snapshot: dict[str, list[dict]] = self._load_existing()
        
        # Track consumption per cache key
        self._consumed: dict[str, int] = defaultdict(int)
        
        # Lock for thread-safe writes
        self._write_lock = threading.Lock()
        
        logger.debug(f"Loaded {sum(len(v) for v in self._snapshot.values())} cached responses")
    
    def _load_existing(self) -> dict[str, list[dict]]:
        """Load existing cache entries into memory.
        
        Entries whose ``run_id`` matches ``self.run_id`` are skipped so that
        a resumed run never reads responses it produced in a prior execution.
        Old entries without a ``run_id`` field are always included.
        """
        entries: dict[str, list[dict]] = defaultdict(list)
        excluded = 0
        
        if not self.cache_path.exists():
            return entries
        
        with open(self.cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    cache_key = entry.get("cache_key")
                    if not cache_key:
                        continue
                    if self.run_id and entry.get("run_id") == self.run_id:
                        excluded += 1
                        continue
                    entries[cache_key].append(entry)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed cache entry")
        
        if excluded:
            logger.info(
                f"Excluded {excluded} cache entries from current run "
                f"(run_id={self.run_id})"
            )
        
        return entries
    
    def get(self, cache_key: str) -> dict | None:
        """Get next cached response for key.
        
        Returns None if no more cached responses are available.
        Ignores entries written by this run.
        """
        available = self._snapshot.get(cache_key, [])
        consumed = self._consumed[cache_key]
        
        if consumed < len(available):
            self._consumed[cache_key] = consumed + 1
            entry = available[consumed]
            logger.debug(f"Cache hit for {cache_key[:16]}... (response {consumed+1}/{len(available)})")
            return entry.get("response")
        
        return None
    
    def put(
        self,
        cache_key: str,
        response: dict,
        config: dict | None = None,
    ) -> None:
        """Write response to cache for future runs.
        
        Appends to file but does NOT update in-memory snapshot.
        Thread-safe: uses lock to serialize writes.
        
        Args:
            cache_key: The hash key for this prompt
            response: The LLM response dict
            config: Optional config dict that generated this prompt (for warm-starting)
        """
        entry = {
            "cache_key": cache_key,
            "response": response,
            "created_at": datetime.now().isoformat(),
        }
        if self.run_id:
            entry["run_id"] = self.run_id
        if config is not None:
            entry["config"] = config
        
        with self._write_lock:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        
        logger.debug(f"Cached response for {cache_key[:16]}...")
    
    def count(self, cache_key: str) -> int:
        """Count available cached responses for a key."""
        return len(self._snapshot.get(cache_key, []))
    
    def remaining(self, cache_key: str) -> int:
        """Count remaining (unconsumed) cached responses for a key."""
        available = len(self._snapshot.get(cache_key, []))
        consumed = self._consumed[cache_key]
        return max(0, available - consumed)

    def get_cached_configs(self) -> list[dict]:
        """Extract unique configs from cache entries.
        
        Returns configs that were stored with cache entries (via config parameter
        to put()). Entries without config are skipped.
        
        Returns:
            List of unique config dicts found in cache
        """
        seen_configs: set[str] = set()
        unique_configs: list[dict] = []
        
        for entries in self._snapshot.values():
            for entry in entries:
                config = entry.get("config")
                if config is None:
                    continue
                
                # Use sorted JSON as dedup key
                config_key = json.dumps(config, sort_keys=True)
                if config_key not in seen_configs:
                    seen_configs.add(config_key)
                    unique_configs.append(config)
        
        logger.debug(f"Found {len(unique_configs)} unique configs in cache")
        return unique_configs

    def get_config_counts(self) -> dict[str, int]:
        """Get count of cached responses per config.
        
        Returns:
            Dict mapping config JSON string → number of cached responses
        """
        counts: dict[str, int] = {}
        
        for entries in self._snapshot.values():
            for entry in entries:
                config = entry.get("config")
                if config is None:
                    continue
                config_key = json.dumps(config, sort_keys=True)
                counts[config_key] = counts.get(config_key, 0) + 1
        
        return counts


def build_cache_key(
    prompt_text: str,
    model_name: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> str:
    """Build a cache key from prompt and config.
    
    Args:
        prompt_text: The prompt text sent to the LLM
        model_name: Model identifier
        temperature: Temperature parameter (None = model default)
        max_tokens: Max tokens parameter (None = model default)
        system_prompt: System prompt if any
    
    Returns:
        SHA256 hash of the key components
    """
    components = {
        "prompt_text": prompt_text,
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
    }
    
    key_str = json.dumps(components, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def build_cache_key_from_prompt(prompt: "PromptRequest") -> str:
    """Convenience wrapper to build cache key from PromptRequest object."""
    return build_cache_key(
        prompt_text=prompt.prompt_text,
        model_name=prompt.model_name,
        temperature=prompt.temperature,
        max_tokens=prompt.max_tokens,
        system_prompt=prompt.system_prompt,
    )

