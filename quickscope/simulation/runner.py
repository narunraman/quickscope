"""
Generic Runner for Quickscope
--------------------------
Executes any Engine by orchestrating LLM calls in a domain-agnostic way.

The Runner:
1. Collects prompts from engines via get_next_prompts()
2. Batches LLM calls for efficiency
3. Routes responses back to engines via update_state()
4. Collects final results via finalize()

Domain-specific logic lives entirely in Engine implementations.
"""

from typing import Any, Callable, TYPE_CHECKING
import threading
import time

from .abstract import (
    BaseEvaluationConfig, 
    Engine,
    LLMResponse, 
    PromptRequest
)
from ..services.llm_service import LLMService
# from ..dataflow.response_cache import ResponseCache, build_cache_key_from_prompt
# from ..dataflow.logging_config import get_logger
from ..dataflow import (
    ResponseCache,
    build_cache_key_from_prompt,
    get_logger,
)

if TYPE_CHECKING:
    import numpy as np
    from .abstract.generator import ScenarioGenerator

runner_logger = get_logger("runner")

class Runner:
    """
    Generic simulation runner.

    Executes scenarios by driving an Engine and making batched LLM calls.
    """

    def __init__(self, llm: LLMService | None = None):
        """
        Initialize runner with LLM service.

        Args:
            llm: LLM service for making API calls. Created if not provided.
        """
        self.llm = llm or LLMService()

    def run_batch(
        self,
        engine: Engine,
        scenarios: list[Any],
        progress_callback: Callable[..., None] | None = None,
        batch_progress_callback: Callable[[int, int], None] | None = None,
        cache: ResponseCache | None = None,
        cache_only: bool = False,
        request_timeout: int | None = None,
        on_scenario_complete: Callable[[str, Any], None] | None = None,
    ) -> list[Any]:
        """
        Run a batch of scenarios using the given engine.

        Args:
            engine: Engine instance to use for execution
            scenarios: List of scenario rows
            progress_callback: Optional callback for round/batch progress updates
            batch_progress_callback: Optional callback(completed, total) for per-request progress
            cache: Optional response cache
            request_timeout: Per-request timeout in seconds (None = no timeout)
            on_scenario_complete: Optional callback(scenario_id, result) called immediately
                                  when a scenario finishes (after its final round).
                                  For single-round scenarios, this fires as each response arrives.
                                  Must be thread-safe.

        Returns:
            List of result objects from engine.finalize()
        """
        if cache_only and cache is None:
            raise ValueError("cache_only=True requires an initialized cache")

        runner_logger.debug("[run_batch] Starting - initializing states")

        # 1. Initialize states for all scenarios
        states: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            state = engine.initialize(scenario)
            states[scenario.scenario_id] = state

        runner_logger.debug(f"[run_batch] Initialized {len(states)} scenario states")

        # 2. Main execution loop
        step = 0
        while True:
            step += 1
            round_start_time = time.perf_counter()
            runner_logger.debug(f"[run_batch] Step {step} - collecting prompts")

            # Collect prompts from all active scenarios
            all_prompts: list[PromptRequest] = []
            prompt_order: list[tuple[str, int]] = []  # (scenario_id, prompt_index)

            active_count = 0
            for scenario in scenarios:
                sid = scenario.scenario_id
                state = states[sid]

                # Skip failed scenarios
                if engine.is_failed(state):
                    continue

                # Get prompts for this scenario
                prompts = engine.get_next_prompts(state)
                if prompts is None:
                    continue  # Scenario is complete

                active_count += 1
                for i, prompt in enumerate(prompts):
                    # Ensure routing info is set
                    prompt.scenario_id = sid
                    all_prompts.append(prompt)
                    prompt_order.append((sid, i))

            # Exit if no active scenarios
            if not all_prompts:
                runner_logger.debug("[run_batch] No active scenarios, exiting loop")
                break

            runner_logger.debug(f"[run_batch] Collected {len(all_prompts)} prompts from {active_count} active scenarios")

            # Calculate max_rounds from scenarios
            max_rounds = max(
                (getattr(s, "num_rounds", 1) for s in scenarios), default=1
            )

            # Progress callback - round_start
            runner_logger.debug("[run_batch] About to call progress_callback(round_start)")
            if progress_callback:
                progress_callback(
                    event="round_start",
                    round_num=step,
                    max_rounds=max_rounds,
                    active_count=active_count,
                )
            runner_logger.debug("[run_batch] progress_callback(round_start) completed")


            # Progress callback - batch_info
            if progress_callback:
                # Count prompts per model
                model_counts: dict[str, int] = {}
                for p in all_prompts:
                    model_counts[p.model_name] = model_counts.get(p.model_name, 0) + 1
                batches = [
                    {"model": model, "count": count}
                    for model, count in model_counts.items()
                ]
                progress_callback(event="batch_info", batches=batches)

            # 3a. Check cache for each prompt
            cache_lookup_start = time.perf_counter()
            cached_responses: list[tuple[int, str, dict]] = []  # (original_idx, text, metadata)
            prompts_needing_llm: list[tuple[int, PromptRequest]] = []  # (original_idx, prompt)
            
            for i, prompt in enumerate(all_prompts):
                if cache:
                    cache_key = build_cache_key_from_prompt(prompt)
                    cached = cache.get(cache_key)
                    if cached:
                        cached_responses.append((i, cached.get("text", ""), cached.get("metadata", {})))
                        runner_logger.debug(f"  [Cache HIT] key={cache_key[:16]}...")
                        continue
                    else:
                        runner_logger.debug(f"  [Cache MISS] key={cache_key[:16]}... (sid={prompt.scenario_id})")
                prompts_needing_llm.append((i, prompt))
            
            # Log cache stats for this batch
            if cache:
                print(f"  [Cache] Hits: {len(cached_responses)}, Misses: {len(prompts_needing_llm)}")
            if progress_callback:
                progress_callback(
                    event="timing_stage",
                    stage="cache_lookup",
                    elapsed_s=time.perf_counter() - cache_lookup_start,
                    cache_hits=len(cached_responses),
                    cache_misses=len(prompts_needing_llm),
                    total_requests=len(all_prompts),
                )
            
            # Update progress for cache hits immediately
            total_requests = len(all_prompts)
            cache_hit_count = len(cached_responses)
            if batch_progress_callback and cache_hit_count > 0:
                batch_progress_callback(cache_hit_count, total_requests)
            
            # 3b. Setup per-scenario tracking for incremental completion
            # Count expected prompts per scenario this round
            prompts_per_scenario: dict[str, int] = {}
            for sid, _ in prompt_order:
                prompts_per_scenario[sid] = prompts_per_scenario.get(sid, 0) + 1
            
            # Track arrived responses per scenario: {sid: {orig_idx: (text, metadata)}}
            scenario_arrivals: dict[str, dict[int, tuple[str, dict]]] = {
                sid: {} for sid in prompts_per_scenario
            }
            
            # Lock for thread-safe access to arrivals and state updates
            arrivals_lock = threading.Lock()
            
            # Helper to process a completed scenario (all prompts arrived)
            def _process_completed_scenario(sid: str) -> None:
                """Process a scenario once all its prompts have arrived."""
                with arrivals_lock:
                    arrivals = scenario_arrivals[sid]
                    expected = prompts_per_scenario[sid]
                    
                    if len(arrivals) < expected:
                        return  # Not all prompts have arrived yet
                    
                    if states[sid].get("_processed_this_round"):
                        return  # Already processed
                    states[sid]["_processed_this_round"] = True
                
                # Build LLMResponse objects for this scenario in correct order
                responses: list[LLMResponse] = []
                for orig_idx in sorted(arrivals.keys()):
                    text, metadata = arrivals[orig_idx]
                    sid_check, prompt_idx = prompt_order[orig_idx]
                    assert sid_check == sid
                    prompt = all_prompts[orig_idx]
                    
                    response = LLMResponse(
                        text=text,
                        metadata=metadata,
                        scenario_id=sid,
                        player_id=prompt.player_id,
                    )
                    responses.append(response)
                
                # Update state with all responses for this scenario
                engine.update_state(states[sid], responses)
                
                # Check if scenario is now complete (no more prompts needed)
                if on_scenario_complete:
                    next_prompts = engine.get_next_prompts(states[sid])
                    if next_prompts is None:
                        # Scenario is complete - finalize and callback immediately
                        result = engine.finalize(states[sid])
                        states[sid]["_finalized"] = True
                        # Handle engines that return multiple results
                        if isinstance(result, list):
                            for r in result:
                                on_scenario_complete(sid, r)
                        else:
                            on_scenario_complete(sid, result)
            
            # Helper to record an arrival and check for completion
            def _record_arrival(orig_idx: int, text: str, metadata: dict) -> None:
                """Record a response arrival and process scenario if complete."""
                sid, _ = prompt_order[orig_idx]
                with arrivals_lock:
                    scenario_arrivals[sid][orig_idx] = (text, metadata)
                    arrived = len(scenario_arrivals[sid])
                    expected = prompts_per_scenario[sid]
                
                if arrived == expected:
                    _process_completed_scenario(sid)
            
            # 3c. Process cached responses immediately
            for orig_idx, text, metadata in cached_responses:
                metadata = dict(metadata)
                metadata["cache_hit"] = True
                # Write to cache tracking (not needed for cached, but keep consistent)
                _record_arrival(orig_idx, text, metadata)
            
            # 3d. Batch LLM call for uncached prompts only
            if prompts_needing_llm:
                if cache_only:
                    # Skip LLM calls; mark cache misses as empty responses.
                    if batch_progress_callback:
                        batch_progress_callback(
                            cache_hit_count + len(prompts_needing_llm), total_requests
                        )
                    for orig_idx, _ in prompts_needing_llm:
                        _record_arrival(orig_idx, "", {"cache_miss": True})
                else:
                    llm_batch_start = time.perf_counter()
                    first_response_elapsed: float | None = None
                    request_completion_markers = {
                        marker
                        for marker in {
                            1,
                            max(1, len(prompts_needing_llm) // 4),
                            max(1, len(prompts_needing_llm) // 2),
                            max(1, (3 * len(prompts_needing_llm)) // 4),
                            len(prompts_needing_llm),
                        }
                        if 1 <= marker <= len(prompts_needing_llm)
                    }
                    completed_request_count = [0]
                    configs = [
                        BaseEvaluationConfig(
                            model_name=p.model_name,
                            temperature=p.temperature,
                            max_tokens=p.max_tokens,
                            system_prompt=p.system_prompt,
                            logprobs=p.logprobs,
                            top_logprobs=p.top_logprobs,
                        )
                        for _, p in prompts_needing_llm
                    ]
                    prompt_texts = [p.prompt_text for _, p in prompts_needing_llm]
                    
                    # Progress callback wrapper
                    def llm_status_callback(completed: int, total: int) -> None:
                        if batch_progress_callback:
                            batch_progress_callback(cache_hit_count + completed, total_requests)
                    
                    # Per-request callback for incremental processing
                    def per_request_callback(llm_idx: int, result: tuple[str, dict | None]) -> None:
                        """Called as each LLM response arrives."""
                        nonlocal first_response_elapsed
                        orig_idx, prompt = prompts_needing_llm[llm_idx]
                        text, metadata = result
                        metadata = metadata or {}
                        completed_request_count[0] += 1
                        elapsed_since_batch_start = time.perf_counter() - llm_batch_start
                        if first_response_elapsed is None:
                            first_response_elapsed = elapsed_since_batch_start
                            if progress_callback:
                                progress_callback(
                                    event="timing_stage",
                                    stage="first_response",
                                    elapsed_s=first_response_elapsed,
                                    completed=completed_request_count[0],
                                    total=len(prompts_needing_llm),
                                )
                        if (
                            progress_callback
                            and completed_request_count[0] in request_completion_markers
                        ):
                            progress_callback(
                                event="request_progress",
                                completed=completed_request_count[0],
                                total=len(prompts_needing_llm),
                                elapsed_s=elapsed_since_batch_start,
                                scenario_id=prompt.scenario_id,
                                model_name=prompt.model_name,
                                timed_out=bool(metadata.get("timed_out")),
                                failed=bool(metadata.get("failed") or metadata.get("error")),
                            )
                        
                        # Cache the response (skip errors)
                        if cache and not metadata.get("error"):
                            cache_key = build_cache_key_from_prompt(prompt)
                            cache.put(
                                cache_key,
                                {"text": text, "metadata": metadata},
                                config=prompt.source_config,
                            )
                            runner_logger.debug(
                                f"  [Cache] Wrote new response for key {cache_key[:16]}..."
                            )
                        
                        # Record arrival and check for scenario completion
                        _record_arrival(orig_idx, text, metadata)
                    
                    # Call generate_batch with per-request callback
                    # (responses are processed via callback, return value not needed)
                    runner_logger.debug(f"[run_batch] Calling llm.generate_batch with {len(prompt_texts)} prompts")
                    self.llm.generate_batch(
                        prompt_texts,
                        configs,
                        status_callback=llm_status_callback,
                        per_request_callback=per_request_callback,
                        request_timeout=request_timeout,
                    )
                    runner_logger.debug("[run_batch] llm.generate_batch completed")
                    if progress_callback:
                        progress_callback(
                            event="timing_stage",
                            stage="llm_batch",
                            elapsed_s=time.perf_counter() - llm_batch_start,
                            completed=completed_request_count[0],
                            total=len(prompts_needing_llm),
                            first_response_s=first_response_elapsed,
                        )
            
            # 3e. Clear per-round tracking flags for next round
            for sid in prompts_per_scenario:
                states[sid].pop("_processed_this_round", None)
            
            # 4. Build scenario_responses for any scenarios not processed via callback
            # (This handles multi-round scenarios where callback wasn't triggered)
            scenario_responses: dict[str, list[LLMResponse]] = {}
            for sid, arrivals in scenario_arrivals.items():
                if states[sid].get("_finalized"):
                    continue  # Already finalized via callback
                if len(arrivals) < prompts_per_scenario[sid]:
                    continue  # Not all prompts arrived (shouldn't happen)
                
                # Build responses if not already processed
                if not states[sid].get("_processed_this_round"):
                    responses: list[LLMResponse] = []
                    for orig_idx in sorted(arrivals.keys()):
                        text, metadata = arrivals[orig_idx]
                        prompt = all_prompts[orig_idx]
                        response = LLMResponse(
                            text=text,
                            metadata=metadata,
                            scenario_id=sid,
                            player_id=prompt.player_id,
                        )
                        responses.append(response)
                    scenario_responses[sid] = responses

            # 5. Update states for any scenarios not processed via callback
            for sid, responses in scenario_responses.items():
                if not states[sid].get("_finalized"):
                    engine.update_state(states[sid], responses)

            if progress_callback:
                progress_callback(
                    event="timing_stage",
                    stage="round_total",
                    elapsed_s=time.perf_counter() - round_start_time,
                    round_num=step,
                    total_requests=len(all_prompts),
                    active_count=active_count,
                )

        # 6. Finalize all scenarios (skip already-finalized ones)
        results = []
        for scenario in scenarios:
            sid = scenario.scenario_id
            if states[sid].get("_finalized"):
                # Already finalized via callback - retrieve from state or skip
                # We need to re-finalize to get the result for the return value
                # but the callback has already been called
                pass
            result = engine.finalize(states[sid])
            # Handle engines that return multiple results.
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

        return results

    def run_from_params(
        self,
        engine: Engine,
        generator: "ScenarioGenerator",
        suggestions: list[dict],
        rng: "np.random.Generator | None" = None,
        progress_callback: Callable[..., None] | None = None,
        batch_progress_callback: Callable[[int, int], None] | None = None,
        cache: ResponseCache | None = None,
        cache_only: bool = False,
        request_timeout: int | None = None,
        on_scenario_complete: Callable[[str, Any, dict], None] | None = None,
    ) -> list[dict]:
        """
        Generate scenarios from suggestions and run them.
        
        This is the preferred entry point - generator is called internally.
        
        Args:
            engine: Engine instance to use for execution
            generator: Scenario generator to create scenarios from params
            suggestions: List of dicts with {ticket_id, params}
            rng: Random generator for reproducibility
            progress_callback: Optional callback for round/batch progress updates
            batch_progress_callback: Optional callback(completed, total) for per-request progress
            cache: Optional response cache for avoiding redundant LLM calls
            request_timeout: Per-request timeout in seconds (None = no timeout)
            on_scenario_complete: Optional callback(scenario_id, result, suggestion_info) called
                                  immediately when a scenario finishes. suggestion_info contains
                                  {ticket_id, params, scenario}. For single-round scenarios,
                                  this fires as each response arrives. Must be thread-safe.
            
        Returns:
            List of dicts with {ticket_id, params, scenario, result}
        """
        # Generate scenarios and track ticket mapping
        generation_start_time = time.perf_counter()
        scenarios = []
        scenario_to_suggestion: dict[str, dict] = {}
        
        for i, suggestion in enumerate(suggestions):
            ticket_id = suggestion["ticket_id"]
            params = suggestion["params"]
            runner_logger.debug(f"[run_from_params] Generating scenario {i+1}/{len(suggestions)} (ticket={ticket_id})")
            scenario = generator.generate(params, rng=rng)
            runner_logger.debug(f"[run_from_params] Scenario {i+1} generated: {scenario.scenario_id}")
            scenarios.append(scenario)
            scenario_to_suggestion[scenario.scenario_id] = {
                "ticket_id": ticket_id,
                "params": params,
                "scenario": scenario,
                "optimizer_meta": suggestion.get("optimizer_meta"),
            }
        if progress_callback:
            progress_callback(
                event="timing_stage",
                stage="scenario_generation",
                elapsed_s=time.perf_counter() - generation_start_time,
                total_scenarios=len(scenarios),
            )
        
        runner_logger.debug(f"[run_from_params] All {len(scenarios)} scenarios generated, calling run_batch()")

        # Create wrapper callback that includes suggestion info
        wrapped_callback: Callable[[str, Any], None] | None = None
        if on_scenario_complete:
            def _make_callback(
                callback: Callable[[str, Any, dict], None],
                mapping: dict[str, dict],
            ) -> Callable[[str, Any], None]:
                def wrapper(sid: str, result: Any) -> None:
                    suggestion_info = mapping.get(sid, {})
                    callback(sid, result, suggestion_info)
                return wrapper
            wrapped_callback = _make_callback(on_scenario_complete, scenario_to_suggestion)
        
        # Run using existing method
        results = self.run_batch(
            engine, scenarios, progress_callback, 
            batch_progress_callback=batch_progress_callback, 
            cache=cache,
            cache_only=cache_only,
            request_timeout=request_timeout,
            on_scenario_complete=wrapped_callback,
        )
        
        # Attach ticket_id and params to each result
        output = []
        for result in results:
            sid = result.scenario_id
            suggestion_info = scenario_to_suggestion.get(sid, {})
            output.append({
                "ticket_id": suggestion_info.get("ticket_id"),
                "params": suggestion_info.get("params"),
                "scenario": suggestion_info.get("scenario"),
                "result": result,
            })
        
        return output
