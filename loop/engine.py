import time
from typing import TYPE_CHECKING, Callable, Optional
from .scratchpad import Scratchpad, Step, ResearchResult
from tools.registry import ToolRegistry
from tools.base import ToolResult
from utils.cost_tracker import log_research_call

if TYPE_CHECKING:
    from pattern.base import BasePattern

class ResearchLoop:
    def __init__(self, llm, registry, settings):
        self.settings = settings
        self.llm = llm
        self.registry = registry

    def run(self, query: str, pattern: "BasePattern", on_hop: Optional[Callable] = None) -> ResearchResult:
        scratchpad = Scratchpad(query)
        self.llm.call_count = 0
        if hasattr(self.llm, "reset_usage_counters"):
            self.llm.reset_usage_counters()

            
        run_start = time.time()          
        sources = []
        api_calls = 0

        max_hops = getattr(self.settings, "max_hops", 6)
        min_hops = getattr(self.settings, "min_hops", 2)

        for hop in range(1, max_hops + 1):
            # Agent thinks
            agent_response = pattern.think(
                query=query,
                scratchpad=scratchpad,
                llm=self.llm,
                registry=self.registry,
            )

            if getattr(agent_response, 'is_final', False) and hop >= min_hops:
                break

            # Run tool
            tool_name = getattr(agent_response, 'tool_name', "web_search")
            resolved_tool = self.registry.get(tool_name)
            if resolved_tool is None:
                resolved_tool = self.registry.get("web_search")
                tool_name = "web_search"
            tool = resolved_tool
            
            #tool = self.registry.get(tool_name) or self.registry.get("web_search")

            tool_input = getattr(agent_response, 'tool_input', query)

            try:
                tool_result = tool.run(
                    query=tool_input,
                    context=scratchpad.context_for_prompt(),
                )
            except Exception as e:
                tool_result = ToolResult(
                    content=f"Tool error: {str(e)}",
                    source=f"{tool_name}: error",
                    confidence=0.4,
                )

            api_calls += 1

            # Post-process
            try:
                tool_result = pattern.post_process(
                    tool_result=tool_result,
                    query=query,
                    scratchpad=scratchpad,
                    llm=self.llm,
                    registry=self.registry,
                )
            except Exception:
                pass

            # Record step (critical for hops)
            step = Step(
                hop_number=hop,
                thought=getattr(agent_response, 'thought', ""),
                tool_used=getattr(tool_result, 'metadata', {}).get("tool_used", tool_name),
                tool_input=tool_input,
                observation=tool_result.content,
                confidence=getattr(tool_result, 'confidence', 0.5),
                pattern=pattern.name,
                grade=getattr(tool_result, 'metadata', {}).get("grade", 0.0),
                corrected=getattr(tool_result, 'metadata', {}).get("corrected", False),
            )
            scratchpad.add_step(step)

            if on_hop is not None:
                try:
                    on_hop(step)
                except Exception:
                    pass

            if tool_result.source:
                sources.append(tool_result.source)

            # Stop condition
            if hop >= min_hops and getattr(tool_result, 'confidence', 0) >= 0.85:
                break

        # Synthesize final answer
        try:
            final_answer = pattern.synthesize(
                query=query,
                scratchpad=scratchpad,
                llm=self.llm,
            )
        except Exception as e:
            final_answer = f"Research completed. {str(e)}"

        sources = list(dict.fromkeys([s for s in sources if s]))
        elapsed = time.time() - run_start

        return ResearchResult(
            query=query,
            final_answer=final_answer,
            pattern_used=pattern.name,
            confidence=scratchpad.last_confidence(),
            hops=scratchpad.hop_count,
            sources=sources,
            api_calls=api_calls,
            steps=scratchpad.steps
        )

        try:
            total_tokens = getattr(self.llm, "total_input_tokens", 0) + getattr(self.llm, "total_output_tokens", 0)
            cost = getattr(self.llm, "total_cost_usd", 0.0)
            log_research_call(
                query=query,
                pattern_used=result.pattern_used,
                hops=result.hops,
                api_calls=result.api_calls,
                confidence=result.confidence,
                elapsed_seconds=elapsed,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
                sources_count=len(sources),
            )
        except Exception:
            pass
 
        return result 
         
