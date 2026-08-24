from typing import Optional, List
#from .base import BaseTool, ToolResult
from .base import BaseTool, ToolResult

class SynthesizerTool(BaseTool):

    @property
    def name(self) -> str:
        return "synthesizer"

    @property
    def description(self) -> str:
        return (
            "Combine and synthesize findings from multiple sources/hops "
            "into a single coherent cited answer. "
            "Call this as the LAST step when you have enough evidence."
        )

    def run(self, query: str, context: Optional[str] = None) -> ToolResult:
        if not context:
            return ToolResult(
                content = "No context provided to synthesize. Gather information first. ",
                source = "Synthesizer",
                confidence = 0.1,
            )
        
        synthesized = (
            f"Synthesized answer for: '{query}'\n\n"
            f"{context}\n\n"
            f"[Synthesized from {context.count('Source:') + context.count('KB') + 1} sources]"
        )
        
        return ToolResult(
            content = synthesized,
            source = "Synthesizer: multi-source merge",
            confidence = 0.82,
            metadata = {"synthesis": True},
        )

    def think(self, query: str, scratchpad, llm, registry):
        pass  # Not used for synthesizer

    def post_process(self, tool_result, query, scratchpad, llm, registry):
        return tool_result

    def synthesize(self, query: str, scratchpad, llm) -> str:
        """Generate clean final answer."""
        context = scratchpad.all_observations()

        prompt = f"""
        Create a clean, professional final answer.

        Question: {query}

        Research Context:
        {context}

        Rules:
        - Start directly with the answer
        - Remove ALL tags: <thought>, <action>, <final_answer>, etc.
        - Use natural language
        - Use bullet points for comparisons or lists
        - Be concise and confident
        """

        answer = llm.chat(
            system="You are a clear, confident research assistant.",
            user=prompt
        )

        answer = answer.replace('{"tool"}', '').replace('{tool}', '').replace('[{tool}]', '').strip()

        for tag in ["<final_answer>", "</final_answer>", "<thought>", "</thought>",
                    "<action>", "</action>", "<thought/>", "<action/>"]:
            answer = answer.replace(tag, "")

        answer = re.sub(r'\{["\']?tool["\']?\}', '', answer)
        answer = re.sub(r'\[\s*tool\s*\]', '', answer)
        return answer.strip()
