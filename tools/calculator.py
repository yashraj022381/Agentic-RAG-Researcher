from .base import BaseTool, ToolResult

class CalculatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Perform arithmetic calculations (percentages, differences, ratios) "
            "on specific numeric values already found in prior research steps. "
            "Input should be a plain arithmetic expression, e.g. '(853.1 - 800) / 800 * 100'."
        )

    def run(self, query: str, context=None) -> ToolResult:
        import re
        # Only allow digits, operators, parentheses, decimal points — never eval() raw input
        if not re.fullmatch(r'[\d\s\.\+\-\*\/\(\)%]+', query.strip()):
            return ToolResult(
                content="Invalid expression — only numbers and + - * / ( ) are allowed.",
                source="Calculator: rejected",
                confidence=0.1,
            )
        try:
            result = eval(query, {"__builtins__": {}}, {})
            return ToolResult(
                content=f"Calculation result: {query} = {result}",
                source="Calculator: computed",
                confidence=0.95,
            )
        except Exception as e:
            return ToolResult(
                content=f"Calculation failed: {e}",
                source="Calculator: error",
                confidence=0.2,
            )
