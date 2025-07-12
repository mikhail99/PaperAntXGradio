import math
import sys
from mcp.server.fastmcp import FastMCP

# Use the high-level FastMCP helper for robust server creation
mcp = FastMCP("standard-calculator-server")

def _calculate_expression(expression: str) -> str:
    """The actual implementation of the calculator."""
    try:
        # A safer eval environment
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names["abs"] = abs
        
        if len(expression) > 200:
            raise ValueError("Expression is too long.")
            
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        # Raise a ValueError, which FastMCP will convert into a proper tool error response.
        raise ValueError(f"Invalid expression: {e}") from e

@mcp.tool()
async def calculate(expression: str) -> str:
    """
    Performs a mathematical calculation. The input should be a single string 
    representing a valid mathematical expression (e.g., '2 + 2', '10 * (4 - 2)', 'sqrt(16)').
    """
    return _calculate_expression(expression)

if __name__ == "__main__":
    import asyncio
    import platform
    
    try:
        # Fix for macOS asyncio issues - run in main thread
        if platform.system() == "Darwin":  # macOS
            # Create a new event loop and set it as the main loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # The run method handles the server lifecycle, including initialization and transport.
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("Calculator server shutting down.")
    except Exception as e:
        print(f"Calculator server crashed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
