#!/usr/bin/env python3
import asyncio
import math
import sys
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.types as types

# Create a simple MCP server
server = Server("simple-calculator-server")

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
        raise ValueError(f"Invalid expression: {e}") from e

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="calculate",
            description="Performs a mathematical calculation. The input should be a single string representing a valid mathematical expression (e.g., '2 + 2', '10 * (4 - 2)', 'sqrt(16)').",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate."
                    }
                },
                "required": ["expression"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "calculate":
        expression = arguments.get("expression", "")
        try:
            result = _calculate_expression(expression)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main function to run the server."""
    # Import and use the stdio transport
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Simple calculator server shutting down.", file=sys.stderr)
    except Exception as e:
        print(f"Simple calculator server crashed with error: {e}", file=sys.stderr)
        sys.exit(1) 