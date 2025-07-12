#!/usr/bin/env python3
import asyncio
import os
import sys
from io import StringIO
import pandas as pd
from mcp.server import Server
from mcp.types import Tool, TextContent

# Create a simple MCP server
server = Server("simple-csv-server")

# This will hold the loaded dataframe
state = {"dataframe": None, "filename": None}

def _ensure_df_loaded():
    """Helper function to check if a DataFrame is loaded."""
    if state["dataframe"] is None:
        raise ValueError("No CSV file loaded. Please use 'load_csv' first.")
    return state["dataframe"]

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="load_csv",
            description="Loads a CSV file from the local filesystem into memory for querying. Expands user home directory `~`.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The local path to the CSV file (e.g., '~/Downloads/data.csv')."
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="get_csv_headers",
            description="Returns the column headers of the currently loaded CSV file as a comma-separated string.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="show_head",
            description="Displays the first few rows of the loaded CSV file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "integer",
                        "description": "The number of rows to display from the top of the file. Defaults to 5."
                    }
                }
            }
        ),
        Tool(
            name="get_summary",
            description="Provides a summary of the loaded CSV file, including column data types, non-null counts, and descriptive statistics for numerical columns.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="query_data",
            description="Queries the loaded CSV data using a pandas query expression. Example: 'Age > 30', '`Country` == \"USA\" and `Salary` > 50000'. Column names with spaces or special characters should be enclosed in backticks ``.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_expression": {
                        "type": "string",
                        "description": "A string containing the pandas query expression."
                    }
                },
                "required": ["query_expression"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "load_csv":
            file_path = arguments.get("file_path", "")
            expanded_path = os.path.expanduser(file_path)
            if not os.path.exists(expanded_path):
                return [TextContent(type="text", text=f"File not found at path: {expanded_path}")]
            
            df = pd.read_csv(expanded_path)
            state["dataframe"] = df
            state["filename"] = os.path.basename(expanded_path)
            return [TextContent(type="text", text=f"Successfully loaded '{state['filename']}' with {len(df)} rows and {len(df.columns)} columns.")]
            
        elif name == "get_csv_headers":
            df = _ensure_df_loaded()
            return [TextContent(type="text", text=", ".join(df.columns))]
            
        elif name == "show_head":
            df = _ensure_df_loaded()
            rows = arguments.get("rows", 5)
            return [TextContent(type="text", text=df.head(rows).to_string())]
            
        elif name == "get_summary":
            df = _ensure_df_loaded()
            info_buffer = StringIO()
            df.info(buf=info_buffer)
            info_str = info_buffer.getvalue()

            numeric_cols = df.select_dtypes(include='number')
            if not numeric_cols.empty:
                describe_str = numeric_cols.describe().to_string()
            else:
                describe_str = "No numerical columns to describe."
            
            result = f"File: {state['filename']}\n\n=== Dataframe Info ===\n{info_str}\n\n=== Descriptive Statistics (for numerical columns) ===\n{describe_str}"
            return [TextContent(type="text", text=result)]
            
        elif name == "query_data":
            df = _ensure_df_loaded()
            query_expression = arguments.get("query_expression", "")
            try:
                result_df = df.query(query_expression, engine='numexpr')
                
                if result_df.empty:
                    return [TextContent(type="text", text="Query returned no results.")]
                
                if len(result_df) > 50:
                    result = f"Query returned {len(result_df)} rows. Showing first 50.\n\n{result_df.head(50).to_string()}"
                else:
                    result = result_df.to_string()
                    
                return [TextContent(type="text", text=result)]
            except Exception as e:
                if "undefined variable" in str(e).lower():
                    return [TextContent(type="text", text=f"Error: Invalid query. One of the column names might be misspelled or needs backticks for special characters (e.g., `Column Name`). Details: {e}")]
                else:
                    return [TextContent(type="text", text=f"Invalid query expression: {e}")]
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        return [TextContent(type="text", text=f"Error executing tool {name}: {e}")]

async def main():
    """Main function to run the server."""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Simple CSV server shutting down.", file=sys.stderr)
    except Exception as e:
        print(f"Simple CSV server crashed with error: {e}", file=sys.stderr)
        sys.exit(1) 