import os
import sys
from io import StringIO
import pandas as pd
from mcp.server.fastmcp import FastMCP

# This will hold the loaded dataframe. A class-based approach would be better for
# handling multiple files or sessions, but a simple dict is fine for this example.
state = {"dataframe": None, "filename": None}

mcp = FastMCP("csv-assistant-server")

@mcp.tool()
async def load_csv(file_path: str) -> str:
    """
    Loads a CSV file from the local filesystem into memory for querying.
    Expands user home directory `~`.
    
    Args:
        file_path: The local path to the CSV file (e.g., '~/Downloads/data.csv').
    """
    expanded_path = os.path.expanduser(file_path)
    if not os.path.exists(expanded_path):
        raise ValueError(f"File not found at path: {expanded_path}")
    
    try:
        df = pd.read_csv(expanded_path)
        state["dataframe"] = df
        state["filename"] = os.path.basename(expanded_path)
        return f"Successfully loaded '{state['filename']}' with {len(df)} rows and {len(df.columns)} columns."
    except Exception as e:
        raise ValueError(f"Failed to load or parse CSV file: {e}") from e

def _ensure_df_loaded():
    """Helper function to check if a DataFrame is loaded."""
    if state["dataframe"] is None:
        raise ValueError("No CSV file loaded. Please use 'load_csv' first.")
    return state["dataframe"]

@mcp.tool()
async def get_csv_headers() -> str:
    """
    Returns the column headers of the currently loaded CSV file as a comma-separated string.
    """
    df = _ensure_df_loaded()
    return ", ".join(df.columns)

@mcp.tool()
async def show_head(rows: int = 5) -> str:
    """
    Displays the first few rows of the loaded CSV file.
    
    Args:
        rows: The number of rows to display from the top of the file. Defaults to 5.
    """
    df = _ensure_df_loaded()
    return df.head(rows).to_string()

@mcp.tool()
async def get_summary() -> str:
    """
    Provides a summary of the loaded CSV file, including column data types, non-null counts, and descriptive statistics for numerical columns.
    """
    df = _ensure_df_loaded()
    
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    info_str = info_buffer.getvalue()

    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        describe_str = numeric_cols.describe().to_string()
    else:
        describe_str = "No numerical columns to describe."
    
    return f"File: {state['filename']}\n\n=== Dataframe Info ===\n{info_str}\n\n=== Descriptive Statistics (for numerical columns) ===\n{describe_str}"


@mcp.tool()
async def query_data(query_expression: str) -> str:
    """
    Queries the loaded CSV data using a pandas query expression.
    Example: 'Age > 30', '`Country` == "USA" and `Salary` > 50000'.
    Column names with spaces or special characters should be enclosed in backticks ``.
    
    Args:
        query_expression: A string containing the pandas query expression.
    """
    df = _ensure_df_loaded()
    try:
        # Using the 'numexpr' engine which is generally safer and faster.
        result_df = df.query(query_expression, engine='numexpr')
        
        if result_df.empty:
            return "Query returned no results."
        
        # Limit output size to prevent flooding the context
        if len(result_df) > 50:
             return f"Query returned {len(result_df)} rows. Showing first 50.\n\n{result_df.head(50).to_string()}"
             
        return result_df.to_string()
    except Exception as e:
        if "undefined variable" in str(e).lower():
             return f"Error: Invalid query. One of the column names might be misspelled or needs backticks for special characters (e.g., `Column Name`). Details: {e}"
        raise ValueError(f"Invalid query expression: {e}") from e

if __name__ == "__main__":
    try:
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("CSV server shutting down.")
    except Exception as e:
        print(f"CSV server crashed with error: {e}", file=sys.stderr)
        sys.exit(1)
