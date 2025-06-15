import sys
import os
import asyncio
import atexit
import threading
from typing import Dict, Optional, Any, Coroutine
from contextlib import AsyncExitStack
from concurrent.futures import Future

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPServerManager:
    def __init__(self):
        self.server_configs: Dict[str, Dict[str, Any]] = self._discover_managed_servers()
        self.active_sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        atexit.register(self.shutdown)

    def run_coroutine_and_get_result(self, coro: Coroutine) -> Any:
        """Runs a coroutine on the manager's event loop and returns the result."""
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            # Using a timeout to prevent indefinite blocking
            return future.result(timeout=60)
        except Exception as e:
            future.cancel()
            raise e

    def _discover_managed_servers(self) -> Dict[str, Dict[str, Any]]:
        """Dynamically discovers MCP server scripts in the 'mcp_servers' directory."""
        servers_dir = os.path.join(os.path.dirname(__file__), 'mcp_servers')
        discovered_servers: Dict[str, Dict[str, Any]] = {}
        
        if not os.path.isdir(servers_dir):
            print(f"Warning: MCP servers directory not found at '{servers_dir}'")
            return {}

        print("Discovering managed MCP servers...")
        for filename in os.listdir(servers_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                # e.g., 'calculator_server.py' -> 'calculator_server'
                server_name = os.path.splitext(filename)[0]
                # e.g., 'calculator_server' -> 'managed_calculator'
                server_id = f"managed_{server_name.replace('_server', '')}"
                script_path = os.path.join(servers_dir, filename)

                discovered_servers[server_id] = {
                    "command": sys.executable,
                    "args": [script_path],
                }
                print(f"  - Discovered '{server_id}' -> {script_path}")
        
        return discovered_servers

    async def get_session(self, server_id: str) -> Optional[ClientSession]:
        if server_id in self.active_sessions:
            return self.active_sessions[server_id]

        server_config = self.server_configs.get(server_id)
        if not server_config:
            print(f"Error: No configuration found for server_id '{server_id}'")
            return None
        
        try:
            print(f"Creating new MCP session for '{server_id}'...")
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
            )

            # The exit stack will manage the lifecycle of these async context managers
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio_transport[0], stdio_transport[1]))
            
            await session.initialize()
            print(f"MCP session for '{server_id}' initialized successfully.")
            self.active_sessions[server_id] = session
            return session
        except Exception as e:
            print(f"Error creating MCP session for '{server_id}': {e}")
            return None

    async def stop_all_async(self):
        print("Stopping all managed MCP sessions...")
        await self.exit_stack.aclose()
        self.active_sessions.clear()
        print("All MCP sessions stopped.")

    def shutdown(self):
        """Stops all servers and shuts down the event loop."""
        if not self._loop.is_running():
            return
        
        print("Shutting down MCP Server Manager...")
        try:
            self.run_coroutine_and_get_result(self.stop_all_async())
        except Exception as e:
            print(f"Error during MCP session shutdown: {e}")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()
            print("MCP Server Manager shut down.")