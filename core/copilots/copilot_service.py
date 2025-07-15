import json
import os
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Generator

from .data_models import Article, Collection
from .collections_manager import CollectionsManager
from .article_manager import ArticleManager
from .llm_service import LLMService
from .mcp_server_manager import MCPServerManager

class CopilotService:
    def __init__(self, collections_manager: CollectionsManager, article_manager: ArticleManager, llm_service: LLMService, mcp_server_manager: MCPServerManager) -> None:
        """Initialize CopilotService with references to the managers"""
        self.collections_manager = collections_manager
        self.article_manager = article_manager
        self.llm_service = llm_service
        self.mcp_server_manager = mcp_server_manager
        self.agents: Dict[str, Any] = self._load_agents()

    def reload(self) -> List[str]:
        """Reloads agent configs, LLM service settings, and restarts MCP sessions."""
        self.llm_service.reload()
        self.agents = self._load_agents()
        
        # Restart MCP sessions to pick up code changes
        try:
            print("Restarting MCP sessions to pick up server code changes...")
            self.mcp_server_manager.run_coroutine_and_get_result(self.mcp_server_manager.stop_all_async())
            print("MCP sessions restarted successfully.")
        except Exception as e:
            print(f"Warning: Error restarting MCP sessions: {e}")
        
        print("Reloaded agents, LLM configurations, and MCP sessions.")
        return self.get_agent_list()

    def _load_agents(self) -> Dict[str, Any]:
        """Load agent configurations from the agents directory."""
        agents = {}
        agents_dir = os.path.join(os.path.dirname(__file__), 'copilot_agents')
        if not os.path.exists(agents_dir):
            print(f"Warning: Agents directory not found at {agents_dir}")
            return {}
        
        for filename in os.listdir(agents_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(agents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        agent_config = json.load(f)
                        if 'name' in agent_config:
                            agent_name = agent_config['name']
                            agents[agent_name] = agent_config
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not decode JSON from {filename}: {e}")
                    except Exception as e:
                        print(f"Error loading agent from {filename}: {e}")
        
        return agents

    def get_agent_list(self) -> List[str]:
        """Returns a list of available agent names."""
        return sorted(list(self.agents.keys()))

    def get_agent_details(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Returns the configuration for a specific agent."""
        return self.agents.get(agent_name)

    def _execute_mcp_tool(self, server_id: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Executes a tool on a managed MCP server by establishing a persistent session.
        This delegates the async execution to the MCPServerManager.
        """
        async def call_tool_async():
            session = await self.mcp_server_manager.get_session(server_id)
            if not session:
                raise ConnectionError(f"Could not establish session with MCP server '{server_id}'.")

            # The mcp-python client library is expected to raise an exception if the tool
            # call results in an error. This simplifies error handling.
            response = await session.call_tool(name=tool_name, arguments=arguments)

            # If no exception is raised, the tool execution was successful.
            return " ".join([part.text for part in response.content if hasattr(part, 'text')]).strip()

        try:
            print(f"Executing tool '{tool_name}' on server '{server_id}' with arguments: {arguments}")
            # The MCPServerManager handles running the coroutine in its own event loop.
            result = self.mcp_server_manager.run_coroutine_and_get_result(call_tool_async())
            print(f"Tool '{tool_name}' executed successfully. Result: {result}")
            return result
        except Exception as e:
            import traceback
            # Capture full exception details
            exception_type = type(e).__name__
            exception_message = str(e)
            full_traceback = traceback.format_exc()
            
            error_message = f"Error executing tool '{tool_name}' on server '{server_id}': {exception_type}: {exception_message}"
            print(f"FULL ERROR DETAILS: {error_message}")
            print(f"TRACEBACK:\n{full_traceback}")
            
            # On failure, try to gracefully shut down sessions to ensure a clean state.
            try:
                self.mcp_server_manager.run_coroutine_and_get_result(self.mcp_server_manager.stop_all_async())
            except Exception as stop_e:
                print(f"Error while stopping MCP sessions after failure: {stop_e}")
            
            # Return a safe error message that won't break JSON serialization
            safe_error = {
                "error": error_message,
                "exception_type": exception_type,
                "details": exception_message
            }
            return json.dumps(safe_error)

    def chat_with_agent(self, agent_name: str, message: str, llm_history: List[Dict[str, Any]], provider: str="ollama") -> Generator[Dict, None, None]:
        """
        Handles a chat interaction with a specific agent, supporting tools.
        This is a generator function to support streaming and tool calls.
        """
        agent_config = self.get_agent_details(agent_name)
        if not agent_config:
            yield {"type": "error", "content": "Agent not found."}
            return

        system_prompt = agent_config.get('model_prompt', None)
        mcp_info = agent_config.get('mcp_info', None)
        tools = mcp_info.get('tools', []) if mcp_info else []
        server_id = mcp_info.get('server_id') if mcp_info else None

        # Add the new user message to the conversation history.
        messages: List[Dict[str, Any]] = llm_history + [{"role": "user", "content": message}]
        
        # Main loop for multi-turn tool use
        while True:
            response_generator = self.llm_service.call_llm(
                messages=messages,
                system_prompt=system_prompt,
                provider=provider,
                model="qwen3:4b",
                tools=tools
            )

            tool_calls_to_process = []
            assistant_response_content = ""
            assistant_message = {"role": "assistant", "content": None}

            for chunk in response_generator:
                if chunk["type"] == "text_chunk":
                    yield chunk
                    if assistant_response_content is None: assistant_response_content = ""
                    assistant_response_content += chunk["content"]
                elif chunk["type"] == "tool_call":
                    tool_calls_to_process.append(chunk["tool_call"])
                    yield chunk
                elif chunk["type"] == "error":
                    yield chunk
                    return

            if assistant_response_content:
                assistant_message["content"] = assistant_response_content
            
            if not tool_calls_to_process:
                break

            if tool_calls_to_process:
                tool_calls_for_history = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls_to_process
                ]
                assistant_message["tool_calls"] = tool_calls_for_history
                if not assistant_response_content:
                    assistant_message.pop("content")
            
            messages.append(assistant_message)

            tool_results = []
            for tool_call in tool_calls_to_process:
                yield {"type": "tool_executing", "tool_call": tool_call}
                
                if not server_id:
                    result = "Error: Agent has tools but no mcp_server_id is configured."
                else:
                    result = self._execute_mcp_tool(
                        server_id, tool_call["name"], tool_call["arguments"]
                    )
                
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_call["name"],
                    "content": result,
                })
                yield {"type": "tool_result", "tool_result": tool_results[-1]}

            # Add tool results to message history for the next LLM call
            if tool_results:
                if provider == 'openai':
                    for res in tool_results:
                        # OpenAI API expects 'tool_call_id' and 'content'. The 'name' is not part of the tool message.
                        messages.append({ "role": "tool", "tool_call_id": res["tool_call_id"], "content": res["content"] })
                elif provider == 'anthropic':
                    # Anthropic expects a single 'user' message containing all tool results.
                    messages.append({
                        "role": "user",
                        "content": [
                            { "type": "tool_result", "tool_use_id": res["tool_call_id"], "content": str(res["content"]) }
                            for res in tool_results
                        ]
                    })
                elif provider == 'gemini':
                    # Gemini requires a single 'tool' role message containing all function responses for a given turn.
                    if tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_results": tool_results
                        })
                elif provider == 'ollama':
                    # For Ollama (and standard OpenAI), each tool result is a separate message.
                    for res in tool_results:
                        messages.append({ "role": "tool", "tool_call_id": res["tool_call_id"], "content": res["content"] })
