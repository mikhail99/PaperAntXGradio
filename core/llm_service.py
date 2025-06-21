import os
import dotenv
#import google.generativeai as genai
import openai
#import anthropic
import json
from typing import List, Dict, Optional, Generator, Any

dotenv.load_dotenv()

class LLMService:
    def __init__(self):
        self.reload()

    def reload(self):
        """Reloads API keys, models, and provider from .env file."""
        print("Reloading LLM service configuration...")
        dotenv.load_dotenv(override=True)
        
        # API Keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Provider and Models
        self.default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

        self.openai_client = None
        self.anthropic_client = None

        if self.gemini_api_key and self.gemini_api_key != 'your_gemini_api_key_here':
            try:
                genai.configure(api_key=self.gemini_api_key)
            except Exception as e:
                print(f"Warning: Failed to configure Gemini: {e}")

        if self.openai_api_key and self.openai_api_key != 'your_openai_api_key_here':
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        if self.anthropic_api_key and self.anthropic_api_key != 'your_anthropic_api_key_here':
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

    def call_llm(self, messages: List[Dict[str, any]], system_prompt: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None) -> Generator[Dict, None, None]:
        provider = (provider or self.default_provider).lower()

        try:
            if provider == "gemini":
                yield from self._call_gemini(messages, system_prompt, tools, model)
            elif provider == "openai":
                yield from self._call_openai(messages, system_prompt, tools, model)
            elif provider == "anthropic":
                yield from self._call_anthropic(messages, system_prompt, tools, model)
            else:
                yield {"type": "error", "content": f"Error: Unsupported LLM provider '{provider}'. Supported providers are 'gemini', 'openai', 'anthropic'."}
        except Exception as e:
            error_message = f"Error calling LLM provider '{provider}': {e}"
            print(f"ERROR: {error_message}")
            yield {"type": "error", "content": error_message}

    def _call_gemini(self, messages: List[Dict[str, Any]], system_prompt: Optional[str], tools: Optional[List[Dict[str, Any]]], model: Optional[str]) -> Generator[Dict, None, None]:
        if not self.gemini_api_key or self.gemini_api_key == 'your_gemini_api_key_here':
            yield {"type": "error", "content": "Error: GEMINI_API_KEY not found or is a placeholder. Please set it in your .env file."}
            return
        
        model_name = model or self.gemini_model
        model_kwargs = {}
        if system_prompt:
            model_kwargs['system_instruction'] = system_prompt
        if tools:
            # The genai SDK expects a specific format for tools.
            gemini_tools = [{"function_declarations": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                } for tool in tools
            ]}]
            model_kwargs['tools'] = gemini_tools
        
        model = genai.GenerativeModel(model_name=model_name, **model_kwargs)
        
        # Convert the generic message history to the format Gemini's SDK expects.
        gemini_messages = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            
            if role == "tool":
                # For Gemini, multiple tool responses for a single turn are grouped in one message.
                if "tool_results" in msg:
                    parts = [
                        {
                            "function_response": {
                                "name": res["tool_name"],
                                "response": {"content": res["content"]},
                            }
                        } for res in msg.get("tool_results", [])
                    ]
                    if parts:
                        gemini_messages.append({"role": "tool", "parts": parts})
            elif msg.get("tool_calls"): # Assistant message with tool calls
                parts = [{
                    "function_call": {
                        "name": tc['function']['name'], 
                        "args": tc['function']['arguments']
                    }
                } for tc in msg['tool_calls']]
                gemini_messages.append({"role": role, "parts": parts})
            else: # Regular user/assistant text message
                content = msg.get("content")
                if content is not None:
                     gemini_messages.append({"role": role, "parts": [{"text": str(content)}]})

        print(f"--- Calling Gemini (model: {model_name}) ---")
        print(f"Messages: {json.dumps(gemini_messages, indent=2)}")
        print("------------------------------------------")

        response_stream = model.generate_content(gemini_messages, stream=True)
        for chunk in response_stream:
            if chunk.parts:
                for part in chunk.parts:
                    if part.text:
                        yield {"type": "text_chunk", "content": part.text}
                    elif part.function_call:
                        yield {
                            "type": "tool_call",
                            "tool_call": {
                                "id": f"call_{part.function_call.name}_{os.urandom(4).hex()}", # Gemini doesn't provide a call ID
                                "name": part.function_call.name,
                                "arguments": dict(part.function_call.args),
                            }
                        }

    def _call_openai(self, messages: List[Dict[str, any]], system_prompt: Optional[str], tools: Optional[List[Dict[str, Any]]], model: Optional[str]) -> Generator[Dict, None, None]:
        if not self.openai_client:
            yield {"type": "error", "content": "Error: OPENAI_API_KEY not found, is a placeholder, or client not initialized. Please set it in your .env file."}
            return

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        for m in messages:
            if m["role"] == "assistant" and "tool_calls" in m:
                full_messages.append(m)
            elif m["role"] == "tool":
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"]
                })
            else: # user or regular assistant message
                full_messages.append({"role": m["role"], "content": m["content"]})
        
        model_name = model or self.openai_model
        
        request_params = {
            "model": model_name,
            "messages": full_messages,
            "stream": True,
        }
        
        if tools:
            openai_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}} for t in tools]
            request_params["tools"] = openai_tools
            request_params["tool_choice"] = "auto"

        print(f"--- Calling OpenAI (model: {model_name}) ---")
        print(f"Messages: {json.dumps(full_messages, indent=2)}")
        print(f"Tools: {json.dumps(tools, indent=2)}")
        print("-----------------------------------------")
        
        try:
            stream = self.openai_client.chat.completions.create(**request_params)
            
            tool_calls = []
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield {"type": "text_chunk", "content": delta.content}
                
                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        if tool_call_chunk.index >= len(tool_calls):
                            tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        
                        tc = tool_calls[tool_call_chunk.index]
                        if tool_call_chunk.id:
                            tc["id"] = tool_call_chunk.id
                        if tool_call_chunk.function.name:
                            tc["function"]["name"] += tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments:
                            tc["function"]["arguments"] += tool_call_chunk.function.arguments

            if tool_calls:
                for tc in tool_calls:
                    yield {
                        "type": "tool_call",
                        "tool_call": {
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"]),
                        }
                    }
        except Exception as e:
            yield {"type": "error", "content": f"OpenAI API Error: {str(e)}"}

    def _call_anthropic(self, messages: List[Dict[str, any]], system_prompt: Optional[str], tools: Optional[List[Dict[str, Any]]], model: Optional[str]) -> Generator[Dict, None, None]:
        if not self.anthropic_client:
            yield {"type": "error", "content": "Error: ANTHROPIC_API_KEY not found, is a placeholder, or client not initialized. Please set it in your .env file."}
            return
        
        # The 'messages' list is expected to be pre-formatted by the calling service,
        # especially for tool results, which should be in Anthropic's specific format.
        anthropic_messages = messages

        model_name = model or self.anthropic_model
        
        api_kwargs = {
            "model": model_name,
            "system": system_prompt,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }
        if tools:
            # Convert to Anthropic's 'input_schema' format.
            anthropic_tools = []
            for tool in tools:
                new_tool = tool.copy()
                new_tool['input_schema'] = new_tool.pop('inputSchema')
                anthropic_tools.append(new_tool)
            api_kwargs["tools"] = anthropic_tools

        print(f"--- Calling Anthropic (model: {model_name}) ---")
        print(f"Messages: {json.dumps(anthropic_messages, indent=2)}")
        print("---------------------------------------------")

        with self.anthropic_client.messages.stream(**api_kwargs) as stream:
            for event in stream:
                if event.type == 'content_block_delta':
                    if event.delta.type == 'text_delta':
                        yield {"type": "text_chunk", "content": event.delta.text}
                elif event.type == 'message_delta' and event.delta.stop_reason == "tool_use":
                    # This event contains the full tool calls, not a delta
                    for content_block in event.message.content:
                        if content_block.type == 'tool_use':
                             yield {
                                "type": "tool_call",
                                "tool_call": {
                                    "id": content_block.id,
                                    "name": content_block.name,
                                    "arguments": content_block.input,
                                }
                            }
                elif event.type == 'content_block_start':
                    if event.content_block.type == 'tool_use':
                        # First chunk of a tool call comes in a content_block_start
                        # But we will use the full tool_use from message_delta
                        pass
