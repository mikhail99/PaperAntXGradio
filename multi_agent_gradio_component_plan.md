# Multi-Agent Unified Chat Gradio Component: Design Plan

## 1. Component Overview
- **Purpose:** Unified chat interface for multiple agents, with agent-tagged messages.
- **Inputs:** List of agents (name, id, description), callable for message handling.
- **Outputs:** Renders chat history (Markdown, tables, plots), agent selection, agent tags.
- **Core Pattern:** Uses Gradio's `ChatMessage` and the [Returning Complex Responses](https://www.gradio.app/guides/creating-a-chatbot-fast#returning-complex-responses) pattern for all agent/user messages, including thoughts, references, markdown, tables, and (future) plots.

---

## 2. Frontend (Svelte/JS)
- **Sidebar:**  
  - List all agents with name, description, and avatar/color.
  - Highlight selected agent.
- **Main Chat Area:**  
  - Unified chat history (all messages, agent/user tagged).
  - Each message is a Gradio `ChatMessage` object:
    - Agent/user tag (color-coded or icon).
    - Markdown-rendered text.
    - Tables (HTML, rendered from backend).
    - (Future) Plots as images or interactive objects.
    - **Thoughts/References as Accordions:**
      - Use `ChatMessage` with `metadata={"title": ...}` to show agent thought process or references as collapsible accordions ([see Gradio guide](https://www.gradio.app/guides/agents-and-tool-usage)).
      - Example: `ChatMessage(role="assistant", content="Reasoning step...", metadata={"title": "🧠 Thinking"})`
      - Example: `ChatMessage(role="assistant", content="• Smith et al. 2023", metadata={"title": "📚 References"})`
- **Input Box:**  
  - Single message input.
  - Send button.
- **file upload.** low prioriry
- **export/download.** low prioriry
- **streaming** low prioriry

---

## 3. Backend (Python)
- **Agent Registry:**  
  - List of agents with id, name, description.
- **Message Handler:**  
  - Receives: agent_id, message, chat history.
  - Calls: agent callable, returns or yields `ChatMessage` objects (with markdown, table HTML, plot image/HTML, and optionally intermediate thoughts/references as accordions via metadata).
  - Follows the [Returning Complex Responses](https://www.gradio.app/guides/creating-a-chatbot-fast#returning-complex-responses) pattern for all outputs.
- **Table/Plot Handling:**  
  - Tables: Convert pandas DataFrame to HTML string and return as `ChatMessage` content.
  - Plots: (Future) Save matplotlib/plotly figure to image, return image path or HTML as `ChatMessage` content.
- **Chat History:**  
  - Maintained as a list of `ChatMessage` objects: `{role, agent_id, content, metadata, timestamp}`.

---

## 4. Data Model
- **Agent:** `{id, name, description, color/icon}`
- **Message:** `gr.ChatMessage` with fields: `{role, agent_id, content (md/table/plot), metadata (for accordions), timestamp}`

---

## 5. Component API
- **Props:**  
  - `agents: List[Agent]`
  - `on_message(agent_id, message, history) -> List[ChatMessage] or generator of ChatMessage`
- **Events:**  
  - `on_agent_select(agent_id)`
  - `on_message_send(message)`

---

## 6. UI/UX Decisions
- **Unified chat:** All messages in one scrollable window, agent/user clearly tagged.
- **Agent switch:** Only affects which agent is used for next message; history is not filtered.
- **file upload, export/download, streaming:** All low priority, can be added later.
- **Tables/plots shown inline in chat.**
- **Thoughts/references shown as accordions using ChatMessage metadata.**
- **All outputs and intermediate steps use the ChatMessage/metadata pattern for maximum flexibility.**

---

## 7. Extensibility
- **Adding plots:** Backend returns plot as image or HTML in a ChatMessage; frontend renders inline.
- **Adding export/streaming:** Can be added later with minimal changes.
- **All future output types (images, citations, etc.) can use ChatMessage with appropriate metadata.**

---

## 8. Implementation Steps
1. Scaffold Gradio custom component (template: Chatbot or MultimodalChatbot).
2. Implement agent sidebar and selection logic.
3. Implement unified chat area with agent/user tags.
4. Implement Markdown and table rendering in chat.
5. Implement backend message handler (agent dispatch, table/plot conversion, ChatMessage with metadata for thoughts/references, following the Returning Complex Responses pattern).
6. Test with multiple agents and various output types.
7. Document API and usage.

---

**References:**
- [Gradio PDF Component Example](https://www.gradio.app/guides/pdf-component-example)
- [Gradio Multimodal Chatbot Part 1](https://www.gradio.app/guides/multimodal-chatbot-part1)
- [Gradio Agents & Tool Usage Guide](https://www.gradio.app/guides/agents-and-tool-usage)
- [Gradio Creating a Chatbot Fast: Returning Complex Responses](https://www.gradio.app/guides/creating-a-chatbot-fast#returning-complex-responses) 