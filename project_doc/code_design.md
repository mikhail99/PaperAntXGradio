## Development Workflow

To develop and run PaperAnt X locally:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app in reload mode for fast iteration:**
   ```bash
   gradio app.py --reload
   ```
   This will auto-reload the UI on code changes. See [Gradio Reload Mode Guide](https://www.gradio.app/guides/developing-faster-with-reload-mode/).
3. **Access the app:**
   Open the provided local URL in your browser.

---

## 3. Code Design

The application uses Gradio v5.29.1   [Blocks API](https://www.gradio.app/guides/blocks-and-event-listeners) for modular, event-driven UI. State is managed with `gr.State` for single-user context. The backend is strictly separated from the UI, and the AI copilot is a backend-only service.

### 3.1. File Structure
```
paperantx_gradio/
│
├── app.py                  # Main Gradio application, UI definition, callbacks
├── requirements.txt        # Python dependencies
│
├── core/                   # Backend logic module
│   ├── __init__.py
│   ├── data_models.py      # Pydantic models for Collection, Article, Tag
│   ├── collection_manager.py # Logic for collection CRUD, tag string parsing
│   ├── article_manager.py  # Logic for article CRUD, rating, tagging, finding new articles
│   ├── chroma_service.py   # Interaction with ChromaDB (storage, vector search)
│   ├── copilot_service.py  # Logic for AI copilot, LLM interaction
│   └── utils.py            # Helper functions, constants
│
└── data/                   # Directory for persistent ChromaDB storage (or configured path)
    └── chroma_db_store/    # Example sub-directory for ChromaDB files
```

### 3.2. Main Components and Responsibilities:

*   **`app.py` (Presentation Layer - Gradio):**
    *   Uses `gr.Blocks()` to define the UI and layout.
    *   Manages state with `gr.State()` for collections, selected articles, chat history, etc.
    *   Implements event-driven callbacks using `.click`, `.change`, and `.submit` methods.
    *   Example of a callback with state:
      ```python
      def handle_create_collection(state, name, description, tags_str):
          # ...
          return updated_state, status_message
      submit_btn.click(handle_create_collection, [state, name, description, tags_str], [state, status])
      ```
    *   See [Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners) for more.

*   **`core/data_models.py` (Domain Layer):**
    *   Defines Pydantic models for `Collection`, `Article`, `Tag` (even if a tag is just a string, a model can enforce structure if it evolves).
    *   Ensures data consistency and validation.

*   **`core/collection_manager.py` (Domain Layer):**
    *   Functions for creating, retrieving, updating, and archiving collections.
    *   Handles parsing of the comma-separated tag strings defined for a collection into a list of usable tag names/paths.
    *   Interacts with `chroma_service.py` to persist/retrieve collection metadata (if not solely in ChromaDB article metadata).

*   **`core/article_manager.py` (Domain Layer):**
    *   Functions for adding articles (manual, batch), rating, tagging, adding notes.
    *   Logic to retrieve articles for a collection, including filtering and identifying "new" or "unreviewed" articles.
    *   Manages relationships between articles and collections.
    *   Interacts with `chroma_service.py` to store/retrieve article data and embeddings.

*   **`core/chroma_service.py` (Persistence & I/O Layer):**
    *   ChromaDB is the primary persistent storage for articles and metadata.
    *   For simple setups or prototyping, you could use JSON files or Gradio's [file upload/download components](https://www.gradio.app/guides/using-blocks-like-functions), but ChromaDB is recommended for scalability and search.

*   **`core/copilot_service.py` (AI/Domain Layer):**
    *   Receives user queries and context (e.g., current collection ID, selected article IDs) from `app.py`.
    *   Constructs prompts for an external LLM.
    *   May use `chroma_service.py` to fetch relevant article content/summaries via semantic search to augment prompts.
    *   Processes LLM responses and formats them for display in the `gr.Chatbot`.

*   **`core/utils.py` (Utility Layer):**
    *   Common utility functions, constants, or small helper classes used across the `core` modules.
    *   Could include specific parsers (e.g., for article IDs from batch import files).

### 3.3. Data Flow Example (Adding a New Collection):
1.  User fills Name, Description, Tags string in the "Collections Management" tab in `app.py`.
2.  User clicks "Create New Collection" button.
3.  Gradio triggers `handle_create_collection(state, name, description, tags_str)` callback in `app.py`.
4.  `handle_create_collection` calls `core.collection_manager.create_collection(name, description, tags_str)`.
5.  `collection_manager.create_collection` might:
    *   Validate input.
    *   Parse `tags_str`.
    *   Potentially interact with `core.chroma_service.py` if collection metadata needs to be stored/indexed there (though for simple collection definitions, ChromaDB might only be used for articles). For this simplified version, collection metadata might be stored in a simple JSON file or Python dictionary managed by `collection_manager` and persisted periodically.
6.  `collection_manager` returns success/failure and new collection data.
7.  `handle_create_collection` updates `gr.State` holding the list of collections and refreshes the `gr.DataFrame` displaying collections.
8.  Gradio automatically re-renders the UI components bound to the updated state.

This structure promotes separation of concerns, making the application easier to develop, test, and maintain. The core backend logic remains largely independent of the Gradio UI, allowing for potential future UI changes or even API exposure with minimal rework of the core functionalities.

---

**For more on Gradio best practices:**
- [Quickstart Guide](https://www.gradio.app/guides/quickstart)
- [Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners)
- [Developing Faster with Reload Mode](https://www.gradio.app/guides/developing-faster-with-reload-mode)