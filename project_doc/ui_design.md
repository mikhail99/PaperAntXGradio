## 2. UI Design (Gradio)

The application uses Gradio v5.x [Blocks API](https://www.gradio.app/guides/blocks-and-event-listeners) for a modular, event-driven, single-page web app. State is managed with `gr.State` for single-user context. The AI copilot is backend-only, and the UI is optimized for desktop use. For rapid development, use Gradio's reload mode.

### Overall Layout:
*   A main title for the application (e.g., "PaperAnt X").
*   A `gr.Dropdown` for **Global Collection Selection** placed prominently, allowing users to choose the active collection. This selection will affect the content displayed in most tabs.
*   A series of `gr.Tabs` for different sections.

### Tab 1: Collections Management
*   **Purpose:** View, create, edit, and archive collections.
*   **Components:**
    *   `gr.DataFrame` or `gr.List` (if simpler display is preferred) to display existing collections (Name, Description, Number of Articles).
        *   Selecting a collection in this list populates the editing fields.
    *   **Creation/Editing Section:**
        *   `gr.Textbox(label="Collection Name")`
        *   `gr.Textbox(label="Collection Description", lines=3)`
        *   `gr.Textbox(label="Tags (comma-separated, e.g., topic/subtopic, method)", lines=2, placeholder="e.g., machine_learning/nlp, methodology/survey")`
        *   `gr.Button("Create New Collection")`
        *   `gr.Button("Update Selected Collection")`
        *   `gr.Button("Archive Selected Collection")` (confirmation might be good)
    *   Status messages via `gr.Markdown` or `gr.Textbox(interactive=False)` for feedback.
    *   **Example: Button callback with state update:**
      ```python
      def create_collection(state, name, desc, tags):
          # ...
          return updated_state, status
      create_btn.click(create_collection, [state, name, desc, tags], [state, status])
      ```
    *   See [Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners) for more patterns.

### Tab 2: Article Management
*   **Purpose:** View, filter, and manage articles within the globally selected collection.
*   **Prerequisite:** A collection must be selected using the Global Collection Selector.
*   **Components:**
    *   **Filtering & Search Section:**
        *   `gr.Textbox(label="Search Articles (Title, Abstract)")`
        *   `gr.CheckboxGroup` or `gr.Dropdown(multiselect=True, label="Filter by Tags")` populated dynamically based on the selected collection's defined tags.
        *   `gr.Button("Apply Filters/Search")`
    *   **Article List:**
        *   `gr.DataFrame(interactive=True)` to display articles (columns: Title, Authors, Year, Rating, Tags). New/unreviewed articles could be highlighted or sorted to the top.
        *   Selecting a row in the DataFrame populates the "Article Detail View".
    *   **Article Detail View (populated on selection):**
        *   `gr.Markdown` or `gr.Textbox(interactive=False)` for Title, Authors, Publication Date, Citation Count, Abstract.
        *   `gr.Textbox(label="My Notes", lines=4, interactive=True)`
        *   `gr.Button("Save Notes")`
        *   `gr.Markdown(label="Assigned Tags:")` displaying current tags.
        *   `gr.CheckboxGroup` or `gr.Dropdown(multiselect=True, label="Modify Tags")` (populated with collection's tags).
        *   `gr.Radio(label="Rating", choices=["Accept", "Reject", "Favorite"])` (Potentially "Downgrade" handled by changing "Accept" to "Reject").
        *   `gr.Button("Update Article Details (Rating/Tags)")`
        *   `gr.Button("Open PDF")` (links to PDF in new tab).
        *   `gr.Dropdown(label="Manage Collections", multiselect=True)` to add/remove article from other collections.
        *   `gr.Button("Update Article Collections")`
    *   **Manual/Batch Add Section:**
        *   `gr.Textbox(label="Add Article by ID (DOI, arXiv ID, etc.)")`
        *   `gr.Button("Fetch and Add Article")`
        *   `gr.File(label="Batch Import Article IDs (CSV)")`
        *   `gr.Button("Process Batch Import")`
    *   **Tip:** Use state to keep track of selected article and update detail view accordingly.

### Tab 3: AI Copilot
*   **Purpose:** Interact with the AI assistant for analysis and insights.
*   **Prerequisite:** A collection should ideally be selected globally to provide context.
*   **Components:**
    *   `gr.Markdown` indicating "Copilot context: [Selected Collection Name]" (or "Global Context" if no collection is active).
    *   `gr.Chatbot` component for displaying the conversation.
    *   `gr.Textbox(label="Your Question:", placeholder="Ask about articles, summaries, comparisons...", lines=2)` for user input.
    *   `gr.Button("Send to Copilot")`
    *   **Note:** All LLM logic is handled in the backend, not in Gradio itself.

**For more Gradio UI inspiration/examples:**
- [Quickstart Guide](https://www.gradio.app/guides/quickstart)
- [Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners)
- [Gradio Example Gallery](https://www.gradio.app/guides)
- [Developing Faster with Reload Mode](https://www.gradio.app/guides/developing-faster-with-reload-mode)