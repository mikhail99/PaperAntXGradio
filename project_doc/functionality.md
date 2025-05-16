## 1. Functionality

PaperAnt X is an AI-powered research article management system designed for single-user, local use. It leverages Gradio v5.x [Blocks API](https://www.gradio.app/guides/blocks-and-event-listeners) for a modular, event-driven UI, with state managed via `gr.State`. The AI copilot is strictly backend-driven. For rapid development, use Gradio's reload mode (`gradio app.py --reload`).

### 1.1. Collection Management
*   **Collection Creation & Organization:**
    *   Users can create up to 20 distinct collections (global limit for this single-user version).
    *   Collections are single-level (no hierarchical nesting of collections themselves).
    *   Each collection includes:
        *   **Name:** A unique identifier for the collection.
        *   **Description:** Text describing the desired articles for the collection (this description can be used by backend recommendation systems).
        *   **Tags Definition:** A comma-separated string of tags. Hierarchical relationships can be indicated by a `parent/child` syntax within a tag string (e.g., "methodology/quantitative, topic/nlp").
        *   **Article List:** Articles associated with the collection.
        *   **(Implicit) Recommendation Queue:** New articles flagged for review within this collection.
*   **Collection Configuration:**
    *   Create new collections with name, description, and tag definitions.
    *   Modify existing collection's name, description, and tag definitions.
    *   Archive collections (mark them as inactive, potentially hiding them from main views).
*   **Tag Management (Simplified):**
    *   Tags are defined as a simple text string (comma-separated) during collection creation or modification.
    *   The system will parse this string to generate a list of usable tags for assignment and filtering.
    *   Tags are collection-specific.

### 1.2. Article Management
*   **Article Addition Methods:**
    *   **Weekly Recommendations:** New articles are sourced via an external process that adds them to ChromaDB and flags them for review within relevant collections. The app will display these for processing.
    *   **Manual Addition:** Add individual articles by providing an Article ID (e.g., DOI, arXiv ID).
    *   **Batch Import:** Upload a simple file (e.g., CSV containing a list of article IDs) for batch addition.
*   **Article Rating and Organization:**
    *   **Rating Options:**
        *   **Accept:** Add the article to the collection's main list.
        *   **Reject:** Remove the article from active consideration/recommendations for this collection.
        *   **Favorite:** Mark an accepted article with a special designation.
        *   **Downgrade:** Move a previously accepted article to the "rejected" state.
    *   **Tag Assignment:**
        *   Assign tags to articles during initial review or when editing an existing article. Tags are selected from the list defined for the collection.
        *   Modify tags for existing articles.
*   **Article Metadata Display:**
    *   Title, Authors, Publication Date, Citation Count, Abstract.
    *   Assigned tags.
    *   User-added notes.
*   **Article Operations:**
    *   Add to / Remove from other collections (articles can belong to multiple collections).
    *   Mark as favorite.
    *   Add/edit personal notes/annotations.
    *   View full text (link to PDF, opens in a new tab).
    *   Modify article tags.
    *   Update article ratings.
*   **Tip:** Use a state variable to track the currently selected article and update the detail view. Example:
  ```python
  def select_article(state, article_id):
      # ...
      return updated_state, article_details
  article_df.select(select_article, [state, article_id], [state, detail_view])
  ```

### 1.3. Weekly Article Processing
*   The system identifies new, unreviewed articles associated with each collection (sourced externally).
*   These new articles are prioritized for review, e.g., displayed at the top of article lists or in a dedicated review interface.
*   Users can rate and assign tags to these new articles as part of the review process.
*   **Pattern:** Use a state variable to manage the review queue and current index. Example:
  ```python
  def next_article(state):
      # ...
      return updated_state, next_article
  next_btn.click(next_article, [state], [state, article_view])
  ```

### 1.4. AI-Powered Analysis (Copilot Integration)
*   **Context-Aware Querying:**
    *   Users can ask natural language questions.
    *   The primary context for questions will be the currently selected collection.
    *   Questions can implicitly reference articles within this collection.
*   **Copilot Capabilities (driven by backend LLM service):**
    *   All LLM logic is handled in the backend, not in Gradio itself.
    *   Answer questions about article content within the current collection.
    *   Summarize the current collection or articles with specific tags within it.
    *   Compare articles within the current collection.
    *   Identify trends based on tags within the current collection.
*   **Conversation Management (Simplified):**
    *   Maintains a single conversation history for the Copilot.
    *   Context is passed to the backend based on the currently active collection.

### 1.5. Search and Discovery
*   **Search Functionality:**
    *   Full-text search across articles within the currently selected collection (e.g., searching titles, abstracts).
    *   Filter articles within a collection by their assigned tags.

---

**Data Persistence:**
- ChromaDB is the primary persistent storage for articles and metadata.
- For prototyping or simple use cases, JSON files or Gradio's file upload/download can be considered, but ChromaDB is recommended for scalability and search.

**For more Gradio best practices and examples:**
- [Quickstart Guide](https://www.gradio.app/guides/quickstart)
- [Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners)
- [Developing Faster with Reload Mode](https://www.gradio.app/guides/developing-faster-with-reload-mode)
- [Gradio Example Gallery](https://www.gradio.app/guides)