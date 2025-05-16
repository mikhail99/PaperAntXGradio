import gradio as gr
import traceback
import pandas as pd
from core.data_models import Collection, Article
from core.collections_manager import CollectionsManager
from core.article_manager import ArticleManager
from core.copilot_service import CopilotService

# Helper to convert collections dict to DataFrame rows
def collections_to_rows(collections):
    rows = []
    for c in collections:
        rows.append([
            c.id,
            c.name,
            c.description,
            c.archived,
            ", ".join([t.name for t in c.tags.values()]),
        ])
    return rows

def articles_to_df(articles):
    data = []
    for a in articles:
        data.append([
            a.id,
            a.title,
            ", ".join(a.authors),
            a.rating or "",
            ", ".join(a.tags),
        ])
    return pd.DataFrame(data, columns=["ID", "Title", "Authors", "Rating", "Tags"])

def style_article_rows(df):
    def row_style(row):
        color = {
            "accept": "background-color: #d0e7ff;",
            "reject": "background-color: #eeeeee;",
            "favorite": "background-color: #d4edda;",
            None: "background-color: #fff9d1;",
            "": "background-color: #fff9d1;",
        }.get(row["Rating"], "background-color: #fff9d1;")
        return [color] * len(row)
    return df.style.apply(row_style, axis=1)

# Initialize the collection manager and article manager
manager = CollectionsManager(persist_directory="data/chroma_db_store")
article_manager = ArticleManager(manager)
copilot_service = CopilotService(manager, article_manager)

# Define collection_options once for all tabs
collection_options = [(c.name, c.id) for c in manager.get_all_collections() if not c.archived]

with gr.Blocks(title="PaperAnt X") as demo:
    selected_collection_id = gr.State("")
    selected_article_id = gr.State("")
    copilot_chat_history = gr.State([])
    selected_article_title = gr.State("")

    gr.Markdown("# PaperAnt X")
    with gr.Tabs():
        with gr.TabItem("Collections Management"):
            with gr.Row():
                collections_df = gr.Dataframe(
                    headers=["ID", "Name", "Description", "Archived", "Tags"],
                    value=collections_to_rows(manager.get_all_collections()),
                    label="Collections",
                    interactive=True,
                    max_height=200,
                )
            with gr.Row():
                name_box = gr.Textbox(label="Collection Name")
                desc_box = gr.Textbox(label="Collection Description", lines=2)
                tags_box = gr.Textbox(label="Tags (comma-separated, e.g., topic/subtopic, method)")
            with gr.Row():
                create_btn = gr.Button("Create New Collection")
                update_btn = gr.Button("Update Selected Collection")
                archive_btn = gr.Button("Archive Selected Collection")
            status = gr.Markdown("")

        with gr.TabItem("Article Management"):
            with gr.Row():
                collection_dropdown = gr.Dropdown(
                    choices=collection_options,
                    label="Select Collection",
                    value=collection_options[0][1] if collection_options else None,
                )
                search_box = gr.Textbox(label="Search Articles (Title, Abstract)")
                tag_filter_dropdown = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="Filter by Tags (select one or more)",
                )
                semantic_search_box = gr.Textbox(label="Semantic Search (Natural Language Query)")
                semantic_search_btn = gr.Button("Semantic Search")

            articles_df = gr.Dataframe(
                value=style_article_rows(pd.DataFrame(columns=["ID", "Title", "Authors", "Rating", "Tags"])),
                label="Articles",
                interactive=False,
                max_height=200,
            )

            article_title_md = gr.Markdown("<i>No article selected</i>")
            with gr.Row():
                with gr.Column():
                    article_abstract_md = gr.Markdown("<i>No article selected</i>")
                with gr.Column():
                    with gr.Row():
                        rating_radio = gr.Radio(label="Rating", choices=["accept", "reject", "favorite"])
                        tags_edit_box = gr.Textbox(label="Modify Tags (comma-separated)")

                    notes_box = gr.Textbox(label="My Notes", lines=4)
                    save_article_btn = gr.Button("Update Article Details (Rating/Tags/Notes)")
            
   
            with gr.Row():
                add_id_box = gr.Textbox(label="Add Article by ID (DOI, arXiv ID, etc.)")
                fetch_add_btn = gr.Button("Fetch and Add Article")
            article_status = gr.Markdown("")

        with gr.TabItem("AI Copilot"):
            with gr.Row():
                copilot_context = gr.Markdown("Copilot context: (select a collection for best results)")
            with gr.Row():
                copilot_chatbot = gr.Chatbot(label="Copilot Conversation", type="messages")
            with gr.Row():
                copilot_input = gr.Textbox(label="Your Question:", placeholder="Ask about articles, summaries, comparisons...", lines=2)
                copilot_send_btn = gr.Button("Send to Copilot")
            copilot_status = gr.Markdown("")

        with gr.TabItem("PaperQA"):
            with gr.Row():
                paperqa_collection_dropdown = gr.Dropdown(
                    choices=collection_options,
                    label="Select Collection",
                    value=collection_options[0][1] if collection_options else None,
                )
            with gr.Row():
                paperqa_input = gr.Textbox(label="Your Question:", placeholder="Ask a detailed question about the collection...", lines=2)
                paperqa_btn = gr.Button("Get Report")
            paperqa_output = gr.Markdown("(Report will appear here)")

        with gr.TabItem("MindMap"):
            with gr.Row():
                mindmap_collection_dropdown = gr.Dropdown(
                    choices=collection_options,
                    label="Select Collection",
                    value=collection_options[0][1] if collection_options else None,
                )
            mindmap_output = gr.Markdown("""
```mermaid
graph TD
    A[Collection] --> B[Tag 1]
    A --> C[Tag 2]
    B --> D[Article 1]
    C --> E[Article 2]
```
""", label="MindMap Diagram")

    # --- Callbacks ---
    def refresh_collections():
        try:
            return collections_to_rows(manager.get_all_collections())
        except Exception as e:
            print(f"Error refreshing collections: {str(e)}")
            print(traceback.format_exc())
            return []

    def handle_create_collection(name, desc, tags_str):
        if not name.strip():
            return gr.update(), "**Name is required**"
        try:
            collection = manager.create_collection(name, desc)
            if tags_str.strip():
                manager.parse_and_add_tags(collection, tags_str)
            return collections_to_rows(manager.get_all_collections()), "**Collection created**"
        except Exception as e:
            error_msg = f"**Error: {str(e)}**"
            print(error_msg)
            print(traceback.format_exc())
            return gr.update(), error_msg

    def handle_select_collection(evt: gr.SelectData):
        try:
            rows = collections_to_rows(manager.get_all_collections())
            if evt.index is None or evt.index >= len(rows):
                return gr.update(), gr.update(), gr.update(), ""
            row = rows[evt.index]
            cid, name, desc, archived, tags = row
            return name, desc, tags, cid
        except Exception as e:
            print(f"Error selecting collection: {str(e)}")
            print(traceback.format_exc())
            return gr.update(), gr.update(), gr.update(), ""

    def handle_update_collection(selected_id, name, desc, tags_str):
        if not selected_id:
            return gr.update(), "**Select a collection to update**"
        try:
            collection = manager.update_collection(selected_id, name, desc)
            if not collection:
                return gr.update(), "**Collection not found**"
            if tags_str.strip():
                manager.parse_and_add_tags(collection, tags_str)
            return collections_to_rows(manager.get_all_collections()), "**Collection updated**"
        except Exception as e:
            error_msg = f"**Error: {str(e)}**"
            print(error_msg)
            print(traceback.format_exc())
            return gr.update(), error_msg

    def handle_archive_collection(selected_id):
        if not selected_id:
            return gr.update(), "**Select a collection to archive**"
        try:
            collection = manager.archive_collection(selected_id)
            if not collection:
                return gr.update(), "**Collection not found**"
            return collections_to_rows(manager.get_all_collections()), "**Collection archived**"
        except Exception as e:
            error_msg = f"**Error: {str(e)}**"
            print(error_msg)
            print(traceback.format_exc())
            return gr.update(), error_msg

    # --- Article Management Callbacks ---
    def get_articles_for_collection(collection_id):
        collection = manager.get_collection(collection_id)
        if not collection:
            return []
        return list(collection.articles.values())

    def handle_article_select(evt: gr.SelectData, collection_id):
        try:
            articles = get_articles_for_collection(collection_id)
            rows = articles_to_df(articles)
            if not evt.index or len(evt.index) == 0:
                return "", "", None, "", "<i>No article selected</i>", "<i>No article selected</i>"
            row_index = evt.index[0]
            if row_index is None or row_index >= len(rows):
                return "", "", None, "", "<i>No article selected</i>", "<i>No article selected</i>"
            selected_article_id = rows.iloc[row_index]["ID"]  # Get ID from the row
            article = None
            for a in articles:
                if a.id == selected_article_id:
                    article = a
                    break
            if not article:
                return "", "", None, "", "<i>No article selected</i>", "<i>No article selected</i>"
            notes = getattr(article, 'notes', "")
            tags = ", ".join(article.tags)
            rating = article.rating
            title_md = f"### {article.title}"
            abstract_md = f"**Abstract:**<br>{article.abstract}" if article.abstract else "<i>No abstract available</i>"
            return notes, tags, rating, selected_article_id, title_md, abstract_md
        except Exception as e:
            print(f"Error selecting article: {str(e)}")
            print(traceback.format_exc())
            return "", "", None, "", "<i>No article selected</i>", "<i>No article selected</i>"

    def handle_save_article(collection_id, article_id, notes, tags_str, rating):
        if not article_id:
            return "**No article selected**", style_article_rows(articles_to_df(get_articles_for_collection(collection_id)))
        try:
            article = article_manager.get_article(collection_id, article_id)
            if not article:
                return "**Article not found**", style_article_rows(articles_to_df(get_articles_for_collection(collection_id)))
            article.notes = notes
            article.tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            article.rating = rating
            updated = article_manager.update_article(collection_id, article)
            status = "**Article updated**" if updated else "**Failed to update article**"
            updated_articles = get_articles_for_collection(collection_id)
            return status, style_article_rows(articles_to_df(updated_articles))
        except Exception as e:
            error_msg = f"**Error updating article: {str(e)}**"
            print(error_msg)
            print(traceback.format_exc())
            updated_articles = get_articles_for_collection(collection_id)
            return error_msg, style_article_rows(articles_to_df(updated_articles))

    def handle_semantic_search(collection_id, query):
        if not collection_id or not query.strip():
            return style_article_rows(pd.DataFrame(columns=["ID", "Title", "Authors", "Rating", "Tags"]))
        try:
            articles = manager.search_articles(collection_id, query, limit=20)
            return style_article_rows(articles_to_df(articles))
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            print(traceback.format_exc())
            return style_article_rows(pd.DataFrame(columns=["ID", "Title", "Authors", "Rating", "Tags"]))

    def update_tag_filter_dropdown(collection_id):
        collection = manager.get_collection(collection_id)
        if not collection:
            return gr.update(choices=[])
        tag_names = [tag.name for tag in collection.tags.values()]
        return gr.update(choices=tag_names)

    def handle_filter_articles(collection_id, search_text, selected_tags):
        if not collection_id:
            return style_article_rows(pd.DataFrame(columns=["ID", "Title", "Authors", "Rating", "Tags"]))
        try:
            # Get all articles for the collection
            articles = get_articles_for_collection(collection_id)
            # Filter by search text (title or abstract)
            if search_text.strip():
                search_lower = search_text.strip().lower()
                articles = [a for a in articles if search_lower in a.title.lower() or search_lower in a.abstract.lower()]
            # Filter by tags
            if selected_tags:
                articles = [a for a in articles if any(tag in a.tags for tag in selected_tags)]
            return style_article_rows(articles_to_df(articles))
        except Exception as e:
            print(f"Error filtering articles: {str(e)}")
            print(traceback.format_exc())
            return style_article_rows(pd.DataFrame(columns=["ID", "Title", "Authors", "Rating", "Tags"]))

    def update_copilot_context(collection_id):
        if not collection_id:
            return "Copilot context: (select a collection for best results)"
        collection = manager.get_collection(collection_id)
        if not collection:
            return "Copilot context: (collection not found)"
        return f"Copilot context: {collection.name} — {collection.description}"

    def handle_copilot_send(user_message, collection_id, chat_history):
        if not user_message.strip():
            # Return all three outputs: chatbot, status, state
            return chat_history, "", chat_history
        # Get mock LLM response
        response = copilot_service.ask(user_message, collection_id)
        # Update chat history (tuples)
        new_history = chat_history + [[user_message, response]]
        # Convert to OpenAI-style messages for gr.Chatbot
        messages = []
        for user, bot in new_history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": bot})
        return messages, "", new_history

    def handle_paperqa(collection_id, question):
        if not collection_id or not question.strip():
            return "Please select a collection and enter a question."
        # Generate a mock long markdown report
        collection = manager.get_collection(collection_id)
        if not collection:
            return "Collection not found."
        report = f"# PaperQA Report\n\n**Collection:** {collection.name}\n\n**Question:** {question}\n\n---\n\n## Mock Analysis\n\nThis is a detailed (mock) report for your question.\n\n- **Number of articles:** {len(collection.articles)}\n- **Tags:** {', '.join([t.name for t in collection.tags.values()])}\n\n### Example Section\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum.\n\n---\n\n*In a real implementation, this would be generated by an LLM.*"
        return report

    def handle_mindmap(collection_id):
        if not collection_id:
            return "Please select a collection."
        collection = manager.get_collection(collection_id)
        if not collection:
            return "Collection not found."
        # Dummy mermaid diagram (could be made dynamic)
        mermaid = f"""
```mermaid
graph TD\n    A[Collection: {collection.name}] --> B[Tag 1]\n    A --> C[Tag 2]\n    B --> D[Article 1]\n    C --> E[Article 2]\n```
"""
        return mermaid

    # Auto-refresh articles table on collection, search, or tag filter change
    def auto_refresh_articles(collection_id, search_text, selected_tags):
        return handle_filter_articles(collection_id, search_text, selected_tags)
    collection_dropdown.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df])
    search_box.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df])
    tag_filter_dropdown.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df])

    # Bind callbacks for collections
    create_btn.click(handle_create_collection, [name_box, desc_box, tags_box], [collections_df, status])
    collections_df.select(handle_select_collection, None, [name_box, desc_box, tags_box, selected_collection_id])
    update_btn.click(handle_update_collection, [selected_collection_id, name_box, desc_box, tags_box], [collections_df, status])
    archive_btn.click(handle_archive_collection, [selected_collection_id], [collections_df, status])

    # Bind callbacks for articles
    articles_df.select(handle_article_select, [collection_dropdown], [notes_box, tags_edit_box, rating_radio, selected_article_id, article_title_md, article_abstract_md])
    save_article_btn.click(handle_save_article, [collection_dropdown, selected_article_id, notes_box, tags_edit_box, rating_radio], [article_status, articles_df])
    semantic_search_btn.click(handle_semantic_search, [collection_dropdown, semantic_search_box], [articles_df])
    collection_dropdown.change(update_tag_filter_dropdown, [collection_dropdown], [tag_filter_dropdown])

    # Copilot tab bindings
    collection_dropdown.change(update_copilot_context, [collection_dropdown], [copilot_context])
    copilot_send_btn.click(handle_copilot_send, [copilot_input, collection_dropdown, copilot_chat_history], [copilot_chatbot, copilot_status, copilot_chat_history])

    # PaperQA tab bindings
    paperqa_collection_dropdown.change(handle_paperqa, [paperqa_collection_dropdown, paperqa_input], [paperqa_output])
    paperqa_btn.click(handle_paperqa, [paperqa_collection_dropdown, paperqa_input], [paperqa_output])

    # MindMap tab bindings
    mindmap_collection_dropdown.change(handle_mindmap, [mindmap_collection_dropdown], [mindmap_output])

if __name__ == "__main__":
    demo.launch() 