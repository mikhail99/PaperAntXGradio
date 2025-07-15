import gradio as gr
from core.collections_manager import CollectionsManager
from core.article_manager import ArticleManager
from core.data_models import Collection, Article
import html
from typing import List
import os

# Initialize managers (in a real app, you might want to pass these in or use a singleton)
manager = CollectionsManager()
print("All collections:", manager.get_all_collections())
article_manager = ArticleManager(manager)

# Helper for table - this global will store the currently displayed/filtered articles
filtered_articles: List[Article] = []

def get_collection_description(collection_name):
    c = manager.get_collection(collection_name)
    return c.description if c else ""

def articles_table_value(articles_list: List[Article]):
    """Prepares data for the articles dataframe and updates the global filtered_articles list."""
    global filtered_articles
    filtered_articles = articles_list
    data = []
    for a in articles_list:
        rating_icon = {
            "favorite": "★",
            "accept": "✓",
            "reject": "✗",
            None: "",
            "": "",
        }.get(a.rating, "")
        tags_text = ", ".join(a.tags)
        data.append([a.title, a.abstract, rating_icon, tags_text])
    headers = ["Title", "Abstract", "Rating", "Tags"]
    return {"data": data, "headers": headers}

def get_articles_for_collection(collection_name):
    collection = manager.get_collection(collection_name)
    if not collection:
        return []
    return list(collection.articles.values())

def create_articles_tab(state):
    collection_options = [c.name for c in manager.get_all_collections() if not c.archived]
    initial_collection_name = collection_options[0] if collection_options else None

    # Determine initial values for UI components
    initial_desc_val = "<i>No collection selected.</i>"
    initial_tag_choices_val = []
    initial_articles_df_data_rows = []
    df_headers = ["Title", "Abstract", "Rating", "Tags"]

    global filtered_articles # Ensure we can set this global on initial load

    if initial_collection_name:
        initial_desc_val = get_collection_description(initial_collection_name) or "<i>No description available.</i>"
        
        _collection_for_tags = manager.get_collection(initial_collection_name)
        if _collection_for_tags:
            initial_tag_choices_val = [tag.name for tag in _collection_for_tags.tags.values()]
        
        _initial_articles_list = get_articles_for_collection(initial_collection_name)
        filtered_articles = _initial_articles_list # Set global here for the selection callback
        for a in _initial_articles_list:
            rating_icon = {
                "favorite": "★", "accept": "✓", "reject": "✗", None: "", "": "",
            }.get(a.rating, "")
            tags_text = ", ".join(a.tags)
            initial_articles_df_data_rows.append([a.title, a.abstract, rating_icon, tags_text])
    else:
        filtered_articles = [] # Initialize if no collection

    with gr.TabItem("📖 Articles"):
        # --- Toolbar Row ---
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                collection_dropdown = gr.Dropdown(
                    choices=collection_options,
                    label="Collection",
                    value=initial_collection_name,
                    interactive=True,
                )
                collection_desc_md = gr.Markdown(value=initial_desc_val) # Use pre-calculated initial value
            with gr.Column(scale=1):
                search_box = gr.Textbox(label="🔍 Search articles...", placeholder="Title, Abstract", scale=2)
                tag_filter_dropdown = gr.Dropdown(
                    choices=initial_tag_choices_val, # Use pre-calculated initial choices
                    value=[], # Initially no tags selected for filtering
                    multiselect=True,
                    label="🏷️ Filter by tags",
                )
            with gr.Column(scale=2):
                semantic_search_box = gr.Textbox(label="🧠 Semantic Search...", placeholder="Natural language query", scale=2)
                semantic_search_btn = gr.Button("Semantic Search", elem_id="semantic-search-btn")

        # --- Articles Table ---
        articles_df = gr.Dataframe(
            headers=df_headers, # Set headers
            value=initial_articles_df_data_rows, # Use pre-calculated initial data rows
            label="Articles",
            interactive=False,
            max_height=300,
            elem_id="articles-table",
            show_search="search",
        )

        # --- Article Details Accordion ---
        with gr.Accordion("Article Details", open=False, elem_classes=["article-details"]):
            article_title_md = gr.Markdown("<i>No article selected</i>")
            with gr.Row():
                with gr.Column(scale=3):
                    article_abstract_md = gr.Markdown("<i>No article selected</i>")
                with gr.Column(scale=2):
                    rating_radio = gr.Radio(
                        label="Rating", 
                        choices=["accept", "reject", "favorite"], 
                        info="✓ Accept, ✗ Reject, ★ Favorite"
                    )
                    tags_edit_box = gr.Textbox(
                        label="Modify Tags (comma-separated)", 
                        placeholder="e.g. AI ethics, fairness, ..."
                    )
                    notes_box = gr.Textbox(
                        label="My Notes", 
                        lines=4, 
                        placeholder="Write your notes here..."
                    )
                    save_article_btn = gr.Button("Update Article Details", elem_classes=["primary"])
        article_status = gr.Markdown(visible=False)

        # --- Article Import ---
        with gr.Accordion("Article Import", open=False):
            with gr.Row():
                add_id_box = gr.Textbox(label="➕ Add Article by ID", placeholder="DOI, arXiv ID, etc.")
                fetch_add_btn = gr.Button("Fetch and Add Article", elem_classes=["primary"])

        # --- Callbacks ---
        def handle_article_select(evt: gr.SelectData, collection_name):
            try:
                global filtered_articles
                if not evt.index or len(evt.index) == 0:
                    return "", "", None, "", "<i>No article selected</i>", "<i>No article selected</i>"
                row_index = evt.index[0]
                if row_index is None or row_index >= len(filtered_articles):
                    return "", "", None, "", "<i>No article selected</i>", "<i>No article selected</i>"
                article = filtered_articles[row_index]
                notes = getattr(article, 'notes', "")
                tags = ", ".join(article.tags)
                rating = article.rating
                title_md = f"### {article.title}\n*by {', '.join(article.authors)}*"
                abstract_md = f"**Abstract:**<br>{article.abstract}" if article.abstract else "<i>No abstract available</i>"
                return notes, tags, rating, article.id, title_md, abstract_md
            except Exception as e:
                print(f"Error selecting article: {str(e)}")
                return "", "", None, "", "<i>No article selected</i>", "<i>No article selected</i>"

        def auto_refresh_articles(collection_name, search_text, selected_tags):
            try:
                if not collection_name:
                    return articles_table_value([])
                articles = get_articles_for_collection(collection_name)
                if search_text.strip():
                    search_lower = search_text.strip().lower()
                    articles = [a for a in articles if search_lower in a.title.lower() or search_lower in a.abstract.lower()]
                if selected_tags:
                    articles = [a for a in articles if any(tag in a.tags for tag in selected_tags)]
                return articles_table_value(articles)
            except Exception as e:
                print(f"Error filtering articles: {str(e)}")
                return articles_table_value([])

        def handle_save_article(collection_name, article_id, notes, tags_str, rating):
            if not article_id:
                return "**No article selected**", articles_table_value(get_articles_for_collection(collection_name))
            try:
                article = article_manager.get_article(collection_name, article_id)
                if not article:
                    return "**Article not found**", articles_table_value(get_articles_for_collection(collection_name))
                article.notes = notes
                article.tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                article.rating = rating
                updated = article_manager.update_article(collection_name, article)
                status = "**Article updated**" if updated else "**Failed to update article**"
                updated_articles = get_articles_for_collection(collection_name)
                return status, articles_table_value(updated_articles)
            except Exception as e:
                error_msg = f"**Error updating article: {str(e)}**"
                print(error_msg)
                updated_articles = get_articles_for_collection(collection_name)
                return error_msg, articles_table_value(updated_articles)

        def handle_semantic_search(collection_name, query):
            if not collection_name or not query.strip():
                return articles_table_value([])
            try:
                articles = manager.search_articles(collection_name, query, limit=20)
                return articles_table_value(articles)
            except Exception as e:
                print(f"Error in semantic search: {str(e)}")
                return articles_table_value([])

        def update_collection_desc(collection_name):
            desc = get_collection_description(collection_name)
            return desc or "<i>No description available.</i>"

        # Bind callbacks
        articles_df.select(handle_article_select, [collection_dropdown], [notes_box, tags_edit_box, rating_radio, state["selected_article_id"], article_title_md, article_abstract_md])
        save_article_btn.click(handle_save_article, [collection_dropdown, state["selected_article_id"], notes_box, tags_edit_box, rating_radio], [article_status, articles_df])
        semantic_search_btn.click(handle_semantic_search, [collection_dropdown, semantic_search_box], [articles_df])
        collection_dropdown.change(update_collection_desc, collection_dropdown, collection_desc_md)

        def update_tag_filter_dropdown(collection_name):
            collection = manager.get_collection(collection_name)
            if not collection:
                return gr.update(choices=[])
            tag_names = [tag.name for tag in collection.tags.values()]
            return gr.update(choices=tag_names)
        collection_dropdown.change(update_tag_filter_dropdown, [collection_dropdown], [tag_filter_dropdown])

        # Auto-refresh articles table on collection, search, or tag filter change
        collection_dropdown.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df])
        search_box.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df])
        tag_filter_dropdown.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df]) 