import gradio as gr
from core.collections_manager import CollectionsManager
from core.article_manager import ArticleManager
from core.data_models import Collection, Article
import html

# Initialize managers (in a real app, you might want to pass these in or use a singleton)
manager = CollectionsManager(persist_directory="data/chroma_db_store")
article_manager = ArticleManager(manager)

# Helper for table
filtered_articles = []

def get_collection_description(collection_id):
    c = manager.get_collection(collection_id)
    return c.description if c else ""

def articles_table_value(articles):
    global filtered_articles
    filtered_articles = articles  # Store for selection callback
    data = []
    for a in articles:
        rating_icon = {
            "favorite": "★",
            "accept": "✓",
            "reject": "✗",
            None: "",
            "": "",
        }.get(a.rating, "")
        tags_text = ", ".join(a.tags)
        data.append([a.title, ", ".join(a.authors), rating_icon, tags_text])
    headers = ["Title", "Authors", "Rating", "Tags"]
    return {"data": data, "headers": headers}

def get_articles_for_collection(collection_id):
    collection = manager.get_collection(collection_id)
    if not collection:
        return []
    return list(collection.articles.values())

def create_articles_tab(state):
    with gr.TabItem("📖 Articles"):
        # --- Toolbar Row ---
        collection_options = [(c.name, c.id) for c in manager.get_all_collections() if not c.archived]
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                collection_dropdown = gr.Dropdown(
                    choices=collection_options,
                    label="Collection",
                    value=collection_options[0][1] if collection_options else None,
                    interactive=True,
                )
                collection_desc_md = gr.Markdown("<i>Select a collection to see its description.</i>")
            with gr.Column(scale=1):
                search_box = gr.Textbox(label="🔍 Search articles...", placeholder="Title, Abstract", scale=2)
                tag_filter_dropdown = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="🏷️ Filter by tags",
                )
            with gr.Column(scale=2):
                semantic_search_box = gr.Textbox(label="🧠 Semantic Search...", placeholder="Natural language query", scale=2)
                semantic_search_btn = gr.Button("Semantic Search", elem_id="semantic-search-btn")

        # --- Articles Table ---
        articles_df = gr.Dataframe(
            value=articles_table_value([]),
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
        def handle_article_select(evt: gr.SelectData, collection_id):
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

        def auto_refresh_articles(collection_id, search_text, selected_tags):
            try:
                if not collection_id:
                    return articles_table_value([])
                articles = get_articles_for_collection(collection_id)
                if search_text.strip():
                    search_lower = search_text.strip().lower()
                    articles = [a for a in articles if search_lower in a.title.lower() or search_lower in a.abstract.lower()]
                if selected_tags:
                    articles = [a for a in articles if any(tag in a.tags for tag in selected_tags)]
                return articles_table_value(articles)
            except Exception as e:
                print(f"Error filtering articles: {str(e)}")
                return articles_table_value([])

        def handle_save_article(collection_id, article_id, notes, tags_str, rating):
            if not article_id:
                return "**No article selected**", articles_table_value(get_articles_for_collection(collection_id))
            try:
                article = article_manager.get_article(collection_id, article_id)
                if not article:
                    return "**Article not found**", articles_table_value(get_articles_for_collection(collection_id))
                article.notes = notes
                article.tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                article.rating = rating
                updated = article_manager.update_article(collection_id, article)
                status = "**Article updated**" if updated else "**Failed to update article**"
                updated_articles = get_articles_for_collection(collection_id)
                return status, articles_table_value(updated_articles)
            except Exception as e:
                error_msg = f"**Error updating article: {str(e)}**"
                print(error_msg)
                updated_articles = get_articles_for_collection(collection_id)
                return error_msg, articles_table_value(updated_articles)

        def handle_semantic_search(collection_id, query):
            if not collection_id or not query.strip():
                return articles_table_value([])
            try:
                articles = manager.search_articles(collection_id, query, limit=20)
                return articles_table_value(articles)
            except Exception as e:
                print(f"Error in semantic search: {str(e)}")
                return articles_table_value([])

        def update_collection_desc(collection_id):
            desc = get_collection_description(collection_id)
            return desc or "<i>No description available.</i>"

        # Bind callbacks
        articles_df.select(handle_article_select, [collection_dropdown], [notes_box, tags_edit_box, rating_radio, state["selected_article_id"], article_title_md, article_abstract_md])
        save_article_btn.click(handle_save_article, [collection_dropdown, state["selected_article_id"], notes_box, tags_edit_box, rating_radio], [article_status, articles_df])
        semantic_search_btn.click(handle_semantic_search, [collection_dropdown, semantic_search_box], [articles_df])
        collection_dropdown.change(update_collection_desc, collection_dropdown, collection_desc_md)

        def update_tag_filter_dropdown(collection_id):
            collection = manager.get_collection(collection_id)
            if not collection:
                return gr.update(choices=[])
            tag_names = [tag.name for tag in collection.tags.values()]
            return gr.update(choices=tag_names)
        collection_dropdown.change(update_tag_filter_dropdown, [collection_dropdown], [tag_filter_dropdown])

        # Auto-refresh articles table on collection, search, or tag filter change
        collection_dropdown.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df])
        search_box.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df])
        tag_filter_dropdown.change(auto_refresh_articles, [collection_dropdown, search_box, tag_filter_dropdown], [articles_df]) 