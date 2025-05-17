import gradio as gr
from core.collections_manager import CollectionsManager
from core.data_models import Collection

manager = CollectionsManager(persist_directory="data/chroma_db_store")

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

def create_collections_tab(state):
    with gr.TabItem("Collections Management"):
        collections_df = gr.Dataframe(
            headers=["ID", "Name", "Description", "Archived", "Tags"],
            value=collections_to_rows(manager.get_all_collections()),
            label="Collections",
            interactive=True,
            max_height=200,
        )
        name_box = gr.Textbox(label="Collection Name")
        desc_box = gr.Textbox(label="Collection Description", lines=2)
        tags_box = gr.Textbox(label="Tags (comma-separated, e.g., topic/subtopic, method)")
        create_btn = gr.Button("Create New Collection")
        update_btn = gr.Button("Update Selected Collection")
        archive_btn = gr.Button("Archive Selected Collection")
        status = gr.Markdown("")

        def refresh_collections():
            try:
                return collections_to_rows(manager.get_all_collections())
            except Exception as e:
                print(f"Error refreshing collections: {str(e)}")
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
                return gr.update(), error_msg

        def handle_select_collection(evt: gr.SelectData):
            try:
                rows = collections_to_rows(manager.get_all_collections())
                if not evt.index or evt.index[0] is None or evt.index[0] >= len(rows):
                    return gr.update(), gr.update(), gr.update(), ""
                row_index = evt.index[0]
                row = rows[row_index]
                cid, name, desc, archived, tags = row
                return name, desc, tags, cid
            except Exception as e:
                print(f"Error selecting collection: {str(e)}")
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
                return gr.update(), error_msg

        # Bind callbacks
        create_btn.click(handle_create_collection, [name_box, desc_box, tags_box], [collections_df, status])
        collections_df.select(handle_select_collection, None, [name_box, desc_box, tags_box, state["selected_collection_id"]])
        update_btn.click(handle_update_collection, [state["selected_collection_id"], name_box, desc_box, tags_box], [collections_df, status])
        archive_btn.click(handle_archive_collection, [state["selected_collection_id"]], [collections_df, status]) 