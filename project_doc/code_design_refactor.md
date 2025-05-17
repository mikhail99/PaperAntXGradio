PaperAntGradio/
├── app.py                        # Main entry point, Gradio Blocks context, state, and wiring
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── data_models.py
│   ├── collections_manager.py
│   ├── article_manager.py
│   ├── copilot_service.py
│   ├── chroma_service.py
│   └── utils.py
├── ui/
│   ├── __init__.py
│   ├── ui_collections.py         # Collections Management UI + callbacks
│   ├── ui_articles.py            # Article Management UI + callbacks
│   ├── ui_copilot.py             # AI Copilot UI + callbacks
│   ├── ui_paperqa.py             # PaperQA UI + callbacks
│   ├── ui_mindmap.py             # MindMap UI + callbacks
│   └── custom_css.py             # Custom CSS/JS for the app
├── state/
│   ├── __init__.py
│   └── state.py                  # Shared gr.State variables and helpers
├── data/
│   └── chroma_db_store/          # ChromaDB data
├── populate_mock_data.py
├── debug_articles.py
└── project_doc/
    └── GradioModular.md
