# Implementation Plan: PaperAnt X

## Iteration 1: Project Setup & Core Structure
- [x] Initialize project repository and directory structure as per code_design.md
- [x] Create `requirements.txt` with Gradio v4.x, Pydantic, ChromaDB, and other essentials
- [x] Scaffold `app.py` with a minimal Gradio Blocks app (title, empty tabs)
- [x] Add a README with setup and run instructions
- [x] Validation: Run `gradio app.py --reload` and confirm the app loads with the correct structure and no errors

## Iteration 2: Data Models & Basic State
- [x] Create empty core module files: `data_models.py`, `collection_manager.py`, `article_manager.py`, `chroma_service.py`, `copilot_service.py`, `utils.py`
- [x] Add a placeholder for persistent data directory
- [x] Implement Pydantic models for Collection, Article, Tag in `data_models.py`
- [x] Add in-memory state management in `app.py` using `gr.State` for collections and articles
- [x] Display collections and articles as empty DataFrames in the UI
- [x] Validation: Add dummy data to state and confirm it appears in the UI tables

## Iteration 3: Collection Management UI & Logic
- [ ] Implement UI for viewing, creating, editing, and archiving collections (Tab 1)
- [ ] Implement backend logic in `collection_manager.py` for CRUD operations
- [ ] Wire up callbacks for create/update/archive actions
- [ ] Parse and display tags from comma-separated input
- [ ] Validation: Create, edit, and archive collections in the UI; verify state updates and UI refresh

## Iteration 4: Article Management UI & Logic
- [ ] Implement UI for viewing, filtering, and managing articles (Tab 2)
- [ ] Implement backend logic in `article_manager.py` for CRUD, rating, tagging, and notes
- [ ] Wire up callbacks for article selection, detail view, and updates
- [ ] Implement manual and batch article addition (UI and backend)
- [ ] Validation: Add, edit, and filter articles; verify detail view and state updates

## Iteration 5: ChromaDB Integration
- [ ] Implement `chroma_service.py` for persistent storage and vector search
- [ ] Connect collection and article managers to ChromaDB for data persistence
- [ ] Add logic to load/save state from ChromaDB on app start/stop
- [ ] Validation: Add and retrieve collections/articles; confirm persistence across app restarts

## Iteration 6: AI Copilot Integration
- [ ] Implement backend LLM prompt/response logic in `copilot_service.py`
- [ ] Add UI for Copilot chat (Tab 4) using `gr.Chatbot`
- [ ] Wire up user input, context passing, and backend response display
- [ ] Validation: Ask questions in Copilot tab and receive relevant responses

## Iteration 7: Search & Tag Filtering
- [ ] Implement full-text search and tag-based filtering in article management
- [ ] Add UI controls for search and tag filters
- [ ] Validation: Search/filter articles and confirm correct results in the UI

## Iteration 8: Polish, Error Handling & Documentation
- [ ] Add user feedback/status messages for all major actions
- [ ] Improve error handling and input validation throughout
- [ ] Update README and in-app help with usage instructions
- [ ] Validation: Test all workflows for robustness and clarity


