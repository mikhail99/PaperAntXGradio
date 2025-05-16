# PaperAnt X

PaperAnt X is an AI-powered research article management system with a modern Gradio UI.

## Requirements
- Python 3.9+
- [VSCode](https://code.visualstudio.com/) (recommended)
- [Conda](https://docs.conda.io/) or venv for environment management

## Setup
1. Clone the repository and activate your environment (e.g., `conda activate PaperAntGradio`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app in reload mode for fast development:
   ```bash
   gradio app.py --reload
   ```
4. Open the provided local URL in your browser.

## Development Notes
- Use Gradio's reload mode to see UI changes instantly.
- The app is structured for modularity and future expansion.
- See `project_doc/` for design and planning docs. 