import gradio as gr
from typing import List, Optional

def process_prompt_with_references(prompt: str, selected_files: List[str]) -> str:
    """Replace @1, @2, etc. with actual file content"""
    if not selected_files:
        return prompt
    
    processed_prompt = prompt
    for i, file_path in enumerate(selected_files):
        reference = f"@{i+1}"
        if reference in processed_prompt:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_name = file_path.split('/')[-1]
                replacement = f"\n--- Content from {file_name} ---\n{content}\n--- End of {file_name} ---\n"
                processed_prompt = processed_prompt.replace(reference, replacement)
            except Exception as e:
                processed_prompt = processed_prompt.replace(reference, f"[Error reading {file_path}: {str(e)}]")
    
    return processed_prompt

def generate_selected_files_html(selected_files: List[str]) -> str:
    """Generate HTML display for selected files with @1, @2 references"""
    if not selected_files:
        return "<div class='selected-files'>No files selected. Select files below to reference them as @1, @2, etc.</div>"
    
    file_list = []
    for i, file_path in enumerate(selected_files):
        file_name = file_path.split('/')[-1]
        file_list.append(f"<span class='file-reference'>@{i+1}</span>: {file_name}")
    
    return f"<div class='selected-files'><strong>Selected:</strong><br>{'<br>'.join(file_list)}</div>"

def generate_file_preview_text(selected_files: List[str]) -> str:
    """Generate preview text showing file references"""
    if not selected_files:
        return "Select documents to see references..."
    
    preview_parts = []
    for i, file_path in enumerate(selected_files):
        file_name = file_path.split('/')[-1]
        preview_parts.append(f"**@{i+1}**: {file_name}")
    
    return "\n".join(preview_parts)

def get_file_reference_css() -> str:
    """Return CSS styles for file reference components"""
    return """
    .selected-files {
        background: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-radius: 6px !important;
        padding: 8px !important;
        margin: 8px 0 !important;
        font-size: 0.85em !important;
    }
    .file-reference {
        color: #6c757d !important;
        font-family: monospace !important;
    }
    """

def create_file_reference_examples() -> List[List[str]]:
    """Return example prompts that demonstrate @1, @2 usage"""
    return [
        ["I want to research LLMs in education"],
        ["Based on @1, help me develop a methodology"],
        ["Using @1 and @2, create a research timeline"],
        ["Compare the findings in @1 with @2"],
        ["Summarize the key points from @1, @2, and @3"]
    ] 