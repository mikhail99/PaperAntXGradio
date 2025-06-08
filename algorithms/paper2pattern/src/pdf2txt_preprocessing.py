import os
import pymupdf


def pdf_to_text_from_path(pdf_path, output_path):
    """
    Extracts text from the given PDF file and saves it to output_path.
    """
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    full_text = '\n'.join([line.strip() for line in full_text.splitlines() if line.strip()])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"Extracted text saved to {output_path}")


def batch_pdf_to_text(papers_dir='papers', outputs_dir='outputs'):
    """
    Recursively finds all PDFs in 'papers' subfolders and converts them to text in 'outputs/[folder_name]/full_text.txt'.
    """
    for root, dirs, files in os.walk(papers_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                # folder_name is the immediate subfolder under 'papers'
                rel_path = os.path.relpath(root, papers_dir)
                folder_name = rel_path.split(os.sep)[0]
                output_dir = os.path.join(outputs_dir, folder_name)
                output_path = os.path.join(output_dir, 'full_text.txt')
                if os.path.exists(output_path):
                    print(f"Skipping {folder_name}: full_text.txt already exists.")
                    continue
                try:
                    pdf_to_text_from_path(pdf_path, output_path)
                except Exception as e:
                    print(f"Failed to process {pdf_path}: {e}")


if __name__ == "__main__":
    batch_pdf_to_text()