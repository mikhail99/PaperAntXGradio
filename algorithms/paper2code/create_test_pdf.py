#!/usr/bin/env python3
"""
Create a test PDF for testing our PDF processing pipeline
"""

import fitz  # PyMuPDF

def create_test_pdf():
    # Create new PDF
    doc = fitz.open()
    page = doc.new_page()
    
    # Add realistic academic paper content with proper formatting
    text_sections = [
        ("ABSTRACT", "This paper presents a novel approach to automated document processing using machine learning techniques.\nWe introduce a three-stage pipeline that achieves state-of-the-art performance on benchmark datasets."),
        
        ("1. INTRODUCTION", "Document processing has become increasingly important in the digital age. Traditional methods rely on\nrule-based systems, but our approach leverages deep learning to extract meaningful information from\ncomplex documents. This work builds upon recent advances in transformer architectures."),
        
        ("2. METHODOLOGY", "We propose a three-stage pipeline:\n1. Text extraction using optical character recognition (OCR)\n2. Structure analysis using convolutional neural networks\n3. Content classification using transformer models\n\nThe methodology involves training a deep convolutional neural network on a dataset of 10,000 labeled\ndocuments. We use a ResNet-50 architecture with custom attention mechanisms for improved performance.\n\nOur approach processes documents through the following algorithm:\n- Algorithm 1: Document preprocessing\n- Algorithm 2: Feature extraction\n- Algorithm 3: Classification and ranking"),
        
        ("3. RESULTS", "Our approach achieved 95.2% accuracy on the test dataset, significantly outperforming baseline methods:\n- Baseline SVM: 78.3%\n- Random Forest: 82.1%\n- BERT baseline: 91.4%\n- Our method: 95.2%\n\nThe methodology was evaluated on three different document types: academic papers, technical reports,\nand legal documents."),
        
        ("4. DISCUSSION", "The results demonstrate the effectiveness of our proposed methodology for automated document processing.\nThe attention mechanism proves particularly useful for handling complex document layouts."),
        
        ("5. CONCLUSION", "This work demonstrates the effectiveness of our proposed methodology for automated document processing.\nFuture work will focus on extending the approach to multilingual documents and real-time processing."),
        
        ("REFERENCES", "[1] Smith, J. (2020). Document Analysis Techniques. Journal of AI Research, 15(3), 123-145.\n[2] Brown, A. et al. (2021). Transformer Models for Document Processing. ICML 2021.\n[3] Wilson, K. (2019). Deep Learning in Document Analysis. Nature Machine Intelligence.")
    ]
    
    # Insert text sections with proper spacing
    y_position = 50
    for header, content in text_sections:
        # Insert header
        page.insert_text((50, y_position), header, fontsize=12, color=(0, 0, 0))
        y_position += 25
        
        # Insert content
        page.insert_text((50, y_position), content, fontsize=10, color=(0, 0, 0))
        y_position += len(content.split('\n')) * 15 + 20  # Adjust for line breaks and spacing
    
    # Save PDF
    filename = 'test_research_paper_formatted.pdf'
    doc.save(filename)
    doc.close()
    
    print(f'Created {filename}')
    return filename

if __name__ == "__main__":
    create_test_pdf() 