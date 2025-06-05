"""
Create a simple test PDF with sample academic content
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def create_test_pdf(filename="test_paper.pdf"):
    """
    Create a simple test PDF with sample academic paper content.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("A Novel Approach to Machine Learning Optimization", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Authors
    authors = Paragraph("John Doe<sup>1</sup>, Jane Smith<sup>2</sup>", styles['Normal'])
    story.append(authors)
    story.append(Spacer(1, 0.1*inch))
    
    # Affiliations
    affil = Paragraph("<sup>1</sup>Department of Computer Science, University A<br/><sup>2</sup>Institute of Technology, University B", styles['Normal'])
    story.append(affil)
    story.append(Spacer(1, 0.3*inch))
    
    # Abstract
    abstract_title = Paragraph("Abstract", styles['Heading1'])
    story.append(abstract_title)
    
    abstract_text = """
    This paper presents a novel optimization algorithm for machine learning models.
    Our approach combines gradient descent with adaptive learning rates to achieve
    faster convergence and better performance. We evaluate our method on several
    benchmark datasets and demonstrate significant improvements over existing
    techniques. The proposed algorithm reduces training time by 30% while
    maintaining or improving accuracy.
    """
    abstract_para = Paragraph(abstract_text, styles['Normal'])
    story.append(abstract_para)
    story.append(Spacer(1, 0.2*inch))
    
    # Keywords
    keywords = Paragraph("<b>Keywords:</b> machine learning, optimization, gradient descent, adaptive learning", styles['Normal'])
    story.append(keywords)
    story.append(Spacer(1, 0.3*inch))
    
    # Introduction
    intro_title = Paragraph("1. Introduction", styles['Heading1'])
    story.append(intro_title)
    
    intro_text = """
    Machine learning optimization is a fundamental challenge in artificial intelligence.
    Traditional gradient descent methods often suffer from slow convergence and local
    minima problems. Recent advances in adaptive learning rate algorithms have shown
    promise, but still face limitations in complex optimization landscapes.
    
    In this work, we propose AdaptiveGrad, a novel optimization algorithm that
    dynamically adjusts learning rates based on gradient history and curvature
    information. Our method addresses the key limitations of existing approaches
    while maintaining computational efficiency.
    """
    intro_para = Paragraph(intro_text, styles['Normal'])
    story.append(intro_para)
    story.append(Spacer(1, 0.2*inch))
    
    # Methodology
    method_title = Paragraph("2. Methodology", styles['Heading1'])
    story.append(method_title)
    
    method_text = """
    Our proposed AdaptiveGrad algorithm consists of three main components:
    
    1. Gradient History Tracking: We maintain a moving average of recent gradients
       to capture momentum information.
    
    2. Curvature Estimation: We use second-order information to estimate the
       local curvature of the loss surface.
    
    3. Adaptive Rate Adjustment: We combine gradient history and curvature
       information to dynamically adjust the learning rate for each parameter.
    
    The algorithm can be formally described as follows:
    θ(t+1) = θ(t) - α(t) * ∇L(θ(t))
    
    where α(t) is the adaptive learning rate computed using our proposed method.
    """
    method_para = Paragraph(method_text, styles['Normal'])
    story.append(method_para)
    story.append(Spacer(1, 0.2*inch))
    
    # Results
    results_title = Paragraph("3. Experimental Results", styles['Heading1'])
    story.append(results_title)
    
    results_text = """
    We evaluated AdaptiveGrad on three benchmark datasets: MNIST, CIFAR-10, and
    ImageNet. Our experiments show consistent improvements in both convergence
    speed and final accuracy compared to baseline methods.
    
    Key findings:
    - 30% faster convergence on average
    - 2-3% improvement in final accuracy
    - Better stability across different network architectures
    - Reduced sensitivity to hyperparameter choices
    """
    results_para = Paragraph(results_text, styles['Normal'])
    story.append(results_para)
    story.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    conclusion_title = Paragraph("4. Conclusion", styles['Heading1'])
    story.append(conclusion_title)
    
    conclusion_text = """
    We have presented AdaptiveGrad, a novel optimization algorithm that significantly
    improves upon existing methods. Our approach demonstrates the importance of
    combining gradient history with curvature information for effective optimization.
    Future work will explore applications to other domains and investigate theoretical
    convergence guarantees.
    """
    conclusion_para = Paragraph(conclusion_text, styles['Normal'])
    story.append(conclusion_para)
    
    # Build PDF
    doc.build(story)
    print(f"Test PDF created: {filename}")
    return filename

if __name__ == "__main__":
    try:
        create_test_pdf()
    except ImportError:
        print("Error: reportlab library not found. Install with: pip install reportlab")
        print("Creating a simple text file instead...")
        
        # Fallback: create a simple text file that can be manually converted to PDF
        with open("test_paper_content.txt", "w") as f:
            f.write("""A Novel Approach to Machine Learning Optimization

John Doe¹, Jane Smith²
¹Department of Computer Science, University A
²Institute of Technology, University B

Abstract

This paper presents a novel optimization algorithm for machine learning models.
Our approach combines gradient descent with adaptive learning rates to achieve
faster convergence and better performance. We evaluate our method on several
benchmark datasets and demonstrate significant improvements over existing
techniques. The proposed algorithm reduces training time by 30% while
maintaining or improving accuracy.

Keywords: machine learning, optimization, gradient descent, adaptive learning

1. Introduction

Machine learning optimization is a fundamental challenge in artificial intelligence.
Traditional gradient descent methods often suffer from slow convergence and local
minima problems. Recent advances in adaptive learning rate algorithms have shown
promise, but still face limitations in complex optimization landscapes.

In this work, we propose AdaptiveGrad, a novel optimization algorithm that
dynamically adjusts learning rates based on gradient history and curvature
information. Our method addresses the key limitations of existing approaches
while maintaining computational efficiency.

2. Methodology

Our proposed AdaptiveGrad algorithm consists of three main components:

1. Gradient History Tracking: We maintain a moving average of recent gradients
   to capture momentum information.

2. Curvature Estimation: We use second-order information to estimate the
   local curvature of the loss surface.

3. Adaptive Rate Adjustment: We combine gradient history and curvature
   information to dynamically adjust the learning rate for each parameter.

The algorithm can be formally described as follows:
θ(t+1) = θ(t) - α(t) * ∇L(θ(t))

where α(t) is the adaptive learning rate computed using our proposed method.

3. Experimental Results

We evaluated AdaptiveGrad on three benchmark datasets: MNIST, CIFAR-10, and
ImageNet. Our experiments show consistent improvements in both convergence
speed and final accuracy compared to baseline methods.

Key findings:
- 30% faster convergence on average
- 2-3% improvement in final accuracy
- Better stability across different network architectures
- Reduced sensitivity to hyperparameter choices

4. Conclusion

We have presented AdaptiveGrad, a novel optimization algorithm that significantly
improves upon existing methods. Our approach demonstrates the importance of
combining gradient history with curvature information for effective optimization.
Future work will explore applications to other domains and investigate theoretical
convergence guarantees.
""")
        print("Text content saved to: test_paper_content.txt")
        print("You can convert this to PDF using any online converter or LibreOffice.") 