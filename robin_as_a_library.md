# `robin` as a Library: An Architectural Deep Dive

You've asked a critical question: "Are we planning to call `robin` like a library?"

The answer is **yes**, and this is a deliberate design choice based on how the `robin` project is built. While it's a complex system, it has been intentionally designed to be importable and usable as a programmatic module.

---

### Evidence from the `robin` Repository

Two key pieces of evidence from the `robin` repository itself confirm that this is the intended approach:

**1. The `robin` README.md explicitly states it:**
> "While this guide focuses on the `robin_demo.ipynb` notebook, the `robin` Python module (in the `robin/` directory) can be imported and its functions (`experimental_assay`, `therapeutic_candidates`, `data_analysis`) can be used programmatically in your own Python scripts for more customized workflows."

This is a clear invitation from the authors to use their tool as a building block in larger applications.

**2. The `robin/__init__.py` file defines a clean public API:**
```python
# robin/robin/__init__.py
from .analyses import data_analysis
from .assays import experimental_assay
from .candidates import therapeutic_candidates
from .configuration import RobinConfiguration

# Define the public API for 'from src import *'
__all__ = [
    "RobinConfiguration",
    "data_analysis",
    "experimental_assay",
    "therapeutic_candidates",
]
```
By explicitly defining `__all__`, the authors are providing a stable, public-facing contract. They are telling us, "These are the functions you can safely import and call."

---

### The Role of the `RobinService` Wrapper

Your question is insightful because simply importing `robin` isn't enough. It's a "heavy" library with side effects (like creating files and directories) and complex configuration. This is where the `RobinService` wrapper becomes essential.

The wrapper's job is to be the **only** part of our application that knows about the complexity of `robin`. It acts as an **adapter**, managing these three messy details:

1.  **Input Complexity**: Our application doesn't need to know how to create a `RobinConfiguration` object. It just passes a simple dictionary to `RobinService`, which handles the complex setup.

2.  **Execution Complexity**: The `RobinService` will handle the `async` calls to `experimental_assay` and `therapeutic_candidates`. It will manage the long-running nature of these calls.

3.  **Output Complexity**: This is the most important part. The `robin` functions don't *return* results directly; they write them to files (`robin_output/DISEASE_NAME.../`). The `RobinService`'s job is to:
    *   Know where to find these output files.
    *   Read the CSVs and text files after the `robin` function completes.
    *   Parse and transform that raw file data into our clean, predictable `ResearchOpportunity` objects.

So, while we are calling `robin` like a library, we are wisely insulating the rest of our application from its messy side effects by using the `RobinService` as a protective barrier. This gives us the best of both worlds: we leverage the power of the external tool without letting its complexity leak into our core application logic. 