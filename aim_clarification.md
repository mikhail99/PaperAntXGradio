# Aim Clarification: Robin vs. @proposal_agent_dspy

It's completely understandable to see these two systems as similar. Both read literature and produce research-related text. However, their fundamental goals are different. Think of it as the difference between a **Scout** and an **Architect**.

---

### `@proposal_agent_dspy` is an Architect 🏛️

An architect is given a specific plot of land and a goal (e.g., "build a three-bedroom house"). Their job is to create a single, detailed, and executable blueprint.

Your current `@proposal_agent_dspy` does exactly this:

*   **Input**: A defined research topic (the "plot of land").
*   **Process**: It surveys the immediate surroundings (literature review on that topic), finds a precise spot to build (a specific knowledge gap), and then drafts a complete, structured plan to build on it.
*   **Output**: A single, detailed research proposal (the "blueprint"). Its aim is to **formalize one idea into a concrete plan.**

---

### `robin` is a Scout 🤠

A scout is sent into a vast, unexplored territory with a broad mission (e.g., "find the best places to establish a new settlement"). They don't draw a blueprint for a house. Instead, they explore widely and return with a map of ranked opportunities.

`robin` does this for scientific research:

*   **Input**: A broad, high-level problem (the "unexplored territory," e.g., a disease).
*   **Process**: It explores the entire landscape, identifying many different potential paths forward (hypotheses, assays, drug targets). It then evaluates and ranks these opportunities based on their potential.
*   **Output**: A ranked list of many different, high-potential research ideas (a "map of ranked opportunities"). Its aim is to **discover and prioritize many possible ideas.**

---

### Why the Integration Plan Combines Them

The integration plan aims to create a seamless workflow from scouting to architecture.

1.  **First, you act as the Scout (`robin`)**: You enter a broad disease area, and the system generates a ranked list of the most promising research avenues. You are no longer starting with a guess. You are starting with a data-driven portfolio of validated ideas.

2.  **Then, you become the Architect (`@proposal_agent_dspy`)**: You select the #1 ranked idea from the scout's map. The system then takes that single, high-potential idea and drafts the detailed blueprint—the full technical proposal.

Without this integration, you have two separate tools. With the integration, you have a single, powerful system that takes you from a high-level question ("What should we even work on for this disease?") all the way to a fundable plan ("Here is the detailed proposal for the most promising approach."). 