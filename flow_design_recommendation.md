# Flow Design Recommendation: Single vs. Two Flows

This document analyzes two primary design patterns for integrating Robin's discovery features with the proposal agent: a single, unified workflow versus two separate, interconnected workflows.

---

## Option 1: Single, Unified Flow

In this model, we would have one main workflow with a "discovery mode" switch, as outlined in the initial integration plan.

*   **User Experience**: The user provides a topic, ticks a box for "Enable Discovery," and the system runs end-to-end, from hypothesis generation to the final proposal.
*   **Analogy**: A fully automated assembly line. Raw materials (the topic) go in one end, and a finished product (the proposal) comes out the other.

### ✅ Pros:
*   **Seamless & Impressive**: Offers a powerful, "one-click" experience that can feel almost magical.
*   **Tight Integration**: State is passed seamlessly in memory between the discovery and proposal writing phases, potentially allowing for richer context.
*   **Simplicity of Use**: For the end-user, there is only one process to learn and manage.

### ❌ Cons:
*   **Rigid & Inflexible**: The user is locked into a linear process. Scientific research is rarely linear. A researcher might want to explore several disease areas first, reflect on the results, and then decide which single idea to turn into a proposal. This single flow doesn't accommodate that "explore-then-decide" pattern well.
*   **High Complexity**: The code for the orchestrator becomes significantly more complex, with many conditional branches. This makes it harder to debug, maintain, and extend.
*   **Long Runtimes**: A full end-to-end run could take a very long time, which can be a poor user experience, especially if they only wanted the initial discovery results.

---

## Option 2: Two Separate, Interconnected Flows

In this model, we would create two distinct functionalities, likely presented as separate tabs in the UI.

1.  **Discovery Flow (`The Scout`)**: Takes a broad topic/disease, runs the Robin-based analysis, and produces a ranked list of research opportunities (assays, candidates) that can be saved.
2.  **Proposal Flow (`The Architect`)**: The existing proposal agent, enhanced to accept its input in two ways:
    *   A manually entered topic (current functionality).
    *   A saved research opportunity from the Discovery Flow.

### ✅ Pros:
*   **Mirrors Real-World Research**: This model aligns perfectly with how researchers work: broad exploration followed by deep, focused work. It allows for human reflection and decision-making between steps.
*   **Modular & Maintainable**: Each flow is self-contained. The code is simpler, cleaner, and much easier to manage and test independently.
*   **Flexibility**: Users can use the tools independently. They can use the Scout to generate ideas for a presentation, or use the Architect to write a proposal for an idea they already had.
*   **Faster Perceived Performance**: Users can run the shorter "Discovery" flow first to get quick, high-level insights before committing to the longer "Proposal" generation.

### ❌ Cons:
*   **Requires a Handoff**: The user must take an explicit action: running the Discovery flow, saving the result, and then loading that result into the Proposal flow. It's less automated than the single-flow model.

---

## 🏆 Recommendation: Two Separate Flows

**It is strongly recommended to implement this as two separate, interconnected flows.**

The benefits in terms of flexibility, modularity, and alignment with the actual scientific research process far outweigh the cost of a slightly less automated user experience. A researcher's critical thinking and decision-making are essential parts of the process, and the two-flow model respects and empowers that, while a single, rigid flow tries to automate it away.

By building two distinct flows, you create two valuable, standalone tools that become even more powerful when used together. This approach is more robust, more user-centric, and will be far easier to build upon in the future. 