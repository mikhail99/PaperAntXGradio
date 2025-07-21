CUSTOM_CSS = '''
/* Add your custom CSS here */

/* Card-like background and border for answer and history sections */
#answer-card, .qa-history-card {
  background: #fafbfc;
  border: 1px solid #e0e0e0;
  border-radius: 10px;
  box-shadow: 0 2px 8px 0 rgba(60,60,60,0.06);
  padding: 1.2em 1.5em;
  margin-bottom: 1em;
}

/* Make Q&A History radio look like a card */
.qa-history-card {
  background: #f5f7fa;
  border: 1px solid #e0e0e0;
  border-radius: 10px;
  box-shadow: 0 1px 4px 0 rgba(60,60,60,0.04);
  padding: 1em 1.2em;
}

/* Primary color for Get Report button */
button:has(> span:contains('Get Report')) {
  background: #ff9800 !important;
  color: #fff !important;
  border: none !important;
  font-weight: 600;
  box-shadow: 0 2px 8px 0 rgba(255,152,0,0.08);
}

/* Add icons to Tabs (using emoji as example) */
.gr-tabitem[data-testid="tabitem-Collections Management"] > div::before {
  content: '\1F4DA  '; /* 📚 */
  font-size: 1.1em;
}
.gr-tabitem[data-testid="tabitem-Article Management"] > div::before {
  content: '\1F4D6  '; /* 📖 */
  font-size: 1.1em;
}
.gr-tabitem[data-testid="tabitem-AI Copilot"] > div::before {
  content: '\1F916  '; /* 🤖 */
  font-size: 1.1em;
}
.gr-tabitem[data-testid="tabitem-PaperQA"] > div::before {
  content: '\1F4DD  '; /* 📝 */
  font-size: 1.1em;
}
.gr-tabitem[data-testid="tabitem-MindMap"] > div::before {
  content: '\1F5FA  '; /* 🗺️ */
  font-size: 1.1em;
}

/* Copilot Tab specific styles */
#copilot-main-container {
    height: calc(100vh - 250px); /* Adjust offset for header/tabs etc. */
}

#copilot-main-container > div { /* Target direct children columns */
    height: 100%;
}

#copilot-chat-column {
    display: flex;
    flex-direction: column;
    height: 100%;
}

#copilot_chatbot {
    flex-grow: 1;
    overflow-y: auto;
    height: 100%;
}

#agent-details-display pre code {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}

/* Agent List Styles */
#agent-list-container .agent-list-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.agent-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 20px;
}

.agent-item {
    display: flex;
    align-items: center;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border-color-primary);
    cursor: pointer;
    transition: background-color 0.2s, border-color 0.2s, box-shadow 0.2s;
    /* By not setting a background-color, it will inherit and work in dark mode. */
}

.agent-item:hover {
    background-color: var(--background-fill-secondary);
}

.agent-item.selected {
    border-color: var(--primary-500);
    /* background-color is removed to avoid contrast issues in dark mode */
    box-shadow: 0 0 0 1px var(--primary-500);
}

.agent-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin-right: 12px;
    flex-shrink: 0;
}

.agent-text {
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.agent-name {
    font-weight: 600;
    color: var(--body-text-color-strong);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.agent-description {
    font-size: 0.9em;
    color: var(--body-text-color);
    word-wrap: break-word;
    line-height: 1.3;
}

/* Agent Details Section Styles */
.agent-details-section {
    border-top: 1px solid var(--border-color-primary);
    padding-top: 16px;
    margin-top: 16px;
}

.agent-details-section h4 {
    margin: 0 0 12px 0;
    color: var(--body-text-color-strong);
    font-size: 1.1em;
    font-weight: 600;
}

.agent-details-content {
    color: var(--body-text-color);
    line-height: 1.4;
}

.agent-details-content h1,
.agent-details-content h2,
.agent-details-content h3 {
    color: var(--body-text-color-strong);
    margin: 12px 0 8px 0;
}

.agent-details-content pre {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
    overflow-x: auto;
    font-size: 0.9em;
}

.agent-details-content code {
    background: var(--background-fill-secondary);
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 0.9em;
}

.agent-details-content ul {
    margin: 8px 0;
    padding-left: 20px;
}

.agent-details-content li {
    margin: 4px 0;
}

/* Custom style for the discrete reload button */
.discrete-reload-button {
    background: transparent !important;
    border: 1px solid var(--border-color-primary) !important;
    color: var(--body-text-color) !important;
    box-shadow: none !important;
    font-weight: 500 !important;
}

.discrete-reload-button:hover {
    background: var(--background-fill-secondary) !important;
    border-color: var(--primary-500) !important;
    color: var(--body-text-color-strong) !important;
}

/* Hide the trigger textboxes but keep them in the DOM */
.hidden-trigger {
    display: none !important;
}
'''
