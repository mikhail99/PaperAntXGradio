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

#copilot-reload-button {
    max-width: 2.5em; /* Adjust size of reload button */
    min-width: 2.5em !important;
}
'''
