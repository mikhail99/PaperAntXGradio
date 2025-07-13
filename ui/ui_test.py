import gradio as gr

agents = [
    {"name": "Finance News Assistant", "desc": "STUB", "icon": "🟣"},
    {"name": "Query Research Assistant", "desc": "STUB", "icon": "🟪"},
]

def select_agent(agent_name):
    return agent_name, f"Selected: {agent_name}"

def create_ui_test_tab():
    with gr.TabItem("🧠 UI Test"):
        with gr.Blocks(css="""
        .agent-btn {
            display: flex;
            align-items: center;
            border: 2px solid #eee;
            border-radius: 12px;
            padding: 10px 20px;
            margin: 8px 0;
            background: #faf9fa;
            font-size: 1.1em;
            cursor: pointer;
            transition: border 0.2s, background 0.2s, box-shadow 0.2s;
            font-weight: 500;
            box-shadow: none;
        }
        .agent-btn:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            background: #f5f5ff;
        }
        .agent-btn.selected {
            border: 2px solid #ff9900;
            background: #fff7e6;
            box-shadow: 0 2px 12px rgba(255,153,0,0.08);
        }
        .agent-icon {
            font-size: 1.5em;
            margin-right: 16px;
        }
        .agent-info {
            display: flex;
            flex-direction: column;
        }
        .agent-name {
            font-weight: bold;
        }
        .agent-desc {
            font-size: 0.95em;
            color: #888;
        }
        """) as demo:
            selected_agent = gr.State(agents[0]["name"])
            agent_status = gr.Markdown()

            with gr.Column():
                btns = []
                for agent in agents:
                    btn = gr.Button(
                        value=f"<span class='agent-icon'>{agent['icon']}</span><span class='agent-info'><span class='agent-name'>{agent['name']}</span><span class='agent-desc'>{agent['desc']}</span></span>",
                        elem_classes="agent-btn"
                    )
                    btns.append(btn)

            def handle_click(agent_idx):
                # Mark the clicked button as selected
                return agents[agent_idx]["name"], f"Selected: {agents[agent_idx]['name']}"

            for idx, btn in enumerate(btns):
                btn.click(
                    fn=lambda _, idx=idx: handle_click(idx),
                    inputs=None,
                    outputs=[selected_agent, agent_status],
                    show_progress=False,
                    queue=False
                )

