# Agent Missions

Autonomous agents in IngestForge can perform multi-hop reasoning, synthesize contradictory information, and perform web searches to augment your local knowledge base.

## 🎯 Defining an Objective

To start a mission, you need a clear research goal.

**Good Objectives**:
*   "Synthesize the implications of the 2024 zoning changes on affordable housing targets."
*   "Trace the character arc of the protagonist in the first three chapters."
*   "Compare the CVSS scores of the last five vulnerabilities discovered in the system."

## 🚀 Launching a Mission

### Web Portal
1.  Navigate to **Agent Mission Control**.
2.  Enter your objective in the **Research Objective** box.
3.  Click **Generate Research Plan**.
4.  Review the roadmap and click **Execute Mission**.

### CLI
```bash
ingestforge agent run "Your research objective" --max-steps 10
```

## 🧠 Monitoring Reasoning

During a mission, you can monitor the agent's "Thoughts" and "Actions" in the **Reasoning Chain**. This provides full transparency into how the agent is reaching its conclusions.

### Actions the Agent can take:
*   `search`: Query the local vector store.
*   `web_search`: Fetch information from the internet.
*   `read_chunk`: Analyze a specific piece of text in depth.
*   `extract_entities`: Identify key names, dates, or concepts.

## 📝 Final Synthesis

Once the agent completes its tasks, it will present a **Final Synthesis**. 

*   **Verified Claims**: The agent links its statements back to source documents using the **Verification Shield**.
*   **Export**: You can export the final research report as a Markdown file for use in other applications.
