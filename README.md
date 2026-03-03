# Agent-as-a-Judge: Backend

This repository contains the core logic, multi-agent workflows, and statistical evaluation scripts for the **[Agent-as-a-Judge](https://github.com/CuzImAram/agent-as-a-judge)** framework. The system is designed to simulate human reasoning in RAG evaluation by utilizing specialized agentic pipelines to assess response utility.

## 📂 Repository Structure

```text
.
├── data/
│   ├── graphs/ # Visualization of outputs
│   ├── output_final/ # Final results of the evaluation
│   │   ├── agent_judgement_120B_GPT_OSS/
│   │   ├── agent_judgement_Qwen3_235/
│   │   ├── agent_judgement_zeroshot_Qwen3_235/
│   │   ├── comp_fully/
│   │   ├── comp_fully_without_ref/
│   │   ├── comp_zeroshot/
│   │   ├── comp_zeroshot_without_ref/
│   │   │   ├── compared_ratings_agent_comp/
│   │   │   ├── compared_ratings_agent_comp_qwen/
│   │   │   ├── compared_ratings_agent_comp_zeroshot-judgements/
│   │   │   ├── krippendorff_topic_all.json
│   │   │   ├── krippendorff_topic_majority.json
│   │   │   ├── krippendorff_topic_majority_qwen.json
│   │   │   ├── krippendorff_topic_zeroshot_judgement.json
│   │   └── grade/
│   └── raw/ # Original data for CrowdRAG25
└── src/
    ├── n8n/
    │   ├── agent-as-a-judge-comp.json # Agentic workflow for pairwise comparison
    │   ├── agent-as-a-judge-grade.json # Agentic workflow for pointwise grading
    │   ├── create_n8n.ps1
    │   └── data/
    └── python/
        ├── data_sender.py
        ├── krippendorff_eval.py
        ├── ratings_eval.py
        └── scripts/

```

## 🚀 Setup and Installation

### 1. n8n Instance (Docker Setup)

To simplify the installation of the required environment, a PowerShell script is provided.

- Run the `create_n8n.ps1` script to automatically create a **Docker container** and install a local **n8n** instance.
- This ensures that the orchestration environment is consistent with the experimental setup described in the paper.

### 2. Workflows

The agentic reasoning trajectories are provided as **pure JSON files** located in the `src/n8n/` directory, as referenced in the methodology of the project.

- Open your n8n instance and use the **"Import from File"** feature to load the Grading and Comparison pipelines.
- Configure your model credentials (e.g., Qwen or GPT-OSS) within the agent nodes.

## 📊 Running the Evaluation

The scripts in the `src/` directory facilitate the automated benchmarking of the AI judges against the human gold standard.

1. **Configure API URLs:** Update the webhook endpoints in `src/data_sender.py`.
2. **Execute Analysis:** Run the `src/python/krippendorff_eval.py` script to compute the inter-rater reliability scores.
3. **View Results:** Each execution generates a specific configuration folder within `data/output/`, containing the detailed agent **judgements** and the resulting **Krippendorff $\alpha$** values.
