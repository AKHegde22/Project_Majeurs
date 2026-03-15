# 🧠 Neuro-Symbolic Deep Research

> **Democratizing Autonomous Deep Research: A Neuro-Symbolic Framework for 7B-Parameter Models via Grammar-Constrained Decoding**

A local, privacy-first autonomous research agent that runs entirely on consumer-grade hardware. It combines a quantized **Llama-3-8B** model with **Grammar-Constrained Decoding (GBNF)** and a **Critic-Planner** feedback loop to deliver reliable, multi-step deep research — no cloud APIs or expensive fine-tuning required.

---

## 📌 Overview

Large frontier models (GPT-4, Claude 3) have demonstrated impressive agentic capabilities, but their computational cost and privacy implications make them unsuitable for edge deployment. Small Language Models (SLMs) in the 7B parameter class are far more accessible, yet they frequently fail in agentic scenarios — generating invalid tool calls, hallucinating functions, or losing coherence over long tasks.

This project bridges that **reliability gap** through a neuro-symbolic architecture:

1. **Grammar-Constrained Decoding (GBNF)** — Forces the LLM to emit strictly valid JSON at the *logit level*, guaranteeing syntactic correctness for every tool call.
2. **Critic-Planner Loop** — A recursive feedback mechanism where a neural critic evaluates each retrieved snippet for relevance before it enters the context window, preventing pollution and hallucination drift.

Together these techniques allow a frozen, quantized 7B model to perform complex, multi-hop research tasks that cause strong industry baselines (e.g., LangChain ReAct) to fail catastrophically.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **100% Syntactic Success Rate** | GBNF grammar constraints guarantee valid JSON output on every inference call |
| **Critic-Planner Feedback Loop** | Semantic verification step filters irrelevant search results before they pollute context |
| **Fully Local Execution** | Runs on CPU/GPU via `llama-cpp-python`; no data leaves your machine |
| **Configurable Search Backend** | Supports **DuckDuckGo** (free, no API key) or **Tavily** (richer results) |
| **Streamlit UI** | Clean web interface with progress tracking, source citations, and raw evidence log |
| **Adjustable Research Depth** | Slider to control the number of planning → search → verify iterations (2–7) |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│           PLANNER (Neural)          │
│  Llama-3-8B + GBNF Grammar Gate     │
│  Output: { "rationale": ...,        │
│            "query": ... }  ← JSON   │
└──────────────────┬──────────────────┘
                   │ search query
                   ▼
         ┌──────────────────┐
         │   Search Tool    │
         │ DuckDuckGo/Tavily│
         └────────┬─────────┘
                  │ raw results + URLs
                  ▼
    ┌─────────────────────────────┐
    │        CRITIC (Neural)      │
    │  Binary: RELEVANT / NOISE   │
    └──────────────┬──────────────┘
                   │ verified facts only
                   ▼
          Knowledge Context
                   │  (repeat N iterations)
                   ▼
    ┌─────────────────────────────┐
    │     SYNTHESIZER (Neural)    │
    │  Final Markdown Report      │
    └─────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A machine with at least **8 GB of available RAM** (the Q4_K_M model requires ~4.9 GB; 16 GB total system RAM recommended to comfortably accommodate the model, OS, and 8192-token context window)
- The quantized model file: **`Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`**
  - Download from [Hugging Face — bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF)
  - Place the `.gguf` file in the **project root directory** (alongside `app.py`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AKHegde22/Project_Majeurs.git
cd Project_Majeurs

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install streamlit llama-cpp-python langchain-community duckduckgo-search
```

> **GPU Acceleration (optional):** To enable GPU layers, install `llama-cpp-python` with CUDA/Metal support. See the [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration) for details.

---

## 🖥️ Usage

### Option A — DuckDuckGo backend (no API key required)

```bash
streamlit run app.py
```

### Option B — Tavily backend (richer, more structured results)

1. Get a free API key at [tavily.com](https://tavily.com)
2. Open `app3.py` and paste your key:
   ```python
   os.environ["TAVILY_API_KEY"] = "your-key-here"
   ```
3. Run:
   ```bash
   streamlit run app3.py
   ```

The app will open in your browser at `http://localhost:8501`.

---

## 🗂️ File Structure

```
Project_Majeurs/
├── app.py                                    # Main app — DuckDuckGo backend, clean source display
├── app3.py                                   # Tavily backend with verbose terminal logging
├── app4.py                                   # DuckDuckGo backend with raw evidence expander
├── Meta-Llama-3-8B-Instruct.Q4_K_M.gguf    # Downloaded model — place here (not in repo)
├── docs/                                     # Project documentation, literature review, slides
└── README.md
```

---

## ⚙️ Configuration

All runtime settings are available in the **sidebar** of the Streamlit UI:

| Setting | Default | Description |
|---|---|---|
| Research Depth (Iterations) | 4 | Number of plan → search → verify cycles |
| Sources per Search *(app3.py)* | 5 | Number of URLs fetched per Tavily query |

---

## 📄 Research Paper

This project accompanies an IEEE-format conference paper:

> *"Democratizing Autonomous Deep Research: A Neuro-Symbolic Framework for 7B-Parameter Models via Grammar-Constrained Decoding"*

A draft of the paper and the full literature review are available in the [`docs/`](docs/) folder.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — Web UI
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Local LLM inference with GBNF grammar support
- [Meta Llama 3 8B Instruct](https://ai.meta.com/blog/meta-llama-3/) — Base language model (quantized Q4_K_M)
- [LangChain Community](https://github.com/langchain-ai/langchain) — Search tool integrations (DuckDuckGo, Tavily)
