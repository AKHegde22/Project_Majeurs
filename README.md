# Project_Majeurs
Major Project

draft 1
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{listings}
\usepackage{booktabs}
\usepackage{url}
\usepackage{multirow}
\usepackage{caption}
\usepackage[utf8]{inputenc}

% --- CODE LISTING STYLES ---
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{researchlog}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\scriptsize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    frame=single,
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

\begin{document}

% =================================================================================
% TITLE
% =================================================================================

\title{Democratizing Autonomous Deep Research: A Neuro-Symbolic Framework for 7B-Parameter Models via Grammar-Constrained Decoding}

\author{\IEEEauthorblockN{1\textsuperscript{st} Given Name Surname}
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{Your University Name}\\
City, Country \\
email@address.com}
}

\maketitle

% =================================================================================
% ABSTRACT
% =================================================================================
\begin{abstract}
The paradigm of Agentic AI—systems capable of autonomous planning, tool usage, and multi-step execution—has largely been realized only by massive, proprietary "frontier" models (e.g., GPT-4, Claude 3). While these models demonstrate impressive capabilities, their computational demands and privacy implications render them unsuitable for edge deployment. Conversely, open-source Small Language Models (SLMs) in the 7B parameter class suffer from significant reliability issues in agentic scenarios. They frequently fail to adhere to strict syntactic requirements for tool calling, hallucinate non-existent functions, or lose semantic coherence over long-horizon planning tasks, leading to catastrophic loop failures. This paper presents a novel, model-agnostic framework designed to bridge this "reliability gap" without the need for computationally expensive fine-tuning. We introduce a Neuro-Symbolic architecture that wraps frozen SLMs in a deterministic control layer. Our approach integrates (1) Inference-time Grammar-Constrained Decoding (GBNF) to strictly enforce JSON schema adherence at the logit level, guaranteeing syntactic correctness, and (2) A recursive "Critic-Planner" feedback loop that mitigates context window pollution through semantic verification. We evaluate this framework on a quantized Llama-3-8B model across distinct domains: technical analysis, current event verification, and multi-hop planning. Results show our framework achieves a 100\% syntactic success rate and successfully completes complex tasks where strong industry baselines (LangChain ReAct) fail catastrophically. Furthermore, we identify a novel "Reasoning Faithfulness Gap" in SLMs, where constrained actions remain correct even when internal Chain-of-Thought reasoning becomes flawed. This work demonstrates that robust agentic behaviors can be unlocked on consumer-grade hardware through neuro-symbolic architecture rather than parameter scaling.
\end{abstract}

\begin{IEEEkeywords}
Large Language Models, Agentic AI, Grammar-Constrained Decoding, Neuro-Symbolic AI, Edge Computing, Llama 3, Robustness.
\end{IEEEkeywords}

% =================================================================================
% I. INTRODUCTION
% =================================================================================
\section{Introduction}
The evolution of Large Language Models (LLMs) has rapidly shifted focus from static text generation to dynamic "Agentic Workflows." In these workflows, the LLM functions as a reasoning engine that perceives an environment, formulates plans, executes actions via external tools (e.g., search APIs, databases), and iterates based on observations [1]. This capability promises to automate complex knowledge work, from deep technical research to administrative scheduling.

However, a stark dichotomy has emerged in the generative AI ecosystem. On one side are "Frontier Models" (e.g., GPT-4o, Gemini 1.5 Ultra), possessing vast parameter counts (>1T estimated) and trained on massive datasets heavily supervised for tool usage. These models are highly reliable but suffer from high latency, immense operational costs, and requirements that data be sent to centralized servers, posing unacceptable privacy risks for many enterprise and personal applications.

On the other side are "Edge Models" or Small Language Models (SLMs), typically in the 7B to 13B parameter range (e.g., Llama-3-8B, Mistral-7B). These models are capable of running locally on consumer-grade hardware, offering zero-latency, private inference. Yet, recent evaluations highlight a critical bottleneck: "stochastic fragility" in autonomous loops [2], [3].

When an SLM is tasked with interacting with a rigid software interface—for example, outputting a specific JSON structure to call a search tool—it frequently fails. Common failure modes include missing syntactic elements (brackets, quotes), hallucinating incorrect parameter names, or "format drift," where the model abandons the structured output entirely and reverts to conversational text. In an autonomous loop, a single such syntax error crashes the entire operation. Furthermore, due to limited context windows and attention mechanisms, SLMs easily become distracted by irrelevant search results, leading to reasoning spirals and infinite loops.

Current industry solutions, such as the ReAct paradigm implemented in popular libraries like LangChain, rely on "prompt engineering" to encourage the model to follow formats and correct its own errors. As we demonstrate, this probabilistic approach is insufficient for 7B models in multi-turn scenarios.

In this paper, we argue that structural reliability in agents should not be a learned probabilistic behavior of the neural network, but a deterministic guarantee of the inference engine. We propose a Neuro-Symbolic framework that acts as a "cognitive exoskeleton" for frozen SLMs. By coupling neural reasoning with symbolic constraints, we address both syntactic fragility and semantic drift.

Our key contributions are:
\begin{enumerate}
    \item A \textbf{Neuro-Symbolic Architecture} that integrates inference-time Grammar-Constrained Decoding (GBNF) to mathematically guarantee valid JSON output for tool calling, regardless of the model's training data.
    \item A \textbf{Recursive Critic-Planner Loop}, a methodological approach designed for small-context models that implements a "System 2" verification step to filter noisy retrieval results before they pollute the working memory.
    \item Empirical evidence identifying a \textbf{"Reasoning Faithfulness Gap"} in 7B models, where the model's explicit Chain-of-Thought reasoning may contradict its actions, and demonstration that constraining the \textit{action} space is more robust than relying on correct reasoning traces.
\end{enumerate}

% =================================================================================
% II. RELATED WORK
% =================================================================================
\section{Related Work}
This research sits at the intersection of three dynamic fields: LLM Agent Architectures, Constrained Generation, and Neuro-Symbolic AI.

\subsection{Evolution of LLM Agents}
The transition from LLMs as chatbots to agents began with advanced prompting techniques. Chain-of-Thought (CoT) prompting [4] demonstrated that eliciting intermediate reasoning steps improves performance on complex tasks. This was extended by the ReAct (Reasoning and Acting) paradigm [5], which established the standard loop of generating a thought, executing an action via a tool, and observing the output. ReAct forms the backbone of most modern agent frameworks like LangChain and AutoGPT.

While effective for large models, ReAct is brittle for SLMs. The burden of maintaining the strict `Thought: ... Action: ... Observation: ...` format over many turns often exceeds the model's coherence abilities. To address this, models like Toolformer [6] and Gorilla [7] introduced fine-tuning specifically for API calls. While these methods achieve high accuracy, they are computationally expensive, require massive curated datasets, and result in rigid models tied to specific tool schemas. If an API changes, the model must be retrained. Our approach differs by being entirely training-free and model-agnostic, relying on inference-time control rather than weight updates.

\subsection{Constrained Decoding and Structured Generation}
Ensuring LLMs generate outputs that adhere to formal languages (like JSON, SQL, or Python) is a critical challenge. Early approaches relied on simple regex matching or post-hoc correction, which are inefficient and unreliable.

Modern constrained decoding intervenes during the token sampling process. Frameworks like Guidance, Outlines, and recent implementations in `llama.cpp` utilize Context-Free Grammars (CFGs) or Grammar-Backus-Naur Forms (GBNF) to construct a finite-state automaton representing valid outputs [8], [9]. During the LLM's forward pass, the decoding engine calculates which tokens are valid transitions in the automaton given the previously generated sequence. It then applies a negative infinity mask to the logits of all invalid tokens before sampling. This mathematically guarantees that the output will conform to the defined grammar.

Our work applies this powerful techniqueSpecifically to the domain of autonomous, multi-turn agents. We demonstrate that GBNF is not just a formatting convenience, but a fundamental requirement for making SLM agents viable.

\subsection{Neuro-Symbolic AI}
Neuro-symbolic AI seeks to bridge the gap between neural networks (excellent at intuition, pattern recognition, and handling noisy data) and symbolic AI (excellent at logic, rules, and guaranteed correctness) [10].

Historically, this involved complex architectures trying to convert neural representations into symbolic logic and back. In the era of LLMs, a new design pattern is emerging: using the LLM as a versatile probabilistic reasoning engine, while wrapping it in deterministic symbolic control structures to ensure safety, reliability, and adherence to business logic. Our framework is a practical implementation of this pattern, using symbolic grammars to constrain the "action space" of the neural planner.

% =================================================================================
% III. SYSTEM ARCHITECTURE
% =================================================================================
\section{System Architecture}
Our framework is built upon the principle of "Neuro-Symbolic Decoupling." We decouple the \textit{semantic reasoning capability} (provided by the neural LLM) from the \textit{syntactic execution capability} (enforced by symbolic engines).

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{User Query.png} 
    \caption{The Proposed Neuro-Symbolic Architecture. The Neural components (Planner, Critic) provide semantic reasoning on noisy text, while the Symbolic components (Grammar Engine, JSON Parser) ensure complete syntactic validity for tool interaction.}
    \label{fig:architecture}
\end{figure}

As illustrated in Fig. \ref{fig:architecture}, the architecture consists of three primary interacting modules organized in a recursive loop.

\subsection{The Neural Planner}
The Planner is the cognitive core, responsible for decomposing a high-level user objective into a sequence of atomic retrieval actions.
\begin{itemize}
    \item \textbf{Input:} The original user query, the current accumulated context (summarized facts), and a history of past queries.
    \item \textbf{Function:} It acts as a standard causal LM, predicting the next most likely tokens based on the prompt.
    \item \textbf{Prompt Engineering:} The system prompt explicitly instructs the model to consider previous failures (from the history) and to formulate a rationale before deciding on a query.
\end{itemize}

\subsection{The Symbolic Grammar Engine}
This component acts as a deterministic gatekeeper between the Planner's neural output and the external world. It resolves the "stochastic fragility" problem.

We define a GBNF grammar that strictly describes the required JSON schema for a tool action. A simplified representation of this grammar is:

\begin{verbatim}
root ::= "{" space "\"rationale\"" space ":" 
string "," space "\"query\"" space ":" string "}"
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | 
"u" [0-9a-fA-F]{4}) )* "\"" space
space ::= [ \t\n\r]*
\end{verbatim}

During inference, at time step $t$, the LLM computes a probability distribution $P(w_t | w_{<t})$ over its entire vocabulary $V$. The Grammar Engine maintains a state machine tracking the partial JSON generated so far ($w_{<t}$). It identifies the set of valid next tokens $V_{valid} \subset V$ permitted by the grammar schema. It then applies a logit mask $M$:

\begin{equation}
    M(w_t) = 
    \begin{cases} 
      0 & \text{if } w_t \in V_{valid} \\
      -\infty & \text{if } w_t \notin V_{valid}
    \end{cases}
\end{equation}

The final probabilities are computed using softmax over the masked logits: $P'(w_t) = \text{softmax}(\text{logits}(w_t) + M(w_t))$. This ensures $P'(w_{invalid}) = 0$. The model is mathematically incapable of generating a syntax error.

\subsection{The Neural Critic}
SLMs have small context windows (typically 8K tokens for Llama-3) and easily get "distracted" by irrelevant text in search results. Blindly appending all search results to the context leads to "context pollution," degrading reasoning ability over time.

The Critic is a secondary neural module designed to mitigate this. It is the same 7B model but invoked with a distinct, discriminative prompt. It takes the original user query and a raw search result snippet as input and performs a binary classification task: \textit{"Does this snippet contain relevant facts to answer the query? YES/NO."}

Only data verified as "YES" by the Critic is processed further. This implements a "System 2" check on the noisy retrieval process, ensuring the limited context window is reserved for high-value information.

% =================================================================================
% IV. METHODOLOGY: THE RECURSIVE LOOP
% =================================================================================
\section{Methodology: The Recursive Research Loop}
The system operates as a state machine that recursively executes a research-verify-update cycle. The precise algorithm is detailed below.

\begin{algorithm}
\caption{Grammar-Constrained Recursive Research}
\label{alg:research_loop}
\begin{algorithmic}[1]
\Require User Query $Q$, Max Iterations $N$
\State $Context \mathcal{C} \leftarrow \text{"No knowledge yet."}$
\State $History \mathcal{H} \leftarrow \emptyset$
\State $G \leftarrow \text{LoadGBNF("action\_schema.gbnf")}$

\For{$i \leftarrow 1$ \textbf{ to } $N$}
    \State \textbf{// Phase 1: Constrained Planning}
    \State $prompt \leftarrow \text{ConstructPrompt}(Q, \mathcal{C}, \mathcal{H})$
    \State \Comment{Force valid JSON generation}
    \State $raw\_json \leftarrow \text{LLM\_Generate}(prompt, \textbf{grammar}=G)$
    \State $Action \leftarrow \text{JSON\_Parse}(raw\_json)$
    
    \State \textbf{// Symbolic Logic Check: Loop Prevention}
    \If{$Action.query \in \mathcal{H}$}
        \State \textbf{continue} \Comment{Skip redundant action}
    \EndIf
    \State $\mathcal{H}.\text{append}(Action.query)$
    
    \State \textbf{// Phase 2: Tool Execution}
    \State $Observation \leftarrow \text{DuckDuckGoSearch}(Action.query)$
    
    \State \textbf{// Phase 3: Neural Criticism}
    \State $Verification \leftarrow \text{CriticLLM}(Q, Observation)$
    
    \If{$Verification == \text{"YES"}$}
        \State $Snippet \leftarrow \text{PruneAndSummarize}(Observation)$
        \State $\mathcal{C} \leftarrow \mathcal{C} + \text{"[FACT]: " } + Snippet$
    \Else
        \State \textbf{continue} \Comment{Discard noisy data}
    \EndIf
\EndFor

\State \textbf{// Phase 4: Final Synthesis}
\State $FinalAnswer \leftarrow \text{LLM}(\text{SynthesizePrompt}(Q, \mathcal{C}))$
\State \Return $FinalAnswer$
\end{algorithmic}
\end{algorithm}

This workflow addresses the core weaknesses of SLMs: the grammar prevents syntactic failure during Planning (lines 7-8), the history check prevents algorithmic looping (lines 11-13), and the critic prevents context overflow and distraction (lines 19-24).

% =================================================================================
% V. EXPERIMENTAL SETUP
% =================================================================================
\section{Experimental Setup}
We evaluated the framework's ability to enable autonomous research on resource-constrained hardware, simulating real-world edge deployment scenarios.

\subsection{Environment \& Hardware}
All experiments were conducted on a standard consumer workstation without datacenter-grade GPUs.
\begin{itemize}
    \item \textbf{CPU:} AMD Ryzen 9 5900X (12-core).
    \item \textbf{RAM:} 32GB DDR4 (System total).
    \item \textbf{Model:} \texttt{Meta-Llama-3-8B-Instruct}.
    \item \textbf{Quantization:} The model was quantized to 4-bit (Q4\_K\_M format) using \texttt{llama.cpp}. This reduces the model footprint to approximately 4.9GB, allowing it to run comfortably alongside the OS and application logic in standard system RAM.
    \item \textbf{Inference Engine:} \texttt{llama-cpp-python} (v0.2.70), utilizing its integrated GBNF grammar sampler for constrained generation.
    \item \textbf{External Tool:} DuckDuckGo Search API (retrieving text snippets only).
\end{itemize}

\subsection{Evaluation Tasks (Multi-Domain)}
To assess generalization capabilities beyond a single domain, we defined three distinct research tasks varying in cognitive complexity and required knowledge freshness:

\begin{enumerate}
    \item \textbf{T1: Technical Comparative Analysis.} \textit{"Compare the energy efficiency of Transformer models vs State Space Models (Mamba) for edge devices."}
    \item \textbf{T2: Current Event Verification.} \textit{"Investigate the current status of the Voyager 1 space probe as of late 2024/2025. Summarize its distance from Earth and instrument status."}
    \item \textbf{T3: Multi-hop Planning \& Reasoning.} \textit{"Identify the host city for the 2028 Summer Olympics. Then, find the average temperature in that city during the month of July. Finally, suggest three outdoor Olympic events suitable for that specific weather."}
\end{enumerate}

\textbf{Task Justification:} T1 requires retrieving niche technical specifications not present in general pre-training data. T2 tests the ability to reject outdated information and find very recent facts. T3 is the most complex, requiring sequential dependency management; the agent cannot formulate the second query (weather) without successfully executing and parsing the first (host city).

\subsection{Baselines and Metrics}
We compare our \textbf{Proposed Neuro-Symbolic Framework} against a **Strong Baseline**: a standard \textbf{LangChain Zero-Shot ReAct Agent}. LangChain is currently the industry-standard framework for building LLM agents. The baseline was initialized with the exact same quantized Llama-3-8B model and search tool access, using its default ReAct prompt templates.

We quantitatively measure:
\begin{itemize}
    \item \textbf{Success Rate:} Percentage of trials ending in a correct final answer without crashing.
    \item \textbf{Failure Mode:} Categorized into Syntax Error (invalid JSON/tool call), Loop (repeating actions until timeout), or Hallucination (generating a correct-looking answer based on false premises).
    \item \textbf{Resource Usage:} Total wall-clock time and peak resident memory usage.
\end{itemize}

% =================================================================================
% VI. RESULTS AND DISCUSSION
% =================================================================================
\section{Results and Discussion}

\subsection{Comparative Analysis against Baseline}
We conducted N=20 trials for each task against the baseline. Table I presents a representative performance comparison.

\begin{table}[htbp]
\caption{Performance Comparison: Proposed Framework vs. LangChain Baseline (Llama-3-8B, CPU)}
\label{tab:comparison}
\begin{center}
\begin{tabular}{|c|l|c|c|c|}
\hline
\textbf{Task} & \textbf{System} & \textbf{Status} & \textbf{Time (s)} & \textbf{Mem (MB)} \\
\hline
\multirow{2}{*}{T1 (Tech)} & LangChain ReAct & Failed (Loop) & 45s & 9089 \\
\cline{2-5}
 & \textbf{Proposed Neuro-Sym} & \textbf{Success} & 197s & 9267 \\
\hline
\hline
\multirow{2}{*}{T2 (News)} & LangChain ReAct & Success & \textbf{22s} & 9094 \\
\cline{2-5}
 & \textbf{Proposed Neuro-Sym} & Success & 181s & 9272 \\
\hline
\hline
\multirow{2}{*}{T3 (Plan)} & LangChain ReAct & \textbf{CRITICAL FAIL} & >1500s & 9103 \\
\cline{2-5}
 & \textbf{Proposed Neuro-Sym} & \textbf{Success} & 131s & 9275 \\
\hline
\end{tabular}
\end{center}
\end{table}

The results demonstrate a decisive reliability advantage for our framework in complex scenarios.

\textbf{Baseline Failure Analysis:} The LangChain agent demonstrated characteristic 7B model fragility. In T1, it frequently got stuck in a loop, proposing the same search query repeatedly until hitting the iteration limit. The most significant failure occurred in T3 (Multi-hop). After successfully retrieving the host city (Los Angeles), the model failed to maintain the strict `Thought/Action/Observation` format required by the ReAct prompt. It began generating conversational text instead of a structured action block. The LangChain parser, unable to find the expected structure, threw errors or, in worse cases, the model entered an infinite generation loop of nonsensical tokens, requiring manual termination after 25 minutes.

\textbf{Proposed Framework Performance:} Our system successfully completed all tasks across all domains. The GBNF constraint ensured 100\% adherence to the action schema, completely eliminating the syntax error failure mode. In T3, it correctly sequenced the three dependent searches (City $\rightarrow$ Weather $\rightarrow$ Events) without format drift.

\textbf{Efficiency Trade-off:} In simple, single-step tasks like T2, the baseline was significantly faster (22s vs 181s). This is due to the overhead of our explicit recursive loop, which forces a separate Planner call and Critic call for every step. The baseline often "guessed" the answer in one step. However, this overhead is the necessary cost of reliability for complex tasks where the baseline fails completely. Memory usage was consistent across both systems, proving our framework adds negligible memory overhead relative to the model weights themselves.

\subsection{Ablation Studies: Deconstructing Reliability}
To isolate the contributions of the architecture's components, we conducted ablation studies on the complex tasks (T1, T3).

\begin{table}[htbp]
\caption{Ablation Study Results (Tasks T1 \& T3)}
\label{tab:ablation}
\begin{center}
\begin{tabular}{|l|c|c|l|}
\hline
\textbf{Config} & \textbf{Grammar?} & \textbf{Critic?} & \textbf{Primary Outcome / Observation} \\
\hline
Full & Yes & Yes & \textbf{Success:} High quality, specific details derived from retrieval. \\
\hline
Ablation A & \textbf{No} & Yes & \textbf{100\% Failure:} Immediate crash due to JSON syntax errors. \\
\hline
Ablation B & Yes & \textbf{No} & \textbf{Degraded Success:} Vague answers; model overwhelmed by noise. \\
\hline
\end{tabular}
\end{center}
\end{table}

\textbf{Impact of Grammar (Ablation A):} Removing the GBNF constraint was catastrophic. Despite the system prompt explicitly requesting JSON, the 7B model frequently failed to close brackets, forgot quotes around keys, or prepended conversational filler (e.g., "Sure, here is the JSON for the search..."). This resulted in a 100\% failure rate due to parser exceptions. This confirms that inference-time constraints are not optional features but mandatory requirements for reliable tool use in SLMs.

\textbf{Impact of Critic (Ablation B):} Without the Critic, the agent was syntactically functional but semantically impaired. It accepted every search result, rapidly filling its context window with marginally relevant text. For T1 (Mamba vs Transformer), the agent became "distracted" by generic marketing text in the search results and the final answer stated "no specific technical comparison found," despite highly relevant technical papers being present in the raw search feed. The Critic is essential for signal-to-noise filtering in small context models.

\subsection{The "Reasoning Faithfulness Gap"}
An analysis of execution logs revealed a significant cognitive science phenomenon we term the "Reasoning Faithfulness Gap."

The Planner is forced to generate a `rationale` (Chain-of-Thought) before the `query`. In several instances, the logs showed the Planner generating a rationale such as, \textit{"The previous searches failed to yield results, so I need to try a new strategy,"} even immediately after the Critic step had successfully verified relevant data and marked it with a "YES."

This indicates a profound cognitive dissonance in 7B models: their internal Chain-of-Thought reasoning does not reliably reflect their actual context history or internal state. They often hallucinate justifications to fit training patterns (like retrying on failure). However, crucially, because our framework decouples the \textit{reasoning text} from the \textit{constrained action schema}, the agent continued to perform the correct next action (e.g., moving to the next part of the problem) despite its flawed internal monologue. This suggests that for SLM agents, constraining the output action space is a far more robust control mechanism than relying on the correctness of their reasoning traces.

% =================================================================================
% VII. CONCLUSION AND FUTURE WORK
% =================================================================================
\section{Conclusion}
This paper demonstrates that the prevailing view of 7B-parameter models as too "unintelligent" for autonomous research tasks is flawed. Their primary weakness in agentic scenarios is not a lack of world knowledge, but a lack of structural discipline and attention span during generation.

By wrapping a quantized Llama-3-8B model in a Neuro-Symbolic framework utilizing Grammar-Constrained Decoding and a recursive Critic loop, we achieved 100\% syntactic reliability across complex, multi-hop planning tasks where industry-standard baselines failed catastrophically. Ablation studies confirmed that both symbolic constraints and neural verification are necessary conditions for this success.

This work unlocks a pathway for deploying reliable, private AI researchers on existing consumer-grade hardware, reducing dependency on centralized frontier models. Future work will focus on integrating local Vector Databases (RAG) to provide long-term memory, overcoming the context window limitations inherent to SLMs, and testing the framework on even smaller sub-4B parameter models.

% =================================================================================
% REFERENCES
% =================================================================================
\begin{thebibliography}{00}

\bibitem{survey2025} Z. Liu et al., "From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review," \textit{arXiv preprint arXiv:2504.19678}, 2025.

\bibitem{slmsecurity2025} "Security and Robustness Challenges of Small Language Models in Autonomous Agent Networks," \textit{ResearchGate}, Oct. 2025.

\bibitem{enkrypt2025} "Small Models, Big Problems: Why Your AI Agents Might Be Sitting Ducks," \textit{Enkrypt AI Blog}, Sep. 2025.

\bibitem{wei2022chain} J. Wei et al., "Chain-of-thought prompting elicits reasoning in large language models," \textit{NeurIPS}, vol. 35, pp. 24824–24837, 2022.

\bibitem{yao2022react} S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," in \textit{International Conference on Learning Representations (ICLR)}, 2023.

\bibitem{schick2023toolformer} T. Schick et al., "Toolformer: Language models can teach themselves to use tools," \textit{NeurIPS}, 2023.

\bibitem{gorilla2023} S. Patil et al., "Gorilla: Large Language Model Connected with Massive APIs," \textit{arXiv preprint arXiv:2305.15334}, 2023.

\bibitem{grammar2023} L. G. Foo et al., "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning," \textit{arXiv preprint arXiv:2305.13971}, 2023.

\bibitem{flexiblegrammar2025} "Flexible and Efficient Grammar-Constrained Decoding," \textit{OpenReview}, June 2025.

\bibitem{neurosymbolic2025} "Neuro-Symbolic AI in 2024: A Systematic Review," \textit{arXiv preprint arXiv:2501.05435}, 2025.

\bibitem{agentsurvey2025} "Agentic AI Frameworks: Architectures, Protocols, and Design Challenges," \textit{arXiv preprint arXiv:2508.10146}, 2025.

\bibitem{touvron2023llama} H. Touvron et al., "Llama 2: Open foundation and fine-tuned chat models," \textit{arXiv preprint arXiv:2307.09288}, 2023.

\bibitem{mamba2023} A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," \textit{arXiv preprint arXiv:2312.00752}, 2023.

\bibitem{llamacpp} G. Gerganov, "llama.cpp: Port of Facebook's LLaMA model in C/C++," \textit{GitHub repository}, 2023.

\end{thebibliography}

\end{document}
