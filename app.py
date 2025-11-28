# app.py - Simplified Neuro-Symbolic UI with Clear Sources
import streamlit as st
import json
import time
import os
import re
# Import the model engine
from llama_cpp import Llama
# Import the search tool
from langchain_community.tools import DuckDuckGoSearchRun
# Import grammar constraint
from llama_cpp.llama_grammar import LlamaGrammar

# ==========================================
# 1. UI CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Neuro-Symbolic Deep Research",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #0E1117;
        color: white;
        border: 1px solid #30333F;
    }
    .stButton>button:hover {
        background-color: #262730;
        border-color: #4F8BF9;
    }
    /* Style for source links to make them look professional */
    a.source-link {
        color: #4F8BF9 !important;
        text-decoration: none;
        font-family: monospace;
        display: block;
        padding: 4px 0;
    }
    a.source-link:hover {
        text-decoration: underline;
    }
    .report-container {
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND SETUP
# ==========================================
# Path to your local quantized model file
MODEL_PATH = "./Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
N_CTX = 8192 # Context window size

@st.cache_resource
def load_model():
    """Loads the model once and caches it to avoid reloading."""
    print(f"[SYSTEM] Loading model from {MODEL_PATH}...")
    # Initialize Llama model for CPU execution
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=-1, # -1 means use CPU for all layers
        verbose=False
    )
    return llm

@st.cache_resource
def load_grammar():
    """Defines the GBNF grammar to force strictly valid JSON output."""
    gbnf_string = r"""
        root ::= "{" space "\"rationale\"" space ":" space string "," space "\"query\"" space ":" space string "}"
        string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
        space ::= [ \t\n\r]*
    """
    return LlamaGrammar.from_string(gbnf_string)

# Initialize the search tool
search_tool = DuckDuckGoSearchRun()

def extract_links(search_results_text):
    """Helper function to extract URLs from search result text."""
    # Basic regex to find http/https URLs
    url_pattern = r'(https?://\S+)'
    links = re.findall(url_pattern, search_results_text)
    # Clean trailing punctuation often caught by regex
    clean_links = [link.rstrip('.,;)"\']') for link in links]
    # Return unique links only
    return list(set(clean_links))

# ==========================================
# 3. CORE AGENT LOGIC (HIDDEN FROM USER)
# ==========================================
def generate_constrained_plan(llm, grammar, objective, context, history):
    """Phase 1: Neural Planner + Symbolic Grammar Gate"""
    history_str = "\n".join([f"- {q}" for q in history]) if history else "None."
    # System prompt to guide the agent
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an autonomous research agent. Objective: {objective}. Context: {context}. Past Queries: {history_str}.
INSTRUCTIONS: Output a JSON object with keys "rationale" and "query". Output ONLY JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # Generate with grammar constraint
    output = llm(prompt, max_tokens=200, grammar=grammar, temperature=0.1)
    text_output = output['choices'][0]['text']
    try:
        # Parse the guaranteed valid JSON
        return json.loads(text_output)
    except json.JSONDecodeError:
        # This should theoretically never happen with an active grammar
        return None

def run_critic(llm, query, data):
    """Phase 3: Neural Critic for Data Verification"""
    # Discriminative prompt for binary classification
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Query: {query}. Data snippet: {data[:1500]}. Does this snippet contain relevant facts to answer the query? Answer strictly YES or NO.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    output = llm(prompt, max_tokens=10)
    decision = output['choices'][0]['text'].strip().upper()
    return "YES" in decision

# ==========================================
# 4. MAIN UI LAYOUT
# ==========================================
st.title("🧠 Neuro-Symbolic Deep Research")
st.caption("Autonomous agent powered by local Llama-3-8B with Grammar Constraints.")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Configuration")
    # Slider to control how many research steps the agent takes
    max_iter = st.slider("Research Depth (Iterations)", min_value=2, max_value=7, value=4, help="More iterations allow for deeper, multi-hop research but take longer.")
    st.divider()
    st.markdown("**Framework Status:** Ready")
    st.caption(f"Model: {os.path.basename(MODEL_PATH)}")

# --- Main Input Area ---
user_query = st.text_area("Enter your research objective:", height=100, 
                         value="Identify the host city for the 2028 Summer Olympics. Then, find the average temperature in that city during the month of July. Finally, suggest three outdoor Olympic events suitable for that specific weather.",
                         help="Describe the complex, multi-step information you want the agent to find.")

# The main action button
run_button = st.button("Initiate Deep Research Cycle")

# --- Main Execution Logic ---
if run_button:
    # Initialize internal state variables
    context_text = "No knowledge gathered yet."
    history_list = []
    sources_list = [] # To store URLs
    
    # Load backend components (cached so it's fast after first run)
    llm_engine = load_model()
    grammar_engine = load_grammar()
        
    # THE RECURSIVE LOOP (Hidden behind a single loading spinner)
    with st.spinner("Performing autonomous deep research... Please wait, this requires significant on-device computation."):
        start_time = time.time()
        # Main loop controlled by max_iter setting
        for i in range(max_iter):
            # --- Step 1: Plan & Constrain ---
            action_json = generate_constrained_plan(llm_engine, grammar_engine, user_query, context_text, history_list)
            if not action_json: break # Stop if planning fails entirely
            query = action_json.get('query')
            # Prevent infinite loops by checking history
            if query in history_list: continue
            history_list.append(query)

            # --- Step 2: Execute Tool & Capture Sources ---
            try:
                search_result = search_tool.run(query)
                # Extract URLs from the raw text result for evidence
                links = extract_links(search_result)
                sources_list.extend(links)
            except Exception:
                search_result = "" # Handle tool failures gracefully

            # --- Step 3: Verify Data (Critic) ---
            # Only add data to context if the Critic marks it as relevant ("YES")
            if search_result and run_critic(llm_engine, query, search_result):
                snippet = search_result[:1000] + "..."
                context_text += f"\n[Fact]: {snippet}"

        # --- Final Step: Synthesize Answer ---
        # Prompt the model to act as a final analyst using only gathered facts
        final_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a senior research analyst. Synthesize a comprehensive, well-structured answer based ONLY on the following verified notes. Do not use outside knowledge. User Question: {user_query}. Notes: {context_text}. Use Markdown formatting for the final report.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        final_output = llm_engine(final_prompt, max_tokens=1500)
        result_text = final_output['choices'][0]['text']
        total_time = time.time() - start_time

    # ===========================
    # RESULTS DISPLAY SECTION
    # ===========================
    st.divider()
    st.subheader("Final Research Report")
    st.caption(f"Generative cycle completed in {total_time:.2f} seconds based on {len(history_list)} distinct search actions.")
    
    # 1. Display the Final Answer
    with st.container():
        st.markdown(f'<div class="report-container">{result_text}</div>', unsafe_allow_html=True)

    # 2. Display Sources in a separate, clear section below
    if sources_list:
        st.subheader("📚 Verified Research Sources")
        st.markdown("The agent extracted data from the following external URLs during its research cycle:")
        # De-duplicate links so the list is clean
        unique_sources = sorted(list(set(sources_list)))
        source_container = st.container()
        with source_container:
            for i, link in enumerate(unique_sources):
                # Create a clean, numbered list of clickable links
                st.markdown(f'{i+1}. <a href="{link}" target="_blank" class="source-link">{link}</a>', unsafe_allow_html=True)
    else:
        st.warning("No external sources were successfully verified by the critic during this run.")