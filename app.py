# app.py - Simplified Neuro-Symbolic UI
import streamlit as st
import json
import time
import os
import re
from llama_cpp import Llama
from langchain_community.tools import DuckDuckGoSearchRun
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
        background-color: #FF4B4B;
        color: white;
    }
    .source-link {
        padding: 5px;
        margin: 2px;
        background-color: #f0f2f6;
        border-radius: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND SETUP
# ==========================================
MODEL_PATH = "./Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
N_CTX = 8192

@st.cache_resource
def load_model():
    """Loads the model once and caches it."""
    print(f"[SYSTEM] Loading model from {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=-1, # CPU execution
        verbose=False
    )
    return llm

@st.cache_resource
def load_grammar():
    """Defines the GBNF grammar for the JSON action."""
    gbnf_string = r"""
        root ::= "{" space "\"rationale\"" space ":" space string "," space "\"query\"" space ":" space string "}"
        string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
        space ::= [ \t\n\r]*
    """
    return LlamaGrammar.from_string(gbnf_string)

# Initialize tool
search_tool = DuckDuckGoSearchRun()

def extract_links(search_results_text):
    """Helper to extract URLs from DuckDuckGo text results."""
    # This is a basic regex for URLs found in the text output
    url_pattern = r'(https?://\S+)'
    links = re.findall(url_pattern, search_results_text)
    # Clean up links (remove trailing punctuation often caught by regex)
    clean_links = [link.rstrip('.,;)"\']') for link in links]
    return list(set(clean_links)) # Return unique links

# ==========================================
# 3. CORE AGENT LOGIC (HIDDEN FROM UI)
# ==========================================
def generate_constrained_plan(llm, grammar, objective, context, history):
    """Neural Planner + Symbolic Gate (Behind the scenes)"""
    history_str = "\n".join([f"- {q}" for q in history]) if history else "None."
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an autonomous research agent. Objective: {objective}. Context: {context}. Past Queries: {history_str}.
Output a JSON with "rationale" and "query".<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    output = llm(prompt, max_tokens=200, grammar=grammar, temperature=0.1)
    text_output = output['choices'][0]['text']
    try:
        return json.loads(text_output)
    except json.JSONDecodeError:
        return None

def run_critic(llm, query, data):
    """Neural Critic (Behind the scenes)"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Query: {query}. Data snippet: {data[:1500]}. Does this contain relevant facts? Answer strictly YES or NO.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    output = llm(prompt, max_tokens=10)
    decision = output['choices'][0]['text'].strip().upper()
    return "YES" in decision

# ==========================================
# 4. MAIN UI LAYOUT
# ==========================================
st.title("🧠 Neuro-Symbolic Deep Research")
st.caption("Powered by local Llama-3-8B with Grammar Constraints.")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("Settings")
    max_iter = st.slider("Research Depth (Iterations)", min_value=2, max_value=7, value=4)

# --- Input Area ---
user_query = st.text_area("What would you like to research?", height=100, 
                         value="Identify the host city for the 2028 Summer Olympics. Then, find the average temperature in that city during the month of July. Finally, suggest three outdoor Olympic events suitable for that specific weather.")
run_button = st.button("Start Research")

# --- Main Execution Logic ---
if run_button:
    # Initialize State
    context_text = "No knowledge gathered yet."
    history_list = []
    sources_list = []
    
    # Load backend quietly
    llm_engine = load_model()
    grammar_engine = load_grammar()
        
    # THE RECURSIVE LOOP (Hidden behind a single spinner)
    with st.spinner("Performing deep research... This may take a few minutes."):
        start_time = time.time()
        for i in range(max_iter):
            # 1. Planner & Grammar
            action_json = generate_constrained_plan(llm_engine, grammar_engine, user_query, context_text, history_list)
            if not action_json: break
            query = action_json.get('query')
            if query in history_list: continue
            history_list.append(query)

            # 2. Tool Execution & Link Extraction
            try:
                search_result = search_tool.run(query)
                # Extract links for evidence display
                links = extract_links(search_result)
                sources_list.extend(links)
            except Exception:
                search_result = ""

            # 3. Critic
            if search_result and run_critic(llm_engine, query, search_result):
                snippet = search_result[:1000] + "..."
                context_text += f"\n[Fact]: {snippet}"

        # Final Synthesis
        final_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Synthesize a comprehensive answer based ONLY on these notes: {context_text}. User Question: {user_query}. Use Markdown formatting.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        final_output = llm_engine(final_prompt, max_tokens=1500)
        result_text = final_output['choices'][0]['text']
        total_time = time.time() - start_time

    # ===========================
    # RESULTS DISPLAY
    # ===========================
    st.divider()
    st.subheader("Research Report")
    st.caption(f"Completed in {total_time:.2f} seconds.")
    
    # Display Sources first to prove authenticity
    if sources_list:
        with st.expander("📚 Research Sources Used (Evidence)", expanded=False):
            unique_sources = list(set(sources_list))
            for link in unique_sources:
                st.markdown(f'<a href="{link}" target="_blank" class="source-link">🔗 {link}</a>', unsafe_allow_html=True)
    
    # Display Final Answer
    st.markdown(result_text)
    st.balloons()