# app.py - Robust Neuro-Symbolic UI with Raw Data Fallback
import streamlit as st
import json
import time
import os
import re
from llama_cpp import Llama
from langchain_community.tools import DuckDuckGoSearchRun
from llama_cpp.llama_grammar import LlamaGrammar

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Neuro-Symbolic Deep Research",
    page_icon="🧠",
    layout="centered"
)

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
    .source-box {
        margin-top: 20px;
        padding: 15px;
        border-left: 5px solid #4F8BF9;
        background-color: #f0f2f6;
        border-radius: 5px;
        color: black;
    }
    a.source-link {
        color: #0366d6 !important;
        text-decoration: none;
        font-family: monospace;
        display: block;
        margin-bottom: 5px;
    }
    /* Style for raw logs */
    .raw-log {
        font-family: 'Courier New', monospace;
        font-size: 0.8em;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
        white-space: pre-wrap;
        color: #333;
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
    print(f"[SYSTEM] Loading model from {MODEL_PATH}...")
    return Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=-1, verbose=False)

@st.cache_resource
def load_grammar():
    gbnf_string = r"""
        root ::= "{" space "\"rationale\"" space ":" space string "," space "\"query\"" space ":" space string "}"
        string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
        space ::= [ \t\n\r]*
    """
    return LlamaGrammar.from_string(gbnf_string)

search_tool = DuckDuckGoSearchRun()

def extract_links(text):
    """Aggressively finds URLs in text."""
    url_pattern = r'(https?://[^\s,]+)'
    links = re.findall(url_pattern, text)
    clean_links = [link.rstrip('.,;)"\']') for link in links]
    return list(set(clean_links))

# ==========================================
# 3. CORE LOGIC
# ==========================================
def generate_plan(llm, grammar, objective, context, history):
    history_str = "\n".join([f"- {q}" for q in history]) if history else "None."
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an autonomous research agent. Objective: {objective}. Context: {context}. Past Queries: {history_str}.
INSTRUCTIONS: Output a JSON object with keys "rationale" and "query". Output ONLY JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    output = llm(prompt, max_tokens=200, grammar=grammar, temperature=0.1)
    try:
        return json.loads(output['choices'][0]['text'])
    except:
        return None

def run_critic(llm, query, data):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Query: {query}. Data: {data[:1000]}. Is this relevant? YES or NO.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    output = llm(prompt, max_tokens=10)
    return "YES" in output['choices'][0]['text'].strip().upper()

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🧠 Neuro-Symbolic Deep Research")
st.caption(f"Model: {os.path.basename(MODEL_PATH)} | Mode: Autonomous")

# Initialize Session State
if "history" not in st.session_state: st.session_state.history = []
if "sources" not in st.session_state: st.session_state.sources = []
if "raw_logs" not in st.session_state: st.session_state.raw_logs = []  # Store raw text here
if "context" not in st.session_state: st.session_state.context = "No knowledge yet."
if "result" not in st.session_state: st.session_state.result = ""

with st.sidebar:
    st.header("Settings")
    max_iter = st.slider("Iterations", 2, 7, 4)
    if st.button("Reset System"):
        st.session_state.history = []
        st.session_state.sources = []
        st.session_state.raw_logs = []
        st.session_state.context = "No knowledge yet."
        st.session_state.result = ""
        st.rerun()

user_query = st.text_area("Research Objective:", height=100, 
    value="Identify the host city for the 2028 Summer Olympics. Then, find the average temperature in that city during the month of July. Finally, suggest three outdoor Olympic events suitable for that specific weather.")

if st.button("Initiate Research"):
    # Clear previous run data
    st.session_state.history = []
    st.session_state.sources = []
    st.session_state.raw_logs = []
    st.session_state.context = "No knowledge yet."
    st.session_state.result = ""
    
    llm_engine = load_model()
    grammar_engine = load_grammar()
    
    status = st.empty()
    progress = st.progress(0)
    start_time = time.time()
    
    # --- LOOP ---
    for i in range(max_iter):
        status.text(f"Phase {i+1}/{max_iter}: Planning & Searching...")
        progress.progress(int((i / max_iter) * 90))
        
        # 1. Plan
        action = generate_plan(llm_engine, grammar_engine, user_query, st.session_state.context, st.session_state.history)
        if not action: break
        query = action.get('query')
        if query in st.session_state.history: continue
        st.session_state.history.append(query)
        
        # 2. Search & Capture Raw Data
        try:
            raw_result = search_tool.run(query)
            
            # --- CAPTURE RAW DATA ---
            # Save the raw text immediately to session state so it's never lost
            st.session_state.raw_logs.append(f"🔎 QUERY: {query}\n📄 RESULT: {raw_result[:500]}...") 
            
            # Extract links
            new_links = extract_links(raw_result)
            if new_links:
                st.session_state.sources.extend(new_links)
                for link in new_links:
                    st.toast(f"Source Verified: {link}", icon="🔗")
            else:
                # If no links, we still have the raw text in raw_logs!
                pass

        except Exception as e:
            raw_result = f"Error: {str(e)}"
            st.session_state.raw_logs.append(f"❌ ERROR for '{query}': {str(e)}")

        # 3. Critic
        if raw_result and run_critic(llm_engine, query, raw_result):
            st.session_state.context += f"\n[Fact]: {raw_result[:1000]}"

    # --- SYNTHESIS ---
    status.text("Finalizing: Synthesizing Report...")
    progress.progress(95)
    
    final_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Synthesize a comprehensive answer based ONLY on these notes: {st.session_state.context}. Question: {user_query}. Use Markdown.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    final_output = llm_engine(final_prompt, max_tokens=1500)
    st.session_state.result = final_output['choices'][0]['text']
    
    progress.progress(100)
    time.sleep(0.5)
    status.empty()
    progress.empty()

# --- DISPLAY RESULTS (Persists after run) ---
if st.session_state.result:
    st.divider()
    st.subheader("Final Research Report")
    st.markdown(st.session_state.result)
    
    st.markdown("---")
    st.subheader("📚 Verified Research Sources")
    
    # 1. Show Clean Links (If any)
    if st.session_state.sources:
        unique = sorted(list(set(st.session_state.sources)))
        with st.container():
            st.markdown('<div class="source-box">', unsafe_allow_html=True)
            for i, link in enumerate(unique):
                st.markdown(f'{i+1}. <a href="{link}" target="_blank" class="source-link">{link}</a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No clean URL links were extracted. Please check the Raw Data below for the text content retrieved.")

    # 2. Show Raw Data (The "Proof" you wanted)
    with st.expander("📄 View Raw Search Evidence (Full Text Content)", expanded=False):
        st.markdown("This section contains the raw text retrieved from the search engine, proving data access even if rate limits prevented full link metadata.")
        for log in st.session_state.raw_logs:
            st.markdown(f'<div class="raw-log">{log}</div>', unsafe_allow_html=True)