# app.py - Neuro-Symbolic UI with Configurable Sources
import streamlit as st
import json
import time
import os
import re
# Import the model engine
from llama_cpp import Llama
# Import the search tool
from langchain_community.tools.tavily_search import TavilySearchResults
# Import grammar constraint
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

# --- TAVILY API SETUP ---
# PASTE YOUR KEY HERE
os.environ["TAVILY_API_KEY"] = ""

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

def process_tavily_results(results):
    """
    Parses the list of results from Tavily.
    """
    links = []
    content_blob = ""
    
    if isinstance(results, list):
        for item in results:
            if 'url' in item:
                links.append(item['url'])
            if 'content' in item:
                content_blob += f"Source ({item.get('url', 'unknown')}): {item['content']}\n\n"
    elif isinstance(results, str):
        content_blob = results
        
    return links, content_blob

# ==========================================
# 3. CORE LOGIC
# ==========================================
def log(message, level="INFO"):
    """Print formatted log messages to terminal."""
    timestamp = time.strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "PLAN": "🧠", "SEARCH": "🔎", "CRITIC": "⚖️", "SUCCESS": "✅", "ERROR": "❌", "SYNTH": "✨"}
    icon = icons.get(level, "📌")
    print(f"[{timestamp}] {icon} [{level}] {message}")

def generate_plan(llm, grammar, objective, context, history):
    log("Starting Neural Planner...", "PLAN")
    plan_start = time.time()
    
    history_str = "\n".join([f"- {q}" for q in history]) if history else "None yet."
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an autonomous research agent working step-by-step toward an objective.

OBJECTIVE: {objective}

KNOWLEDGE GATHERED SO FAR: {context}

QUERIES ALREADY MADE (DO NOT REPEAT THESE): {history_str}

INSTRUCTIONS: 
1. Analyze what information you still need to fully answer the objective.
2. Generate a NEW search query that is DIFFERENT from all past queries.
3. Focus on the NEXT piece of missing information.
4. Output a JSON with "rationale" (why this query) and "query" (the search query).
Output ONLY valid JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    output = llm(prompt, max_tokens=200, grammar=grammar, temperature=0.3)
    plan_time = time.time() - plan_start
    
    try:
        result = json.loads(output['choices'][0]['text'])
        log(f"Plan generated in {plan_time:.2f}s - Query: '{result.get('query', 'N/A')[:50]}...'", "PLAN")
        log(f"Rationale: {result.get('rationale', 'N/A')[:80]}...", "PLAN")
        return result
    except:
        log(f"Plan generation failed after {plan_time:.2f}s", "ERROR")
        return None

def run_critic(llm, query, data):
    log(f"Running Critic on query: '{query[:50]}...'", "CRITIC")
    critic_start = time.time()
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Query: {query}. Data: {data[:1500]}. Is this relevant? YES or NO.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    output = llm(prompt, max_tokens=10)
    decision = "YES" in output['choices'][0]['text'].strip().upper()
    
    critic_time = time.time() - critic_start
    verdict = "RELEVANT ✓" if decision else "NOISE ✗"
    log(f"Critic verdict: {verdict} (took {critic_time:.2f}s)", "CRITIC")
    
    return decision

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🧠 Neuro-Symbolic Deep Research")
st.caption(f"Model: {os.path.basename(MODEL_PATH)}")

# Initialize Session State
if "history" not in st.session_state: st.session_state.history = []
if "sources" not in st.session_state: st.session_state.sources = []
if "raw_logs" not in st.session_state: st.session_state.raw_logs = [] 
if "context" not in st.session_state: st.session_state.context = "No knowledge yet."
if "result" not in st.session_state: st.session_state.result = ""

with st.sidebar:
    st.header("Settings")
    # UPDATED: Slider for Iterations
    max_iter = st.slider("Research Depth (Iterations)", 2, 7, 4)
    # NEW: Slider for Sources per Search
    num_sources = st.slider("Sources per Search", 2, 10, 5, help="Number of distinct websites to fetch per query. Higher = more variety.")
    
    if st.button("Reset System"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

user_query = st.text_area("Research Objective:", height=100, 
    value="Identify the host city for the 2028 Summer Olympics. Then, find the average temperature in that city during the month of July. Finally, suggest three outdoor Olympic events suitable for that specific weather.")

if st.button("Initiate Research"):
    # Clear previous run
    st.session_state.history = []
    st.session_state.sources = []
    st.session_state.raw_logs = []
    st.session_state.context = "No knowledge yet."
    st.session_state.result = ""
    st.session_state.total_time = 0
    
    print("\n" + "="*60)
    log("🚀 RESEARCH SESSION STARTED", "INFO")
    log(f"Objective: {user_query[:100]}...", "INFO")
    print("="*60)
    
    llm_engine = load_model()
    grammar_engine = load_grammar()
    
    # Initialize Search Tool DYNAMICALLY with user config
    search_tool = TavilySearchResults(max_results=num_sources)
    log(f"Search tool initialized with max_results={num_sources}", "INFO")
    
    status = st.empty()
    progress = st.progress(0)
    start_time = time.time()
    
    # --- LOOP ---
    for i in range(max_iter):
        print("\n" + "-"*40)
        log(f"ITERATION {i+1}/{max_iter} STARTED", "INFO")
        print("-"*40)
        
        iter_start = time.time()
        status.text(f"Phase {i+1}/{max_iter}: Planning & Searching ({num_sources} sources)...")
        progress.progress(int((i / max_iter) * 90))
        
        # 1. Plan
        action = generate_plan(llm_engine, grammar_engine, user_query, st.session_state.context, st.session_state.history)
        if not action:
            log("No action generated, breaking loop", "ERROR")
            break
        query = action.get('query')
        
        # Handle duplicate queries - retry up to 3 times with higher temperature
        retry_count = 0
        while query in st.session_state.history and retry_count < 3:
            retry_count += 1
            log(f"Duplicate query detected, retrying ({retry_count}/3): '{query[:40]}...'", "INFO")
            action = generate_plan(llm_engine, grammar_engine, user_query, st.session_state.context, st.session_state.history)
            if action:
                query = action.get('query')
            else:
                break
        
        if query in st.session_state.history:
            log(f"Still duplicate after retries, skipping iteration", "INFO")
            continue
            
        st.session_state.history.append(query)
        
        # 2. Search (TAVILY)
        log(f"Executing search: '{query[:50]}...'", "SEARCH")
        search_start = time.time()
        try:
            # Tavily returns a LIST of objects
            raw_result_obj = search_tool.run(query)
            search_time = time.time() - search_start
            
            # Helper to parse Tavily's specific format
            links, content_text = process_tavily_results(raw_result_obj)
            log(f"Search completed in {search_time:.2f}s - Found {len(links)} sources", "SEARCH")
            
            # Save logs
            st.session_state.raw_logs.append(f"🔎 QUERY: {query}\n📄 CONTENT: {content_text[:500]}...")
            
            # Save Links
            if links:
                st.session_state.sources.extend(links)
                for link in links:
                    log(f"Source: {link}", "SUCCESS")
                    st.toast(f"Source Verified: {link}", icon="🔗")
            else:
                log(f"No links found for query", "INFO")
                st.toast(f"No links found for {query}", icon="⚠️")

        except Exception as e:
            content_text = ""
            log(f"Search failed: {str(e)}", "ERROR")
            st.session_state.raw_logs.append(f"❌ ERROR for '{query}': {str(e)}")

        # 3. Critic
        if content_text and run_critic(llm_engine, query, content_text):
            st.session_state.context += f"\n[Fact]: {content_text[:1000]}"
            log("Data added to knowledge base", "SUCCESS")
        
        iter_time = time.time() - iter_start
        log(f"Iteration {i+1} completed in {iter_time:.2f}s", "INFO")

    # --- SYNTHESIS ---
    print("\n" + "="*40)
    log("SYNTHESIS PHASE STARTED", "SYNTH")
    print("="*40)
    
    status.text("Finalizing: Synthesizing Report...")
    progress.progress(95)
    
    synth_start = time.time()
    final_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Synthesize a comprehensive answer based ONLY on these notes: {st.session_state.context}. Question: {user_query}. Use Markdown.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    final_output = llm_engine(final_prompt, max_tokens=1500)
    st.session_state.result = final_output['choices'][0]['text']
    synth_time = time.time() - synth_start
    log(f"Synthesis completed in {synth_time:.2f}s", "SYNTH")
    
    total_time = time.time() - start_time
    st.session_state.total_time = total_time
    
    print("\n" + "="*60)
    log(f"🏁 RESEARCH SESSION COMPLETED", "INFO")
    log(f"Total Time: {total_time:.2f}s", "INFO")
    log(f"Iterations: {len(st.session_state.history)}", "INFO")
    log(f"Sources Found: {len(st.session_state.sources)}", "INFO")
    print("="*60 + "\n")
    
    progress.progress(100)
    time.sleep(0.5)
    status.empty()
    progress.empty()

# --- DISPLAY RESULTS ---
if st.session_state.result:
    st.divider()
    
    # Display time taken
    if "total_time" in st.session_state and st.session_state.total_time > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Total Time", f"{st.session_state.total_time:.2f}s")
        with col2:
            st.metric("🔄 Iterations", len(st.session_state.history))
        with col3:
            st.metric("🔗 Sources", len(set(st.session_state.sources)))
    
    st.subheader("Final Research Report")
    st.markdown(st.session_state.result)
    
    st.markdown("---")
    st.subheader("📚 Verified Research Sources")
    
    if st.session_state.sources:
        unique = sorted(list(set(st.session_state.sources)))
        with st.container():
            st.markdown('<div class="source-box">', unsafe_allow_html=True)
            st.write(f"The agent successfully extracted data from {len(unique)} unique external URLs:")
            for i, link in enumerate(unique):
                st.markdown(f'{i+1}. <a href="{link}" target="_blank" class="source-link">{link}</a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No external sources were successfully validated.")

    # Show Raw Data Fallback
    with st.expander("📄 View Raw Search Evidence (Full Text Content)", expanded=False):
        for log in st.session_state.raw_logs:
            st.markdown(f'<div class="raw-log">{log}</div>', unsafe_allow_html=True)