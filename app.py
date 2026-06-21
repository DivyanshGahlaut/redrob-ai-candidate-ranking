import streamlit as st
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add workspace root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Target file paths
REPO_ROOT = Path(__file__).resolve().parent
SUBMISSION_PATH = REPO_ROOT / "outputs" / "submission.csv"
CANDIDATES_PATH = REPO_ROOT / "pub_dataset" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
if not CANDIDATES_PATH.exists():
    # Fallback to search in workspace
    search_paths = list(REPO_ROOT.glob("**/candidates.jsonl"))
    if search_paths:
        CANDIDATES_PATH = search_paths[0]

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Redrob AI Recruiter Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS for modern UI design (Glassmorphism & harmonized colors)
st.markdown("""
<style>
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #50E3C2;
        --bg-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(90deg, #4A90E2, #50E3C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .candidate-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        border-color: rgba(74, 144, 226, 0.5);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #50E3C2;
    }
    .badge-top {
        background-color: rgba(80, 227, 194, 0.15);
        color: #50E3C2;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-mid {
        background-color: rgba(74, 144, 226, 0.15);
        color: #4A90E2;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-low {
        background-color: rgba(149, 165, 166, 0.15);
        color: #95a5a6;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_type=True)

# Helper: Load ranked list
@st.cache_data
def load_ranked_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Helper: Load a specific candidate's full profile from JSONL
def find_candidate_profile(candidate_id):
    if not os.path.exists(CANDIDATES_PATH):
        return None
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if candidate_id in line:
                cand = json.loads(line)
                if cand.get("candidate_id") == candidate_id:
                    return cand
    return None

# --- Main App ---
st.markdown('<div class="main-header">Redrob AI Recruiter Assistant</div>', unsafe_allow_type=True)
st.markdown('<div class="sub-header">Premium talent discovery intelligence, scoring explanation, and conversational recruiter query answering.</div>', unsafe_allow_type=True)

# Initialize Gemini Client if Key Provided
api_key = os.environ.get("GEMINI_API_KEY", "")
with st.sidebar:
    st.image("https://redrob.io/static/media/logo.5492d04a.svg", width=150)
    st.markdown("---")
    st.markdown("### LLM API Configuration")
    user_key = st.text_input("Gemini API Key", value=api_key, type="password", help="Enter your Gemini API key to enable conversational explanations.")
    if user_key:
        api_key = user_key
        os.environ["GEMINI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("### Pipeline Mode Status")
    embeddings_file = REPO_ROOT / "data" / "candidate_embeddings.npy"
    if embeddings_file.exists():
        st.success("🤖 Transformer Mode Active (Sentence-Transformers + FAISS)")
    else:
        st.warning("📉 Fallback Mode Active (TF-IDF + SVD)")

df_ranks = load_ranked_csv(SUBMISSION_PATH)

if df_ranks is None:
    st.info("No submission output found. Please run the ranking pipeline first using the command:")
    st.code("python rank.py --candidates <path_to_candidates.jsonl> --out outputs/submission.csv")
else:
    # Set up layout columns
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Top Ranked Candidates (Shortlist)")
        
        # Display the ranked list with clean details
        search_query = st.text_input("🔍 Search shortlist by Candidate ID or description text:")
        
        filtered_df = df_ranks
        if search_query:
            filtered_df = df_ranks[
                df_ranks['candidate_id'].str.contains(search_query, case=False) |
                df_ranks['reasoning'].str.contains(search_query, case=False)
            ]

        for _, row in filtered_df.iterrows():
            cid = row['candidate_id']
            rank = row['rank']
            score = row['score']
            reasoning = row['reasoning']
            
            band = "badge-top" if rank <= 10 else ("badge-mid" if rank <= 50 else "badge-low")
            band_lbl = "Top 10" if rank <= 10 else ("Top 50" if rank <= 50 else "Shortlisted")
            
            with st.container():
                st.markdown(f"""
                <div class="candidate-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight: 700; font-size: 1.1rem; color:#4A90E2;">Rank #{rank} — {cid}</span>
                        <span class="{band}">{band_lbl}</span>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.95rem; line-height: 1.4;">
                        <strong>Score:</strong> <span class="metric-value">{score:.4f}</span><br>
                        <strong>AI Summary:</strong> {reasoning}
                    </div>
                </div>
                """, unsafe_allow_type=True)
                
                # Selection button to trigger details view in Col 2
                if st.button(f"Inspect Profile & Ask AI about {cid}", key=f"btn_{cid}"):
                    st.session_state["selected_candidate_id"] = cid

    with col2:
        st.subheader("Interactive AI Assistant")
        
        # Inspect Selected Profile
        selected_cid = st.session_state.get("selected_candidate_id")
        if not selected_cid and len(df_ranks) > 0:
            selected_cid = df_ranks.iloc[0]['candidate_id']
            st.session_state["selected_candidate_id"] = selected_cid
            
        if selected_cid:
            st.markdown(f"### Profile Inspection: **{selected_cid}**")
            with st.spinner("Retrieving full profile..."):
                profile_data = find_candidate_profile(selected_cid)
            
            if profile_data:
                # Layout profile overview
                prof = profile_data.get("profile", {})
                st.write(f"**Current Role:** {prof.get('current_title', 'N/A')} at {prof.get('current_company', 'N/A')}")
                st.write(f"**Experience:** {prof.get('years_of_experience', 0)} years | **Location:** {prof.get('location', 'N/A')}")
                
                # Show tabs for structured details
                tab1, tab2, tab3 = st.tabs(["Skills & Signals", "Work History", "Education"])
                with tab1:
                    skills_list = [f"{s.get('name')} ({s.get('proficiency')}, {s.get('duration_months', 0)}mo)" for s in profile_data.get("skills", [])]
                    st.write("**Claimed Skills:**", ", ".join(skills_list) if skills_list else "None listed")
                    
                    signals = profile_data.get("redrob_signals", {})
                    st.write("**Platform Signals:**")
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("Recruiter Response Rate", f"{signals.get('recruiter_response_rate', 0.0):.0%}")
                    col_s1.metric("Interview Completion Rate", f"{signals.get('interview_completion_rate', 0.0):.0%}")
                    col_s2.metric("Notice Period", f"{signals.get('notice_period_days', 30)} days")
                    col_s2.metric("Open to Work", "Yes" if signals.get("open_to_work_flag") else "No")

                with tab2:
                    for i, role in enumerate(profile_data.get("career_history", [])):
                        st.write(f"**{role.get('title')}** at *{role.get('company')}* ({role.get('duration_months', 0)} months)")
                        st.write(f"_{role.get('description', '')}_")
                        st.write("---")
                
                with tab3:
                    for i, edu in enumerate(profile_data.get("education", [])):
                        st.write(f"**{edu.get('degree', 'Degree')}** in {edu.get('field', 'Field')}")
                        st.write(f"*{edu.get('school')}* ({edu.get('start_year', 'N/A')} - {edu.get('end_year', 'N/A')})")
                        st.write("---")

                # Conversational AI Explanations
                st.markdown("### Ask AI Assistant about this Candidate")
                
                if not api_key:
                    st.info("⚠️ Please enter a Gemini API Key in the sidebar to enable interactive conversational Q&A.")
                else:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    
                    # Predefined quick questions
                    quick_q = st.selectbox("Quick Queries:", [
                        "Select a quick query...",
                        "Explain why this candidate was ranked at this position.",
                        "What are this candidate's main gaps or risks against the JD?",
                        "Summarize their career progression and stability.",
                        "Is there any risk of them being a keyword-stuffer or honeypot?"
                    ])
                    
                    custom_q = st.text_input("Or enter a custom question:")
                    
                    query = custom_q if custom_q else (quick_q if quick_q != "Select a quick query..." else None)
                    
                    if query:
                        with st.spinner("AI Assistant is analyzing..."):
                            # Construct prompt grounding the model in the candidate profile
                            prompt = f"""
                            You are an expert AI recruiter assistant. You have access to a full candidate profile and their ranking information against a Senior AI Engineer job description.
                            
                            Rank Details:
                            - Candidate ID: {selected_cid}
                            - Rank: {df_ranks[df_ranks['candidate_id'] == selected_cid]['rank'].values[0]}
                            - Score: {df_ranks[df_ranks['candidate_id'] == selected_cid]['score'].values[0]:.4f}
                            - Base Reasoning Summary: {df_ranks[df_ranks['candidate_id'] == selected_cid]['reasoning'].values[0]}
                            
                            Candidate Profile Data:
                            {json.dumps(profile_data, indent=2)}
                            
                            User Question: {query}
                            
                            Please answer the user's question concisely, referencing specific facts (years, companies, response rates, skill weights) from the profile data. Be honest about any risks or concerns. Do not make up any facts.
                            """
                            try:
                                response = model.generate_content(prompt)
                                st.markdown("#### AI Response:")
                                st.write(response.text)
                            except Exception as e:
                                st.error(f"Failed to generate response: {e}")
            else:
                st.error("Full profile could not be found in the dataset.")
