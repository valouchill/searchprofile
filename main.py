# AI Recruiter PRO — v18.0 (Architecture Hybride CV-Thèque + Audit)
# -------------------------------------------------------------------
import streamlit as st
import json, io, re, uuid, time
from typing import Optional, Dict, List, Any
from copy import deepcopy
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# Clients API
import openai
from pypdf import PdfReader
from supabase import create_client, Client

# -----------------------------
# 0. CONFIGURATION & SETUP
# -----------------------------
st.set_page_config(page_title="AI Recruiter PRO v18", layout="wide", page_icon="🧠")

# CSS (Gardé de ta version précédente pour le style)
st.markdown("""
<style>
    :root { --primary:#2563eb; --score-good:#16a34a; --score-bad:#dc2626; }
    .stApp { background: #f8fafc; font-family: 'Inter', sans-serif; }
    .score-badge { font-size: 1.5rem; font-weight: 900; color: white; width: 60px; height: 60px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }
    .sc-good { background: #16a34a; } .sc-mid { background: #d97706; } .sc-bad { background: #dc2626; }
    .evidence-box { background: #f1f5f9; border-left: 4px solid #cbd5e1; padding: 10px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 1. CONNEXIONS (SUPABASE & IA)
# -----------------------------
@st.cache_resource
def init_connections():
    try:
        # Supabase
        supa_url = st.secrets["supabase"]["url"]
        supa_key = st.secrets["supabase"]["key"]
        supabase: Client = create_client(supa_url, supa_key)
        
        # OpenAI (Pour Embeddings seulement)
        openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Groq (Pour l'Audit Llama 3)
        groq_client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=st.secrets["GROQ_API_KEY"]
        )
        return supabase, openai_client, groq_client
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")
        return None, None, None

supabase, openai_client, groq_client = init_connections()

# -----------------------------
# 2. SCHÉMAS DE DONNÉES (Ta structure JSON)
# -----------------------------
class Infos(BaseModel):
    nom: str = "Candidat"; email: str = "N/A"; tel: str = "N/A"; ville: str = ""; linkedin: str = ""; poste_actuel: str = ""

class Scores(BaseModel):
    global_: int = Field(0, alias="global"); tech: int = 0; experience: int = 0; fit: int = 0

class Preuve(BaseModel):
    skill: str; preuve: str; niveau: str

class Competences(BaseModel):
    match_details: List[Preuve] = []; manquant_critique: List[str] = []; manquant_secondaire: List[str] = []

class Analyse(BaseModel):
    verdict_auditeur: str = "En attente"; red_flags: List[str] = []

class CandidateData(BaseModel):
    infos: Infos = Infos()
    scores: Scores = Scores()
    analyse: Analyse = Analyse()
    competences: Competences = Competences()
    historique: List[dict] = []
    entretien: List[dict] = []

DEFAULT_DATA = CandidateData().dict(by_alias=True)

# -----------------------------
# 3. FONCTIONS "INTELLIGENTES"
# -----------------------------

def clean_pdf_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text[:8000] # Limite augmentée pour stockage

def extract_pdf_safe(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return clean_pdf_text(text)
    except: return ""

# --- A. LE TAMIS (Vectorisation) ---
def get_embedding(text: str) -> List[float]:
    """Transforme un texte en vecteur via OpenAI"""
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def ingest_cv_to_db(file, text):
    """Calcule le vecteur et envoie tout dans Supabase"""
    vector = get_embedding(text)
    data, count = supabase.table('candidates').insert({
        "nom_fichier": file.name,
        "contenu_texte": text,
        "embedding": vector
    }).execute()
    return data

# --- B. LE MICROSCOPE (Audit Groq) ---
AUDITOR_PROMPT = """
ROLE: Auditeur de Recrutement Impitoyable.
TACHE: Vérifier factuellement l'adéquation CV vs OFFRE.
PRINCIPE: "Pas écrit = Pas acquis".
REGLES PUNITIVES:
1. Si un 'CRITERE IMPERATIF' manque explicitement = Score Global max 40/100 (Disqualifié).
2. Départ 100 points. -10 par compétence manquante. -15 par red flag (trous, instabilité).
FORMAT JSON STRICT (Comme défini précédemment).
"""

def audit_candidate_groq(job: str, cv: str, criteria: str) -> dict:
    user_prompt = f"--- OFFRE ---\n{job[:2000]}\n\n--- CRITERES IMPERATIFS ---\n{criteria}\n\n--- CV CANDIDAT ---\n{cv[:3500]}"
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": AUDITOR_PROMPT}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"Err audit: {e}")
        return DEFAULT_DATA

# -----------------------------
# 4. INTERFACE UTILISATEUR (WALL D'AO vs INGESTION)
# -----------------------------

st.title("🧠 AI Recruiter PRO — Central Brain")

# Onglets principaux
tab_search, tab_ingest = st.tabs(["🔎 WALL D'AO (Matching)", "📥 INGESTION CV (Upload)"])

# --- ONGLET 1 : WALL D'AO (Le Matching) ---
with tab_search:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("1. L'Offre (AO)")
        job_txt = st.text_area("Collez la description du poste ici", height=300, placeholder="Recherche Développeur Python Senior...")
        criteria = st.text_area("Critères Éliminatoires (Dealbreakers)", height=100, placeholder="Anglais C1, 5 ans exp...")
        threshold = st.slider("Seuil de pertinence sémantique", 0.3, 0.8, 0.45, help="Plus c'est haut, plus c'est strict.")
        limit_search = st.number_input("Nombre de CVs à auditer", 5, 50, 10)
        
        run_search = st.button("🚀 LANCER LA CHASSE", type="primary")

    with c2:
        st.subheader("2. Résultats de l'Audit")
        if run_search and job_txt:
            with st.status("🧠 Activation du Cerveau Recruteur...", expanded=True) as status:
                
                # 1. Vector Search
                status.write("🔍 Analyse sémantique de l'offre...")
                job_vector = get_embedding(job_txt)
                
                status.write("📡 Interrogation de la CV-Thèque (Supabase)...")
                response = supabase.rpc('match_candidates', {
                    'query_embedding': job_vector,
                    'match_threshold': threshold,
                    'match_count': limit_search
                }).execute()
                
                candidates_found = response.data
                
                if not candidates_found:
                    status.update(label="❌ Aucun profil pertinent trouvé dans la base.", state="error")
                    st.error("Essayez de baisser le seuil de pertinence ou d'enrichir la base.")
                else:
                    status.write(f"✅ {len(candidates_found)} profils présélectionnés par le Tamis Sémantique.")
                    
                    # 2. Audit Punitif
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, cand in enumerate(candidates_found):
                        status.write(f"🔬 Audit approfondi de : {cand['nom_fichier']}...")
                        # On lance l'IA Auditrice sur le texte récupéré de la DB
                        audit_data = audit_candidate_groq(job_txt, cand['contenu_texte'], criteria)
                        
                        # On injecte le nom du fichier si l'IA ne l'a pas trouvé
                        if audit_data['infos']['nom'] == "Candidat Inconnu":
                            audit_data['infos']['nom'] = cand['nom_fichier']
                            
                        results.append(audit_data)
                        progress_bar.progress((i + 1) / len(candidates_found))
                    
                    status.update(label="🎉 Analyse Terminée !", state="complete")
                    
                    # 3. Affichage des résultats
                    sorted_res = sorted(results, key=lambda x: x['scores']['global'], reverse=True)
                    
                    for res in sorted_res:
                        score = res['scores']['global']
                        s_color = "sc-good" if score >= 70 else "sc-mid" if score >= 50 else "sc-bad"
                        
                        with st.expander(f"{res['infos']['nom']} — Score: {score}/100", expanded=(score >= 60)):
                            col_a, col_b = st.columns([4, 1])
                            with col_a:
                                st.markdown(f"**Verdict:** {res['analyse']['verdict_auditeur']}")
                                if res['competences']['manquant_critique']:
                                    st.error(f"⛔ Manquants: {', '.join(res['competences']['manquant_critique'])}")
                            with col_b:
                                st.markdown(f"<div class='score-badge {s_color}'>{score}</div>", unsafe_allow_html=True)

# --- ONGLET 2 : INGESTION (Alimenter la base) ---
with tab_ingest:
    st.header("Alimenter la CV-Thèque")
    st.info("Les CVs uploadés ici sont vectorisés et stockés à vie dans Supabase pour les futurs matchings.")
    
    uploaded_files = st.file_uploader("Chargez les CVs (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("💾 Ingérer et Indexer") and uploaded_files:
        bar = st.progress(0)
        success_count = 0
        
        for i, f in enumerate(uploaded_files):
            try:
                txt = extract_pdf_safe(f.read())
                if len(txt) > 50:
                    ingest_cv_to_db(f, txt)
                    success_count += 1
            except Exception as e:
                st.error(f"Erreur sur {f.name}: {e}")
            bar.progress((i+1)/len(uploaded_files))
            
        st.success(f"✅ {success_count} CVs ajoutés à la base de connaissances !")
