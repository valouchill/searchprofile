# AI Recruiter PRO — v29.0 (JSON RESCUE EDITION)
# -------------------------------------------------------------------
import streamlit as st
import json, io, re, uuid, time
from typing import Optional, List, Dict, Any
from copy import deepcopy
import pandas as pd
from pydantic import BaseModel, Field

# Clients API
import openai
from pypdf import PdfReader
from supabase import create_client, Client

# -----------------------------
# 0. CONFIGURATION & STYLE
# -----------------------------
st.set_page_config(page_title="AI Recruiter PRO v29", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    :root { --primary:#dc2626; --bg-app:#f8fafc; --text-main:#0f172a; }
    .stApp { background: var(--bg-app); color: var(--text-main); font-family: 'Inter', sans-serif; }
    .stButton button { border-radius: 8px; font-weight: 700; height: 45px; }
    .score-badge { 
        font-size: 1.6rem; font-weight: 900; color: white; 
        width: 65px; height: 65px; border-radius: 14px; 
        display: flex; align-items: center; justify-content: center;
    }
    .sc-good { background: #16a34a; } .sc-mid { background: #d97706; } .sc-bad { background: #dc2626; }
    .evidence-box { background: #fff; border-left: 4px solid #cbd5e1; padding: 10px; margin-bottom: 8px; }
    .ev-missing { border-left-color: #ef4444; background: #fff1f2; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 1. CONNEXIONS
# -----------------------------
@st.cache_resource
def init_connections():
    try:
        supa_url = st.secrets["supabase"]["url"]
        supa_key = st.secrets["supabase"]["key"]
        supabase: Client = create_client(supa_url, supa_key)
        
        openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        groq_client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
        return supabase, openai_client, groq_client
    except Exception as e:
        st.error(f"⚠️ Erreur Connexion : {e}")
        return None, None, None

supabase, openai_client, groq_client = init_connections()

# -----------------------------
# 2. MODELS DE DONNÉES
# -----------------------------
class Infos(BaseModel):
    nom: str = "Candidat Inconnu"; email: str = "N/A"; tel: str = "N/A"; ville: str = ""; linkedin: str = ""; poste_actuel: str = ""

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
    historique: List[dict] = []; entretien: List[dict] = []
    raw_response: str = "" 

DEFAULT_DATA = CandidateData().dict(by_alias=True)

# -----------------------------
# 3. FONCTIONS LOGIQUES
# -----------------------------
def clean_pdf_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text[:8000]

def extract_pdf_safe(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        clean = clean_pdf_text(text)
        if len(clean) < 50: return ""
        return clean
    except: return ""

def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    for attempt in range(3):
        try:
            return openai_client.embeddings.create(input=[text[:8000]], model="text-embedding-3-small").data[0].embedding
        except openai.RateLimitError: time.sleep(2)
        except Exception: break
    st.error("❌ Erreur OpenAI Embedding.")
    st.stop()

def ingest_cv_to_db(file, text):
    vector = get_embedding(text)
    supabase.table('candidates').insert({
        "nom_fichier": file.name, "contenu_texte": text, "embedding": vector
    }).execute()

# --- NETTOYEUR DE JSON (LE SAUVEUR) ---
def clean_json_string(text: str) -> str:
    """Enlève le markdown ```json ... ``` et les commentaires"""
    # Enlever les balises code markdown
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    # Chercher le premier { et le dernier }
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end != -1:
        return text[start:end]
    return text

def safe_json_loads(text: str) -> Dict:
    """Tente de charger le JSON même s'il est sale"""
    cleaned = clean_json_string(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

# --- PROMPT SIMPLIFIÉ POUR LLAMA 8B ---
AUDITOR_PROMPT = """
Tu es un recruteur expert. Analyse le CV par rapport à l'offre (AO).
Règles:
1. Si un critère critique manque, score < 30.
2. Si le CV est bon, score > 70.
3. REPONDS UNIQUEMENT AVEC LE JSON CI-DESSOUS. RIEN D'AUTRE.

{
    "infos": { "nom": "Nom trouvé ou Inconnu", "poste_actuel": "..." },
    "scores": { "global": 0, "tech": 0, "experience": 0, "fit": 0 },
    "competences": {
        "match_details": [ {"skill": "...", "preuve": "...", "niveau": "..."} ],
        "manquant_critique": ["..."],
        "manquant_secondaire": ["..."]
    },
    "analyse": { "verdict_auditeur": "...", "red_flags": ["..."] },
    "historique": [ {"titre": "...", "entreprise": "...", "duree": "..."} ],
    "entretien": [ {"question": "...", "reponse_attendue": "..."} ]
}
"""

def audit_candidate_groq(ao_text: str, cv_text: str, criteria: str) -> dict:
    # On réduit le contexte pour économiser des tokens
    user_prompt = f"AO: {ao_text[:1000]}... CRITERES: {criteria}... CV: {cv_text[:1500]}..."
    
    safe_data = deepcopy(DEFAULT_DATA)
    
    try:
        res = groq_client.chat.completions.create(
            # Modèle léger
            model="llama-3.1-8b-instant", 
            messages=[{"role": "system", "content": AUDITOR_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            # Force JSON Mode (Important pour Llama 3.1)
            response_format={"type": "json_object"} 
        )
        
        raw_content = res.choices[0].message.content
        safe_data['raw_response'] = raw_content
        
        # Tentative de parsing robuste
        ai_json = safe_json_loads(raw_content)
        
        if ai_json:
            for key, value in ai_json.items():
                if key in safe_data and isinstance(safe_data[key], dict) and isinstance(value, dict):
                    safe_data[key].update(value)
                else: safe_data[key] = value
        else:
             safe_data['analyse']['verdict_auditeur'] = "Erreur JSON: L'IA a mal formaté la réponse."

        return safe_data
        
    except openai.RateLimitError:
        safe_data['analyse']['verdict_auditeur'] = "⚠️ Erreur 429: Quota Groq dépassé."
        return safe_data
    except Exception as e:
        safe_data['analyse']['verdict_auditeur'] = f"Erreur: {str(e)}"
        return safe_data

# -----------------------------
# 4. INTERFACE
# -----------------------------
st.title("🛡️ AI Recruiter PRO — V29 (JSON Fix)")

# --- TABS ---
tab_search, tab_ingest = st.tabs(["🔎 AUDIT", "📥 INGESTION"])

# --- ONGLET 1 : RECHERCHE ---
with tab_search:
    col_upload, col_criteria = st.columns([1, 1])
    ao_content = ""
    
    with col_upload:
        st.subheader("1. L'Offre (AO)")
        ao_pdf = st.file_uploader("Fiche de Poste (PDF)", type="pdf")
        ao_manual = st.text_area("Ou texte", height=100)
        
        if ao_pdf:
            txt = extract_pdf_safe(ao_pdf.read())
            if txt: 
                ao_content = txt
                st.success(f"✅ PDF lu ({len(txt)} chars)")
            else: st.error("⚠️ PDF vide.")
        elif ao_manual: ao_content = ao_manual

    with col_criteria:
        st.subheader("2. Critères")
        criteria = st.text_area("Dealbreakers", height=100)
        threshold = st.slider("Seuil Matching", 0.3, 0.8, 0.45)
        limit = st.number_input("Nb Profils", 1, 20, 5)
    
    st.divider()
    
    if st.button("🚀 LANCER L'AUDIT", type="primary"):
        if not ao_content:
            st.error("⚠️ Texte de l'offre vide.")
        else:
            with st.status("Traitement...", expanded=True) as status:
                q_vec = get_embedding(ao_content[:8000])
                res_db = supabase.rpc('match_candidates', {'query_embedding': q_vec, 'match_threshold': threshold, 'match_count': limit}).execute()
                cands = res_db.data
                
                if not cands:
                    status.update(label="❌ 0 Candidat trouvé", state="error")
                else:
                    status.write(f"✅ {len(cands)} profils. Analyse IA...")
                    final_results = []
                    bar = st.progress(0)
                    
                    for i, c in enumerate(cands):
                        audit = audit_candidate_groq(ao_content, c['contenu_texte'], criteria)
                        
                        infos = audit.get('infos', {})
                        if not infos.get('nom') or infos.get('nom') == "Candidat Inconnu":
                            if 'infos' not in audit: audit['infos'] = {}
                            audit['infos']['nom'] = c.get('nom_fichier', 'Fichier')
                            
                        final_results.append(audit)
                        bar.progress((i+1)/len(cands))
                    
                    status.update(label="Terminé", state="complete")
                    final_results.sort(key=lambda x: x.get('scores', {}).get('global', 0), reverse=True)
                    
                    st.subheader("Résultats")
                    
                    for i, r in enumerate(final_results):
                        sc = r.get('scores', {}).get('global', 0)
                        nom = r.get('infos', {}).get('nom', 'Inconnu')
                        
                        # CLÉ UNIQUE POUR DEBUG
                        with st.expander(f"{nom} — Score {sc}/100", expanded=(sc>=0)):
                            
                            # DEBUG: Afficher la réponse brute si JSON invalide
                            if "Erreur JSON" in r['analyse'].get('verdict_auditeur', ''):
                                st.error("L'IA a renvoyé du texte invalide. Voici le contenu brut :")
                                st.text_area("Raw", r.get('raw_response', ''), height=100, key=f"bug_{i}")
                            
                            c1, c2 = st.columns([4, 1])
                            c1.info(r['analyse'].get('verdict_auditeur'))
                            c2.metric("Score", sc)
                            
                            manq = r.get('competences', {}).get('manquant_critique', [])
                            if manq: c1.error(f"Manques : {manq}")

# --- ONGLET 2 : INGESTION ---
with tab_ingest:
    st.header("CV-Thèque")
    files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Indexer") and files:
        bar = st.progress(0)
        for i, f in enumerate(files):
            try:
                txt = extract_pdf_safe(f.read())
                if len(txt) > 50: ingest_cv_to_db(f, txt)
            except: pass
            bar.progress((i+1)/len(files))
        st.success("Fini")
