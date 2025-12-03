# AI Recruiter PRO — v20.0 (Version Finale Stable)
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
st.set_page_config(page_title="AI Recruiter PRO v20", layout="wide", page_icon="🔍")

st.markdown("""
<style>
    :root { --primary:#2563eb; --bg:#f8fafc; }
    .stApp { background: var(--bg); font-family: 'Inter', sans-serif; }
    
    /* STYLE BARRE DE RECHERCHE */
    .stTextInput input { font-size: 1.2rem; padding: 10px; border-radius: 25px; border: 2px solid #e2e8f0; }
    .stButton button { border-radius: 25px; padding: 0 30px; font-weight: bold; }
    
    /* CARDS RÉSULTATS */
    .score-badge { font-size: 1.4rem; font-weight: 900; color: white; width: 50px; height: 50px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
    .sc-good { background: #16a34a; } .sc-mid { background: #d97706; } .sc-bad { background: #dc2626; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 1. CONNEXIONS SÉCURISÉES
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
        st.error(f"⚠️ Erreur Connexion Secrets : {e}")
        return None, None, None

supabase, openai_client, groq_client = init_connections()

# -----------------------------
# 2. MODELS DE DONNÉES (STRUCTURE)
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

DEFAULT_DATA = CandidateData().dict(by_alias=True)

# -----------------------------
# 3. FONCTIONS UTILITAIRES & ROBUSTES
# -----------------------------

def clean_pdf_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text[:8000]

def extract_pdf_safe(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return clean_pdf_text(text)
    except: return ""

# --- FONCTION EMBEDDING AVEC RETRY AUTOMATIQUE (Anti-Crash) ---
def get_embedding(text: str) -> List[float]:
    """Tente de vectoriser le texte. Réessaie si OpenAI bloque (Rate Limit)."""
    text = text.replace("\n", " ")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding
        except openai.RateLimitError:
            wait = (attempt + 1) * 2
            st.toast(f"⚠️ OpenAI Rate Limit. Pause de {wait}s...", icon="⏳")
            time.sleep(wait)
        except Exception as e:
            st.error(f"Erreur OpenAI Critique: {e}")
            st.stop()
            
    st.error("❌ Impossible de contacter OpenAI après 3 essais. Vérifiez vos crédits.")
    st.stop()

def ingest_cv_to_db(file, text):
    """Envoie le CV et son vecteur dans Supabase"""
    vector = get_embedding(text)
    supabase.table('candidates').insert({
        "nom_fichier": file.name, "contenu_texte": text, "embedding": vector
    }).execute()

def save_search_history(query, criteria, count):
    """Sauvegarde l'AO dans l'historique"""
    try:
        supabase.table('search_history').insert({
            "query_text": query, "criteria_used": criteria, "results_count": count
        }).execute()
    except: pass

# -----------------------------
# 4. LE CERVEAU (AUDITEUR IA BLINDÉ)
# -----------------------------
AUDITOR_PROMPT = """
ROLE: Auditeur de Recrutement.
TACHE: Score CV vs RECHERCHE UTILISATEUR.
REGLES:
1. Si critère impératif absent = Score < 40.
2. Structure JSON stricte requise.
"""

def audit_candidate_groq(query: str, cv: str, criteria: str) -> dict:
    """Version crash-proof : utilise des valeurs par défaut si l'IA échoue"""
    user_prompt = f"--- DEMANDE RECRUTEUR ---\n{query}\n\n--- CRITERES ---\n{criteria}\n\n--- CV ---\n{cv[:3500]}"
    
    # 1. Structure de secours
    safe_data = deepcopy(DEFAULT_DATA)

    try:
        # 2. Appel IA
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": AUDITOR_PROMPT}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}, 
            temperature=0.0
        )
        
        # 3. Parsing et Fusion sécurisée
        ai_json = json.loads(res.choices[0].message.content)
        
        for key, value in ai_json.items():
            if key in safe_data and isinstance(safe_data[key], dict) and isinstance(value, dict):
                safe_data[key].update(value) # Fusionne les sous-dictionnaires (infos, scores)
            else:
                safe_data[key] = value # Remplace le reste (listes)
                
        return safe_data

    except Exception as e:
        print(f"Erreur Audit: {e}")
        return safe_data # Retourne la structure vide mais valide en cas de pépin

# -----------------------------
# 5. INTERFACE UTILISATEUR
# -----------------------------
st.title("🔍 Google of Recruitment")

# --- SIDEBAR : HISTORIQUE ---
with st.sidebar:
    st.header("🗂️ Mes Appels d'Offres")
    if st.button("🔄 Rafraîchir"): st.rerun()
    
    try:
        # Récupération historique
        history = supabase.table('search_history').select("*").order('created_at', desc=True).limit(8).execute()
        for h in history.data:
            st.markdown(f"""
            <div style="padding:12px; background:white; border-radius:8px; border:1px solid #cbd5e1; margin-bottom:8px;">
                <div style="font-weight:600; font-size:0.9rem; color:#1e293b;">{h['query_text'][:35]}...</div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">
                    🎯 {h['results_count']} profils • {h['created_at'][:10]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    except: st.caption("Historique vide ou erreur DB.")

# --- TABS PRINCIPAUX ---
tab_search, tab_ingest = st.tabs(["🔎 RECHERCHER UN PROFIL", "📥 AJOUTER DES CVs"])

# --- ONGLET RECHERCHE ---
with tab_search:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # BARRE DE RECHERCHE
    col_bar, col_go = st.columns([6, 1])
    with col_bar:
        search_query = st.text_input("Tapez votre besoin en langage naturel :", placeholder="Ex: Je cherche un Commercial B2B Expert SaaS qui parle Anglais...")
    with col_go:
        st.write("") # Spacer visuel
        st.write("")
        launch = st.button("RECHERCHER", type="primary", use_container_width=True)

    # OPTIONS AVANCÉES
    with st.expander("⚙️ Affiner les critères (Optionnel)"):
        criteria = st.text_area("Critères Bloquants (Dealbreakers)", height=70)
        threshold = st.slider("Précision Sémantique", 0.3, 0.8, 0.45)
        limit = st.number_input("Max CVs", 5, 50, 10)

    # RÉSULTATS
    if launch and search_query:
        st.divider()
        with st.status("🚀 Lancement de la recherche...", expanded=True) as status:
            
            # 1. Vector Search
            status.write("🧠 Compréhension de la demande...")
            q_vec = get_embedding(search_query)
            
            status.write("🗄️ Scan de la CV-Thèque...")
            res_db = supabase.rpc('match_candidates', {
                'query_embedding': q_vec, 'match_threshold': threshold, 'match_count': limit
            }).execute()
            
            cands = res_db.data
            count = len(cands)
            
            # 2. Sauvegarde AO
            status.write("💾 Création de l'Appel d'Offres...")
            save_search_history(search_query, criteria, count)
            
            if not cands:
                status.update(label="❌ Aucun candidat trouvé.", state="error")
                st.warning("Reformulez ou baissez la précision.")
            else:
                status.write(f"✅ {count} Profils identifiés. Démarrage de l'Audit IA...")
                
                # 3. Audit IA
                final_results = []
                bar = st.progress(0)
                
                for i, c in enumerate(cands):
                    # Audit sécurisé
                    audit = audit_candidate_groq(search_query, c['contenu_texte'], criteria)
                    
                    # Fallback Nom Fichier si l'IA n'a pas trouvé le nom
                    if audit['infos']['nom'] == "Candidat Inconnu": 
                        audit['infos']['nom'] = c['nom_fichier']
                    
                    final_results.append(audit)
                    bar.progress((i+1)/count)
                
                status.update(label="🎉 Recherche terminée !", state="complete")
                
                # AFFICHAGE TRIÉ
                final_results.sort(key=lambda x: x['scores']['global'], reverse=True)
                
                st.subheader(f"Résultats pour : {search_query}")
                for r in final_results:
                    sc = r['scores']['global']
                    color = "sc-good" if sc >= 70 else "sc-mid" if sc >= 50 else "sc-bad"
                    
                    with st.expander(f"{r['infos']['nom']} — {sc}/100", expanded=(sc>=60)):
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"**Verdict:** {r['analyse']['verdict_auditeur']}")
                            if r['competences']['manquant_critique']:
                                st.error(f"Manque: {', '.join(r['competences']['manquant_critique'])}")
                            
                            # Petits badges skills
                            if r['competences']['match_details']:
                                s_list = [f"`{x['skill']}`" for x in r['competences']['match_details'][:4]]
                                st.markdown("Skills: " + " ".join(s_list))
                                
                        with c2:
                            st.markdown(f"<div class='score-badge {color}'>{sc}</div>", unsafe_allow_html=True)

# --- ONGLET INGESTION ---
with tab_ingest:
    st.header("Alimenter la base")
    files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Indexation Vectorielle") and files:
        bar = st.progress(0)
        count_ok = 0
        for i, f in enumerate(files):
            try:
                txt = extract_pdf_safe(f.read())
                if len(txt) > 50: 
                    ingest_cv_to_db(f, txt)
                    count_ok += 1
            except Exception as e:
                st.toast(f"Erreur sur {f.name}")
            bar.progress((i+1)/len(files))
        st.success(f"{count_ok} CVs indexés avec succès !")
