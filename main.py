# AI Recruiter PRO — v24.0 (AO PDF Support + Scoring Matriciel Précis)
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
st.set_page_config(page_title="Valentin IC/Candi - Search", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    :root {
        --primary:#2563eb; --bg-app:#f8fafc; --text-main:#0f172a; --border:#cbd5e1;
        --score-good:#16a34a; --score-mid:#d97706; --score-bad:#dc2626;
    }
    .stApp { background: var(--bg-app); color: var(--text-main); font-family: 'Inter', sans-serif; }
    
    /* INPUTS */
    .stTextInput input, .stTextArea textarea { border-radius: 8px; border: 1px solid #cbd5e1; }
    .stButton button { border-radius: 8px; font-weight: 700; height: 45px; }

    /* CARD HEADER */
    .name-title { font-size: 1.4rem; font-weight: 800; color: #1e293b; margin: 0; }
    .job-subtitle { font-size: 0.9rem; color: #64748b; margin-top: 2px; font-weight: 500; }
    
    /* SCORE BADGE */
    .score-badge { 
        font-size: 1.6rem; font-weight: 900; color: white; 
        width: 65px; height: 65px; border-radius: 14px; 
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .sc-good { background: linear-gradient(135deg, #16a34a, #15803d); }
    .sc-mid { background: linear-gradient(135deg, #d97706, #b45309); }
    .sc-bad { background: linear-gradient(135deg, #dc2626, #b91c1c); }

    /* EVIDENCE BOXES */
    .section-header { font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin: 12px 0 8px 0; }
    .evidence-box { background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid #cbd5e1; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    .ev-skill { font-weight: 700; color: #334155; font-size: 0.9rem; }
    .ev-proof { font-size: 0.85rem; color: #475569; font-style: italic; margin-top: 2px; }
    .ev-missing { border-left-color: #ef4444; background: #fff1f2; }
    .tag { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-right: 5px; background: #eff6ff; color: #1d4ed8; border: 1px solid #dbeafe; }
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
    """Vectorisation avec Retry pour éviter les plantages"""
    text = text.replace("\n", " ")
    for attempt in range(3):
        try:
            return openai_client.embeddings.create(input=[text[:8000]], model="text-embedding-3-small").data[0].embedding
        except openai.RateLimitError:
            time.sleep((attempt + 1) * 2)
        except Exception: break
    st.error("❌ Erreur OpenAI. Vérifiez vos crédits.")
    st.stop()

def ingest_cv_to_db(file, text):
    vector = get_embedding(text)
    supabase.table('candidates').insert({
        "nom_fichier": file.name, "contenu_texte": text, "embedding": vector
    }).execute()

def save_search_history(query, criteria, count):
    try:
        supabase.table('search_history').insert({
            "query_text": query[:200], "criteria_used": criteria[:200], "results_count": count
        }).execute()
    except: pass

# --- NOUVEAU PROMPT : SCORING MATRICIEL ---
AUDITOR_PROMPT = """
ROLE: Expert Recrutement Senior (Analytique & Précis).
TACHE: Évaluer un CV par rapport à une Fiche de Poste (AO) complète.

SYSTÈME DE SCORING PONDÉRÉ (TOTAL /100) :
1. COMPÉTENCES TECH (40 pts) : Stack technique, outils, hard skills.
2. EXPÉRIENCE (30 pts) : Durée, pertinence du secteur, séniorité.
3. FIT & SOFT SKILLS (30 pts) : Clarté, présentation, mots-clés culturels.

RÈGLES DE DÉDUCTION :
- Ne mets JAMAIS 0 sauf si le CV est vide.
- Si une compétence critique manque : pénalité forte (-15 pts) mais pas élimination totale si le reste est excellent.
- Cherche les synonymes (ex: "Vente" = "Négociation").

FORMAT JSON REQUIS :
{
    "infos": { "nom": "...", "email": "...", "tel": "...", "ville": "...", "linkedin": "...", "poste_actuel": "..." },
    "scores": { "global": int, "tech": int (0-40), "experience": int (0-30), "fit": int (0-30) },
    "competences": {
        "match_details": [ {"skill": "Nom", "preuve": "Preuve trouvée", "niveau": "Fort/Moyen"} ],
        "manquant_critique": ["..."],
        "manquant_secondaire": ["..."]
    },
    "analyse": { "verdict_auditeur": "Analyse nuancée...", "red_flags": ["..."] },
    "historique": [ {"titre": "...", "entreprise": "...", "duree": "..."} ],
    "entretien": [ {"question": "...", "reponse_attendue": "..."} ]
}
"""

def audit_candidate_groq(ao_text: str, cv_text: str, criteria: str) -> dict:
    user_prompt = f"--- FICHE DE POSTE (AO) ---\n{ao_text[:3000]}\n\n--- CRITÈRES CLÉS ---\n{criteria}\n\n--- CV CANDIDAT ---\n{cv_text[:3500]}"
    safe_data = deepcopy(DEFAULT_DATA)
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": AUDITOR_PROMPT}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}, temperature=0.0
        )
        ai_json = json.loads(res.choices[0].message.content)
        # Fusion des données
        for key, value in ai_json.items():
            if key in safe_data and isinstance(safe_data[key], dict) and isinstance(value, dict):
                safe_data[key].update(value)
            else: safe_data[key] = value
        return safe_data
    except Exception as e:
        print(f"Err IA: {e}")
        return safe_data

# -----------------------------
# 4. INTERFACE UTILISATEUR
# -----------------------------
st.title("🎯 AI Recruiter PRO — Precision Edition")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Historique")
    if st.button("🔄 Rafraîchir"): st.rerun()
    try:
        hist = supabase.table('search_history').select("*").order('created_at', desc=True).limit(5).execute()
        for h in hist.data:
            st.markdown(f"**{h['query_text'][:25]}...** ({h['results_count']} profils)")
    except: st.caption("Rien pour l'instant.")

# --- TABS ---
tab_search, tab_ingest = st.tabs(["🔎 RECHERCHE (FICHE DE POSTE)", "📥 INGESTION CV"])

# --- ONGLET 1 : RECHERCHE AVANCEE (PDF AO) ---
with tab_search:
    col_upload, col_criteria = st.columns([1, 1])
    
    ao_content = ""
    
    with col_upload:
        st.subheader("1. L'Offre (AO)")
        ao_pdf = st.file_uploader("📄 Charger la Fiche de Poste (PDF)", type="pdf")
        ao_manual = st.text_area("Ou tapez le besoin ici", height=100, placeholder="Ex: Chef de projet senior...")
        
        # Logique de priorité : PDF > Texte
        if ao_pdf:
            txt = extract_pdf_safe(ao_pdf.read())
            if txt:
                ao_content = txt
                st.success("✅ Fiche de poste lue avec succès !")
        elif ao_manual:
            ao_content = ao_manual

    with col_criteria:
        st.subheader("2. Critères & Filtres")
        criteria = st.text_area("Dealbreakers (Points bloquants)", height=100, placeholder="Ex: Anglais C1 impératif, Pas de freelance...")
        threshold = st.slider("Largeur du filet (Matching)", 0.3, 0.8, 0.45)
        limit = st.number_input("Nombre de CVs à auditer", 5, 50, 10)
    
    st.divider()
    launch = st.button("🚀 LANCER L'ANALYSE PRÉCISE", type="primary", use_container_width=True)

    if launch:
        if not ao_content:
            st.error("⚠️ Veuillez fournir une fiche de poste (PDF ou Texte).")
        else:
            with st.status("🧠 Analyse Matricielle en cours...", expanded=True) as status:
                
                # 1. Vector Search
                status.write("📐 Vectorisation de l'AO...")
                q_vec = get_embedding(ao_content[:8000]) # On vectorise l'AO complet
                
                status.write("🗄️ Recherche des profils compatibles...")
                res_db = supabase.rpc('match_candidates', {'query_embedding': q_vec, 'match_threshold': threshold, 'match_count': limit}).execute()
                cands = res_db.data
                count = len(cands)
                save_search_history(ao_content[:50], criteria, count)

                if not cands:
                    status.update(label="❌ Aucun candidat pertinent trouvé.", state="error")
                else:
                    status.write(f"✅ {count} profils trouvés. Calcul du Score Pondéré...")
                    
                    final_results = []
                    bar = st.progress(0)
                    
                    for i, c in enumerate(cands):
                        # On envoie l'AO complet et le CV complet à l'IA
                        audit = audit_candidate_groq(ao_content, c['contenu_texte'], criteria)
                        
                        # Gestion Nom
                        infos = audit.get('infos', {})
                        if not infos.get('nom') or infos.get('nom') == "Candidat Inconnu":
                            if 'infos' not in audit: audit['infos'] = {}
                            audit['infos']['nom'] = c.get('nom_fichier', 'Dossier')

                        final_results.append(audit)
                        bar.progress((i+1)/count)
                    
                    status.update(label="🎉 Analyse terminée !", state="complete")
                    
                    # Tri par Score Global
                    final_results.sort(key=lambda x: x.get('scores', {}).get('global', 0), reverse=True)
                    
                    st.subheader("Résultats de l'Analyse")
                    
                    for r in final_results:
                        scores = r.get('scores', {})
                        infos = r.get('infos', {})
                        sc = scores.get('global', 0)
                        
                        # Couleurs
                        s_cls = "sc-good" if sc >= 70 else "sc-mid" if sc >= 50 else "sc-bad"
                        
                        with st.expander(f"{infos.get('nom')} — Score {sc}/100", expanded=(sc>=60)):
                            
                            c1, c2 = st.columns([4, 1])
                            with c1:
                                st.markdown(f"<div class='name-title'>{infos.get('nom')}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='job-subtitle'>{infos.get('poste_actuel','')}</div>", unsafe_allow_html=True)
                                
                                # Sous-Scores
                                col_s1, col_s2, col_s3 = st.columns(3)
                                col_s1.metric("Tech / Hard Skills", f"{scores.get('tech',0)}/40")
                                col_s2.metric("Expérience", f"{scores.get('experience',0)}/30")
                                col_s3.metric("Fit / Soft Skills", f"{scores.get('fit',0)}/30")
                                
                                st.info(f"💡 {r['analyse'].get('verdict_auditeur', '...')}")
                                
                                # Manquants Critiques
                                manquants = r.get('competences', {}).get('manquant_critique', [])
                                if manquants: st.error(f"⚠️ Manque: {', '.join(manquants)}")

                            with c2:
                                st.markdown(f"<div class='score-badge {s_cls}'>{sc}</div>", unsafe_allow_html=True)

                            st.divider()
                            
                            # Détails Compétences
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown("**✅ Points Forts**")
                                matches = r.get('competences', {}).get('match_details', [])
                                if matches:
                                    for m in matches:
                                        if isinstance(m, dict): s, p = m.get('skill',''), m.get('preuve','')
                                        else: s, p = m.skill, m.preuve
                                        st.markdown(f"- **{s}**: *{p}*")
                            with cols[1]:
                                st.markdown("**❌ Points Faibles**")
                                secs = r.get('competences', {}).get('manquant_secondaire', [])
                                if secs: st.markdown(", ".join(secs))
                                
                            # Historique rapide
                            st.divider()
                            st.caption("Extrait Parcours:")
                            hist = r.get('historique', [])
                            if hist:
                                for h in hist[:2]:
                                    if isinstance(h, dict): t, e = h.get('titre',''), h.get('entreprise','')
                                    else: t, e = h.titre, h.entreprise
                                    st.markdown(f"• {t} chez {e}")

# --- ONGLET 2 : INGESTION ---
with tab_ingest:
    st.header("Alimenter la CV-Thèque")
    files = st.file_uploader("PDFs Candidats", type="pdf", accept_multiple_files=True)
    if st.button("Indexer") and files:
        bar = st.progress(0)
        for i, f in enumerate(files):
            try:
                txt = extract_pdf_safe(f.read())
                if len(txt) > 50: ingest_cv_to_db(f, txt)
            except: pass
            bar.progress((i+1)/len(files))
        st.success("Terminé !")
