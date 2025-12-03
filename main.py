# AI Recruiter PRO — v22.0 (Stable Fix + Prompt Punitif + UI Premium)
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
st.set_page_config(page_title="AI Recruiter PRO v22", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    :root {
        --primary:#2563eb; --bg-app:#f8fafc; --text-main:#0f172a; --border:#cbd5e1;
        --score-good:#16a34a; --score-mid:#d97706; --score-bad:#dc2626;
    }
    .stApp { background: var(--bg-app); color: var(--text-main); font-family: 'Inter', sans-serif; }
    
    /* INPUTS SEARCH ENGINE */
    .stTextInput input { font-size: 1.1rem; padding: 12px; border-radius: 12px; border: 2px solid #e2e8f0; }
    .stButton button { border-radius: 12px; font-weight: 700; height: 50px; }

    /* TYPOGRAPHY CARD */
    .name-title { font-size: 1.5rem; font-weight: 800; color: #1e293b; margin: 0; }
    .job-subtitle { font-size: 0.95rem; color: #64748b; margin-top: 4px; font-weight: 500; }
    .section-header { font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin-bottom: 10px; letter-spacing: 0.5px; margin-top: 10px;}

    /* SCORE BADGE */
    .score-badge { 
        font-size: 1.8rem; font-weight: 900; color: white; 
        width: 70px; height: 70px; border-radius: 16px; 
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .sc-good { background: linear-gradient(135deg, #16a34a, #15803d); }
    .sc-mid { background: linear-gradient(135deg, #d97706, #b45309); }
    .sc-bad { background: linear-gradient(135deg, #dc2626, #b91c1c); }

    /* EVIDENCE BOXES */
    .evidence-box { background: #f8fafc; border-left: 4px solid #cbd5e1; padding: 12px 15px; margin-bottom: 10px; border-radius: 0 6px 6px 0; }
    .ev-skill { font-weight: 700; color: #334155; font-size: 0.95rem; }
    .ev-proof { font-size: 0.9rem; color: #475569; font-style: italic; margin-top: 4px; }
    .ev-missing { border-left-color: #ef4444; background: #fef2f2; }
    .ev-missing .ev-skill { color: #991b1b; }

    /* TAGS */
    .tag { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin-right: 6px; margin-bottom: 6px; }
    .tag-blue { background: #eff6ff; color: #1d4ed8; border: 1px solid #dbeafe; }
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
        if len(clean) < 50: return "ERREUR: PDF Illisible ou Vide."
        return clean
    except: return ""

def get_embedding(text: str) -> List[float]:
    """Vectorisation avec Retry"""
    text = text.replace("\n", " ")
    for attempt in range(3):
        try:
            return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding
        except openai.RateLimitError:
            time.sleep((attempt + 1) * 2)
        except Exception:
            break
    st.error("❌ Erreur OpenAI. Vérifiez vos crédits (Billing).")
    st.stop()

def ingest_cv_to_db(file, text):
    vector = get_embedding(text)
    supabase.table('candidates').insert({
        "nom_fichier": file.name, "contenu_texte": text, "embedding": vector
    }).execute()

def save_search_history(query, criteria, count):
    try:
        supabase.table('search_history').insert({
            "query_text": query, "criteria_used": criteria, "results_count": count
        }).execute()
    except: pass

# --- PROMPT PUNITIF (DEMANDÉ) ---
AUDITOR_PROMPT = """
ROLE: Auditeur de Recrutement Impitoyable (Sanction Immédiate).
TACHE: Vérifier factuellement l'adéquation CV vs OFFRE.
PRINCIPE: "Pas écrit = Pas acquis".

RÈGLES DE SCORING PUNITIF (PRIORITÉ ABSOLUE):
1. IDENTIFICATION DES DEALBREAKERS :
   - Regarde la section "CRITERES IMPERATIFS" fournie.
   - Si le CV ne mentionne pas EXPLICITEMENT un de ces critères (ex: "Anglais courant", "Expérience 5 ans", "Python"), c'est un MANQUE CRITIQUE.

2. CALCUL DU SCORE GLOBAL (0-100) :
   - Si UN SEUL manquant critique est détecté : Le score GLOBAL est plafonné à 40/100 MAXIMUM. (C'est éliminatoire).
   - Si TOUS les critiques sont présents :
     * Départ à 100.
     * -10 points par compétence secondaire manquante.
     * -15 points si l'expérience est trop courte.
     * -15 points pour des "Red Flags" (instabilité, trous).

3. PREUVES OBLIGATOIRES :
   - Tu ne peux valider une compétence QUE si tu peux citer le CV. Sinon, c'est un manquant.

STRUCTURE JSON REQUISE :
{
    "infos": { "nom": "Nom complet", "email": "...", "tel": "...", "ville": "...", "linkedin": "...", "poste_actuel": "..." },
    "scores": { "global": int, "tech": int (0-10), "experience": int (0-10), "fit": int (0-10) },
    "competences": {
        "match_details": [ {"skill": "Nom Skill", "preuve": "Citation du CV prouvant la skill", "niveau": "Expert/Confirmé/Junior"} ],
        "manquant_critique": ["LISTE DES DEALBREAKERS MANQUANTS ICI"],
        "manquant_secondaire": ["Skill C"]
    },
    "analyse": {
        "verdict_auditeur": "Phrase tranchante. Si score < 40, commence par 'DISQUALIFIÉ : ...'.",
        "red_flags": ["Flag 1", "Flag 2"]
    },
    "historique": [ {"titre": "...", "entreprise": "...", "duree": "...", "contexte": "Secteur/Taille"} ],
    "entretien": [ {"cible": "Lacune identifiée", "question": "Question piège pour vérifier", "reponse_attendue": "..."} ]
}
"""

def audit_candidate_groq(query: str, cv: str, criteria: str) -> dict:
    user_prompt = f"--- OFFRE ---\n{query}\n\n--- CRITERES IMPERATIFS ---\n{criteria}\n\n--- CV ---\n{cv[:3500]}"
    
    # Structure de secours
    safe_data = deepcopy(DEFAULT_DATA)

    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": AUDITOR_PROMPT}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"}, 
            temperature=0.0
        )
        
        ai_json = json.loads(res.choices[0].message.content)
        
        # Fusion sécurisée (Safe Merge)
        for key, value in ai_json.items():
            if key in safe_data and isinstance(safe_data[key], dict) and isinstance(value, dict):
                safe_data[key].update(value)
            else:
                safe_data[key] = value
                
        return safe_data
    except Exception as e:
        print(f"Err IA: {e}")
        return safe_data

# -----------------------------
# 4. INTERFACE UTILISATEUR
# -----------------------------
st.title("🛡️ AI Recruiter PRO — Punitif Edition")

# --- SIDEBAR HISTORIQUE ---
with st.sidebar:
    st.header("🗂️ Derniers A.O.")
    if st.button("🔄 Rafraîchir"): st.rerun()
    try:
        hist = supabase.table('search_history').select("*").order('created_at', desc=True).limit(6).execute()
        for h in hist.data:
            st.markdown(f"""
            <div style="padding:12px; background:white; border-radius:8px; border:1px solid #cbd5e1; margin-bottom:8px;">
                <div style="font-weight:600; font-size:0.9rem; color:#1e293b;">{h['query_text'][:30]}...</div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">🎯 {h['results_count']} profils • {h['created_at'][:10]}</div>
            </div>""", unsafe_allow_html=True)
    except: st.caption("Historique vide.")

# --- MAIN TABS ---
tab_search, tab_ingest = st.tabs(["🔎 RECHERCHE & ANALYSE", "📥 INGESTION CV"])

# --- ONGLET RECHERCHE ---
with tab_search:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_bar, col_go = st.columns([6, 1])
    with col_bar:
        search_query = st.text_input("Votre besoin en langage naturel :", placeholder="Ex: Je cherche un Chef de Projet BTP avec 5 ans d'expérience...")
    with col_go:
        st.write("")
        st.write("")
        launch = st.button("LANCER", type="primary", use_container_width=True)

    with st.expander("⚙️ Critères & Options"):
        criteria = st.text_area("Dealbreakers (Critères éliminatoires)", height=70)
        threshold = st.slider("Précision Sémantique", 0.3, 0.8, 0.45)
        limit = st.number_input("Max Profils", 5, 50, 10)

    if launch and search_query:
        st.divider()
        with st.status("🧠 Analyse en cours...", expanded=True) as status:
            
            # 1. Vector Search
            status.write("📐 Vectorisation & Matching...")
            q_vec = get_embedding(search_query)
            res_db = supabase.rpc('match_candidates', {'query_embedding': q_vec, 'match_threshold': threshold, 'match_count': limit}).execute()
            cands = res_db.data
            count = len(cands)
            save_search_history(search_query, criteria, count)
            
            if not cands:
                status.update(label="❌ Aucun candidat trouvé.", state="error")
            else:
                status.write(f"✅ {count} profils identifiés. Audit PUNITIF en cours...")
                
                # 2. Audit IA
                final_results = []
                bar = st.progress(0)
                for i, c in enumerate(cands):
                    audit = audit_candidate_groq(search_query, c['contenu_texte'], criteria)
                    
                    # Sécurisation Nom
                    infos = audit.get('infos', {})
                    if not infos.get('nom') or infos.get('nom') == "Candidat Inconnu":
                        if 'infos' not in audit: audit['infos'] = {}
                        audit['infos']['nom'] = c['nom_fichier']
                        
                    final_results.append(audit)
                    bar.progress((i+1)/count)
                
                status.update(label="🎉 Terminé !", state="complete")
                
                # Tri Sécurisé
                final_results.sort(key=lambda x: x.get('scores', {}).get('global', 0), reverse=True)
                
                # --- AFFICHAGE PREMIUM BLINDÉ ---
                st.subheader(f"Résultats de l'AO : {search_query}")
                
                for r in final_results:
                    # Extraction sécurisée (Fix TypeError)
                    scores = r.get('scores', {})
                    infos = r.get('infos', {})
                    analyse = r.get('analyse', {})
                    competences = r.get('competences', {})
                    historique = r.get('historique', [])
                    entretien = r.get('entretien', [])

                    sc = scores.get('global', 0)
                    s_cls = "sc-good" if sc >= 70 else "sc-mid" if sc >= 50 else "sc-bad"
                    nom_cand = infos.get('nom', 'Inconnu')
                    
                    with st.expander(f"{nom_cand} — Score {sc}/100", expanded=(sc>=60)):
                        
                        # EN-TÊTE
                        c_main, c_badge = st.columns([4, 1])
                        with c_main:
                            st.markdown(f"<div class='name-title'>{nom_cand}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='job-subtitle'>{infos.get('poste_actuel','')} • {infos.get('ville','')}</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style='margin-top:10px;'>
                                <span class='tag tag-blue'>✉️ {infos.get('email','N/A')}</span>
                                <span class='tag tag-blue'>📱 {infos.get('tel','N/A')}</span>
                                <span class='tag tag-blue'><a href='{infos.get('linkedin','#')}' target='_blank'>LinkedIn</a></span>
                            </div>""", unsafe_allow_html=True)

                            # Verdict & Alertes
                            red_flags = analyse.get('red_flags', [])
                            if red_flags:
                                for flag in red_flags: st.error(f"🚩 {flag}")
                            
                            manquants = competences.get('manquant_critique', [])
                            if manquants:
                                st.error(f"⛔ **DISQUALIFIÉ :** Manque de {', '.join(manquants)}")
                            
                            st.info(f"💡 **Avis Auditeur :** {analyse.get('verdict_auditeur', '...')}")

                        with c_badge:
                            st.markdown(f"<div class='score-badge {s_cls}'>{sc}</div>", unsafe_allow_html=True)
                            st.caption("Score Fiabilité")

                        st.divider()

                        # COLONNES PREUVES vs MANQUES
                        col_match, col_miss = st.columns(2)
                        with col_match:
                            st.markdown("<div class='section-header'>✅ Compétences Prouvées</div>", unsafe_allow_html=True)
                            match_details = competences.get('match_details', [])
                            if match_details:
                                for item in match_details:
                                    if isinstance(item, dict):
                                        s, n, p = item.get('skill',''), item.get('niveau',''), item.get('preuve','')
                                    else: s, n, p = item.skill, item.niveau, item.preuve # Fallback
                                        
                                    st.markdown(f"""
                                    <div class='evidence-box'>
                                        <div class='ev-skill'>{s} <span style='font-weight:400; color:#64748b;'>({n})</span></div>
                                        <div class='ev-proof'>"{p}"</div>
                                    </div>""", unsafe_allow_html=True)
                            else: st.caption("Aucune preuve solide.")

                        with col_miss:
                            st.markdown("<div class='section-header'>❌ Points Manquants</div>", unsafe_allow_html=True)
                            if manquants:
                                for m in manquants:
                                    st.markdown(f"""
                                    <div class='evidence-box ev-missing'>
                                        <div class='ev-skill'>CRITIQUE : {m}</div>
                                        <div class='ev-proof'>Absence totale détectée.</div>
                                    </div>""", unsafe_allow_html=True)
                            
                            sec = competences.get('manquant_secondaire', [])
                            if sec:
                                st.markdown("**Secondaires :** " + ", ".join([f"<span style='color:#64748b'>{x}</span>" for x in sec]), unsafe_allow_html=True)

                        st.divider()

                        # HISTORIQUE
                        c_hist, c_quest = st.columns(2)
                        with c_hist:
                            st.markdown("<div class='section-header'>📅 Parcours</div>", unsafe_allow_html=True)
                            if historique:
                                for h in historique[:3]:
                                    if isinstance(h, dict): t, e, d = h.get('titre',''), h.get('entreprise',''), h.get('duree','')
                                    else: t, e, d = h.titre, h.entreprise, h.duree
                                    st.markdown(f"**{t}** chez *{e}*")
                                    st.caption(f"{d}")
                        
                        with c_quest:
                            st.markdown("<div class='section-header'>🎤 Questions Entretien</div>", unsafe_allow_html=True)
                            if entretien:
                                for q in entretien:
                                    if isinstance(q, dict): qu, re = q.get('question',''), q.get('reponse_attendue','')
                                    else: qu, re = q.question, q.reponse_attendue
                                    with st.expander(f"❓ Question"):
                                        st.write(f"**Q:** {qu}")
                                        st.caption(f"💡 Attendu : {re}")

# --- ONGLET INGESTION ---
with tab_ingest:
    st.header("Alimenter la CV-Thèque")
    files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Indexer") and files:
        bar = st.progress(0)
        for i, f in enumerate(files):
            try:
                txt = extract_pdf_safe(f.read())
                if len(txt) > 50: ingest_cv_to_db(f, txt)
            except: pass
            bar.progress((i+1)/len(files))
        st.success("Terminé !")
