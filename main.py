# AI Recruiter PRO — v31.0 (GESTION BDD + EDIT + DELETE)
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
st.set_page_config(page_title="AI Recruiter PRO v31", layout="wide", page_icon="💎")

st.markdown("""
<style>
    :root {
        --primary:#2563eb; --bg-app:#f8fafc; --text-main:#0f172a; --border:#cbd5e1;
        --score-good:#16a34a; --score-mid:#d97706; --score-bad:#dc2626;
    }
    .stApp { background: var(--bg-app); color: var(--text-main); font-family: 'Inter', sans-serif; }
    
    /* INPUTS */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] { border-radius: 12px; border: 2px solid #e2e8f0; }
    .stButton button { border-radius: 12px; font-weight: 700; height: 50px; }

    /* TYPOGRAPHY CARD */
    .name-title { font-size: 1.6rem; font-weight: 800; color: #1e293b; margin: 0; line-height: 1.2; }
    .job-subtitle { font-size: 0.95rem; color: #64748b; margin-top: 4px; font-weight: 500; }
    .section-header { font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin-bottom: 10px; letter-spacing: 0.5px; margin-top: 15px;}

    /* SCORE BADGE */
    .score-badge { 
        font-size: 2rem; font-weight: 900; color: white; 
        width: 80px; height: 80px; border-radius: 16px; 
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px -1px rgba(0, 0, 0, 0.2);
    }
    .sc-good { background: linear-gradient(135deg, #16a34a, #15803d); }
    .sc-mid { background: linear-gradient(135deg, #d97706, #b45309); }
    .sc-bad { background: linear-gradient(135deg, #dc2626, #991b1b); }

    /* EVIDENCE BOXES */
    .evidence-box { background: #f8fafc; border-left: 4px solid #cbd5e1; padding: 12px 15px; margin-bottom: 8px; border-radius: 0 8px 8px 0; }
    .ev-skill { font-weight: 700; color: #334155; font-size: 0.95rem; }
    .ev-proof { font-size: 0.9rem; color: #475569; font-style: italic; margin-top: 4px; }
    .ev-missing { border-left-color: #ef4444; background: #fff1f2; }
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

# --- NETTOYEUR JSON ---
def clean_json_string(text: str) -> str:
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end != -1: return text[start:end]
    return text

def safe_json_loads(text: str) -> Dict:
    cleaned = clean_json_string(text)
    try: return json.loads(cleaned)
    except: return None

# --- GESTION BDD ---
def fetch_all_candidates():
    # On ne récupère pas le contenu_texte ni l'embedding pour aller vite
    try:
        res = supabase.table('candidates').select("id, nom_fichier, created_at").order("created_at", desc=True).execute()
        return res.data
    except: return []

def update_candidate_name(id_cand, new_name):
    try:
        supabase.table('candidates').update({"nom_fichier": new_name}).eq("id", id_cand).execute()
        st.toast("✅ Nom mis à jour !")
        time.sleep(1)
        st.rerun()
    except Exception as e: st.error(f"Erreur update: {e}")

def delete_candidate(id_cand):
    try:
        supabase.table('candidates').delete().eq("id", id_cand).execute()
        st.toast("🗑️ CV supprimé !")
        time.sleep(1)
        st.rerun()
    except Exception as e: st.error(f"Erreur delete: {e}")

# --- PROMPT V29 ---
AUDITOR_PROMPT = """
Tu es un recruteur expert.
TACHE: Analyser CV vs AO.
IMPORTANT: REPONDS UNIQUEMENT AVEC LE JSON.

SCORING RULES (0-100):
- Dealbreaker manquant = Score < 30.
- Excellent = > 80.

STRUCTURE JSON:
{
    "infos": { "nom": "...", "poste_actuel": "...", "email": "...", "tel": "...", "ville": "...", "linkedin": "..." },
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
    user_prompt = f"AO: {ao_text[:1000]}... CRITERES: {criteria}... CV: {cv_text[:1500]}..."
    safe_data = deepcopy(DEFAULT_DATA)
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", # Eco Model
            messages=[{"role": "system", "content": AUDITOR_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"} 
        )
        raw_content = res.choices[0].message.content
        safe_data['raw_response'] = raw_content
        
        ai_json = safe_json_loads(raw_content)
        if ai_json:
            for key, value in ai_json.items():
                if key in safe_data and isinstance(safe_data[key], dict) and isinstance(value, dict):
                    safe_data[key].update(value)
                else: safe_data[key] = value
        else: safe_data['analyse']['verdict_auditeur'] = "Erreur JSON (Voir Debug)"
        return safe_data
    except Exception as e:
        safe_data['analyse']['verdict_auditeur'] = f"Erreur API: {str(e)}"
        return safe_data

# -----------------------------
# 4. INTERFACE
# -----------------------------
st.title("💎 AI Recruiter PRO — Management")

# --- TABS ---
tab_search, tab_ingest, tab_manage = st.tabs(["🔎 RECHERCHE & ANALYSE", "📥 INGESTION CV", "🗄️ GESTION BDD"])

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
        st.subheader("2. Paramètres")
        criteria = st.text_area("Dealbreakers (Points Bloquants)", height=100)
        threshold = st.slider("Seuil Matching", 0.3, 0.8, 0.45)
        limit = st.number_input("Nb Profils", 1, 20, 5)
    
    st.divider()
    
    if st.button("🚀 LANCER L'ANALYSE", type="primary"):
        if not ao_content:
            st.error("⚠️ Texte de l'offre vide.")
        else:
            with st.status("Traitement en cours...", expanded=True) as status:
                q_vec = get_embedding(ao_content[:8000])
                res_db = supabase.rpc('match_candidates', {'query_embedding': q_vec, 'match_threshold': threshold, 'match_count': limit}).execute()
                cands = res_db.data
                
                if not cands:
                    status.update(label="❌ 0 Candidat trouvé", state="error")
                else:
                    status.write(f"✅ {len(cands)} profils identifiés. Audit IA (Eco-Mode)...")
                    final_results = []
                    bar = st.progress(0)
                    
                    for i, c in enumerate(cands):
                        audit = audit_candidate_groq(ao_content, c['contenu_texte'], criteria)
                        
                        infos = audit.get('infos', {})
                        if not infos.get('nom') or infos.get('nom') == "Candidat Inconnu":
                            if 'infos' not in audit: audit['infos'] = {}
                            audit['infos']['nom'] = c.get('nom_fichier', 'Fichier Sans Nom')
                            
                        final_results.append(audit)
                        bar.progress((i+1)/len(cands))
                    
                    status.update(label="Analyse Terminée", state="complete")
                    final_results.sort(key=lambda x: x.get('scores', {}).get('global', 0), reverse=True)
                    
                    st.subheader(f"Résultats ({len(final_results)})")
                    
                    # --- AFFICHAGE STYLE V17 (PREMIUM) ---
                    for i, r in enumerate(final_results):
                        sc = r.get('scores', {}).get('global', 0)
                        infos = r.get('infos', {})
                        analyse = r.get('analyse', {})
                        competences = r.get('competences', {})
                        historique = r.get('historique', [])
                        entretien = r.get('entretien', [])
                        nom = infos.get('nom', 'Inconnu')
                        
                        # Code couleur
                        s_cls = "sc-good" if sc >= 70 else "sc-mid" if sc >= 40 else "sc-bad"
                        
                        with st.expander(f"{nom} — Score {sc}/100", expanded=(sc>=60)):
                            
                            # EN-TÊTE RICHE (Style V17)
                            c_main, c_badge = st.columns([4, 1])
                            with c_main:
                                st.markdown(f"<div class='name-title'>{nom}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='job-subtitle'>{infos.get('poste_actuel', '')} • {infos.get('ville', '')}</div>", unsafe_allow_html=True)
                                
                                # Tags
                                st.markdown(f"""
                                <div style='margin-top:10px;'>
                                    <span class='tag tag-blue'>✉️ {infos.get('email', 'N/A')}</span>
                                    <span class='tag tag-blue'>📱 {infos.get('tel', 'N/A')}</span>
                                    <span class='tag tag-blue'><a href='{infos.get('linkedin', '#')}' target='_blank'>LinkedIn</a></span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Verdict & Red Flags
                                red_flags = analyse.get('red_flags', [])
                                if red_flags:
                                    for flag in red_flags: st.error(f"🚩 {flag}")
                                
                                manquants = competences.get('manquant_critique', [])
                                if manquants: st.error(f"⛔ **Manquants :** {', '.join(manquants)}")
                                
                                st.info(f"💡 **Avis:** {analyse.get('verdict_auditeur', '...')}")

                            with c_badge:
                                st.markdown(f"<div class='score-badge {s_cls}'>{sc}</div>", unsafe_allow_html=True)
                                st.caption("Score Fiabilité")

                            st.divider()

                            # COLONNES PREUVES (Style V17)
                            col_match, col_miss = st.columns(2)
                            with col_match:
                                st.markdown("<div class='section-header'>✅ Points Forts Validés</div>", unsafe_allow_html=True)
                                match_details = competences.get('match_details', [])
                                if match_details:
                                    for item in match_details:
                                        if isinstance(item, dict): s, n, p = item.get('skill',''), item.get('niveau',''), item.get('preuve','')
                                        else: s, n, p = item.skill, item.niveau, item.preuve
                                        st.markdown(f"""
                                        <div class='evidence-box'>
                                            <div class='ev-skill'>{s} <span style='font-weight:400; color:#64748b;'>({n})</span></div>
                                            <div class='ev-proof'>"{p}"</div>
                                        </div>""", unsafe_allow_html=True)
                                else: st.caption("Rien de notable.")

                            with col_miss:
                                st.markdown("<div class='section-header'>❌ Lacunes & Risques</div>", unsafe_allow_html=True)
                                if manquants:
                                    for m in manquants:
                                        st.markdown(f"""
                                        <div class='evidence-box ev-missing'>
                                            <div class='ev-skill' style='color:#b91c1c;'>CRITIQUE : {m}</div>
                                            <div class='ev-proof'>Absence totale détectée.</div>
                                        </div>""", unsafe_allow_html=True)
                                
                                secs = competences.get('manquant_secondaire', [])
                                if secs: st.markdown("**Secondaires :** " + ", ".join([f"<span style='color:#64748b'>{x}</span>" for x in secs]), unsafe_allow_html=True)

                            st.divider()

                            # HISTORIQUE & QUESTIONS
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
                                        with st.expander("❓ Question suggérée"):
                                            st.write(f"**Q:** {qu}")
                                            st.caption(f"💡 Attendu : {re}")
                            
                            # DEBUG SAFE ZONE
                            if "Erreur" in str(analyse.get('verdict_auditeur', '')):
                                st.divider()
                                st.warning("Debug JSON (L'IA a eu du mal)")
                                st.text_area("Raw", r.get('raw_response', ''), height=100, key=f"debug_{i}")

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

# --- ONGLET 3 : GESTION BDD ---
with tab_manage:
    st.header("🗄️ Gestion de la Base de Données")
    
    # 1. Charger la liste
    candidates = fetch_all_candidates()
    
    if not candidates:
        st.info("La base est vide pour l'instant.")
    else:
        # Création d'un DataFrame pour affichage propre
        df = pd.DataFrame(candidates)
        df['Date'] = pd.to_datetime(df['created_at']).dt.strftime('%d/%m/%Y %H:%M')
        
        st.dataframe(df[['nom_fichier', 'Date', 'id']], use_container_width=True)
        
        st.divider()
        st.subheader("🛠️ Actions sur un candidat")
        
        # Sélecteur
        # On crée une liste de tuples (Label affiché, ID réel)
        options = {f"{c['nom_fichier']} ({c['created_at'][:10]})": c['id'] for c in candidates}
        selected_label = st.selectbox("Sélectionnez un candidat à modifier/supprimer :", list(options.keys()))
        selected_id = options[selected_label]
        
        col_edit, col_del = st.columns([3, 1])
        
        with col_edit:
            new_name = st.text_input("Nouveau nom du fichier/profil :", value=selected_label.split(" (")[0])
            if st.button("💾 Renommer le candidat"):
                update_candidate_name(selected_id, new_name)
        
        with col_del:
            st.write("") # Spacer
            st.write("") 
            if st.button("🗑️ Supprimer définitivement", type="primary"):
                delete_candidate(selected_id)
