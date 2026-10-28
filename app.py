import re
from datetime import date

import pandas as pd
import streamlit as st
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None


st.set_page_config(page_title="SkillBridge AI", page_icon="🌉", layout="centered")

# ------------------------------- STYLE -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Manrope', sans-serif; }
    .stApp { background: radial-gradient(ellipse at top, #2A1757 0%, #170C35 45%, #0F0724 100%); }
    .block-container { max-width: 820px; padding-top: 2.5rem; padding-bottom: 4rem; }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

    .brand { font-family: 'Cormorant Garamond', serif; font-weight: 700; font-size: 1.35rem; color: #D4AF37; margin-bottom: .8rem; }
    .title { font-family: 'Cormorant Garamond', serif; font-weight: 700; font-size: 3.1rem; line-height: 1.05; color: #F5EFE0; margin: 0 0 .7rem 0; }
    .rule { height: 2px; width: 90px; background: linear-gradient(90deg, #D4AF37, rgba(212,175,55,0)); margin: 0 0 1rem 0; }
    .sub { color: #BDB3DB; font-size: 1.05rem; max-width: 560px; margin-bottom: 1.8rem; line-height: 1.6; }

    h2, h3 { font-family: 'Cormorant Garamond', serif !important; color: #F5EFE0 !important; font-weight: 700 !important; }
    h3 { font-size: 1.55rem !important; margin-top: 1.3rem !important; padding-bottom: .2rem; border-bottom: 1px solid rgba(212,175,55,.28); }
    h2 { font-size: 2.1rem !important; }
    .hint { color: #BDB3DB; font-size: .9rem; margin: -.4rem 0 .7rem 0; }

    div[data-testid="stForm"] { border: 1px solid rgba(212,175,55,.35); border-radius: 6px; padding: 1.8rem 1.8rem 1.3rem;
        background: rgba(35,21,71,.72); box-shadow: 0 18px 50px rgba(0,0,0,.35); }
    .stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div { border-color: rgba(212,175,55,.25) !important; }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button { border-radius: 4px; font-weight: 700; padding: .7rem 1.3rem; }
    .stFormSubmitButton > button { background: linear-gradient(180deg, #E3C25A, #C49B22); color: #1A0F3A; border: none; width: 100%; font-size: 1.02rem; }
    .stFormSubmitButton > button:hover { background: linear-gradient(180deg, #EDCF70, #D4AF37); color: #1A0F3A; }
    .stDownloadButton > button { background: transparent; color: #D4AF37; border: 1px solid #D4AF37; }
    .stDownloadButton > button:hover { background: rgba(212,175,55,.12); color: #E3C25A; border-color: #E3C25A; }

    hr { border-color: rgba(212,175,55,.3) !important; }
    button[data-baseweb="tab"] { font-weight: 600; }

    .chip { display: inline-block; padding: .25rem .8rem; margin: .15rem .25rem .15rem 0; border-radius: 4px;
            font-size: .85rem; font-weight: 600; }
    .chip.ok { background: rgba(212,175,55,.16); color: #EBCB6B; border: 1px solid rgba(212,175,55,.5); }
    .chip.gap { background: rgba(190,90,140,.18); color: #F2B6D3; border: 1px solid rgba(220,120,170,.5); }
    .chip.neutral { background: rgba(255,255,255,.07); color: #CFC6E8; border: 1px solid rgba(255,255,255,.14); }

    .score-wrap { display: flex; align-items: baseline; gap: 1rem; margin: .4rem 0 .4rem; }
    .score { font-family: 'Cormorant Garamond', serif; font-size: 4.6rem; font-weight: 700; color: #D4AF37; line-height: 1; }
    .score-label { font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; font-weight: 600; color: #F5EFE0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- DATA -------------------------------
SKILL_GROUPS = {
    "Programming": ["python", "java", "javascript", "typescript", "c++", "c#", "c programming", "golang", "rust", "php", "ruby", "kotlin", "swift"],
    "Data & AI": ["machine learning", "deep learning", "artificial intelligence", "data science", "data analysis", "data visualization", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision", "generative ai", "llm", "statistics", "power bi", "tableau", "excel", "sql"],
    "Web": ["html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "streamlit", "rest api", "graphql"],
    "Cloud & DevOps": ["aws", "azure", "google cloud", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "linux", "git", "github", "serverless", "networking"],
    "Databases": ["mysql", "postgresql", "mongodb", "redis", "sqlite", "database design", "data modeling"],
    "Security": ["cybersecurity", "network security", "penetration testing", "ethical hacking", "cryptography", "owasp", "siem"],
    "Core & Professional": ["data structures", "algorithms", "object-oriented programming", "problem solving", "communication", "teamwork", "leadership", "agile", "scrum", "testing", "unit testing", "debugging"],
}
ALL_SKILLS = sorted({s for g in SKILL_GROUPS.values() for s in g}, key=len, reverse=True)

ALIASES = {
    "js": "javascript", "ts": "typescript", "ml": "machine learning", "ai": "artificial intelligence",
    "sklearn": "scikit-learn", "node": "node.js", "nodejs": "node.js", "postgres": "postgresql",
    "powerbi": "power bi", "gcp": "google cloud", "oop": "object-oriented programming",
    "golang": "golang", "cpp": "c++",
}

ROLES = {
    "Data Analyst": {
        "skills": ["sql", "excel", "python", "pandas", "statistics", "power bi", "data visualization", "communication"],
        "projects": ["Build an interactive sales dashboard", "Analyze a public dataset and write up your findings", "Solve a business case study using SQL"],
        "certs": ["Microsoft Power BI Data Analyst", "Google Data Analytics Certificate"],
    },
    "Machine Learning Engineer": {
        "skills": ["python", "machine learning", "statistics", "scikit-learn", "pandas", "numpy", "deep learning", "git", "rest api"],
        "projects": ["Train and evaluate a classification model", "Deploy a model behind a REST API", "Build a reproducible end-to-end ML pipeline"],
        "certs": ["Machine Learning Specialization", "TensorFlow Developer learning path"],
    },
    "Python Developer": {
        "skills": ["python", "object-oriented programming", "data structures", "algorithms", "git", "sql", "rest api", "testing", "django"],
        "projects": ["Build a CRUD REST API", "Create a tested automation tool", "Build a Django app with user login"],
        "certs": ["Python programming specialization", "PCAP – Certified Associate Python Programmer"],
    },
    "Full Stack Developer": {
        "skills": ["javascript", "html", "css", "react", "node.js", "rest api", "sql", "git", "testing"],
        "projects": ["Build a full-stack task manager", "Create an e-commerce prototype with login", "Deploy a responsive portfolio with an API"],
        "certs": ["Meta Front-End Developer", "Full-Stack Web Development specialization"],
    },
    "Cloud Engineer": {
        "skills": ["linux", "aws", "azure", "docker", "kubernetes", "networking", "git", "terraform", "ci/cd"],
        "projects": ["Deploy a containerized web app", "Build a CI/CD pipeline", "Provision cloud infrastructure with Terraform"],
        "certs": ["AWS Cloud Practitioner", "Microsoft Azure Fundamentals"],
    },
    "Cybersecurity Analyst": {
        "skills": ["network security", "linux", "cybersecurity", "siem", "cryptography", "python", "owasp"],
        "projects": ["Set up a home lab with security monitoring", "Write a threat model for a web app", "Analyze sample logs for suspicious activity"],
        "certs": ["CompTIA Security+", "Google Cybersecurity Certificate"],
    },
    "Software Engineer": {
        "skills": ["python", "java", "data structures", "algorithms", "object-oriented programming", "git", "sql", "testing", "problem solving"],
        "projects": ["Build a well-tested app with a clear architecture", "Solve and document algorithm problems", "Create a REST API with a database"],
        "certs": ["Software design specialization", "Git and GitHub learning path"],
    },
}
CUSTOM = "Paste a job description instead"

COURSES = {
    "python": "Python fundamentals and intermediate programming",
    "sql": "SQL joins, aggregation, and window functions",
    "machine learning": "Supervised learning and model evaluation",
    "pandas": "Data manipulation with pandas",
    "power bi": "Data modeling and dashboards in Power BI",
    "javascript": "Modern JavaScript and async programming",
    "react": "React components, hooks, and state",
    "docker": "Docker images, containers, and Compose",
    "aws": "AWS core services and cloud basics",
    "git": "Git branching and pull requests",
    "statistics": "Probability and hypothesis testing",
    "scikit-learn": "Pipelines and evaluation with scikit-learn",
    "rest api": "Designing, securing, and testing REST APIs",
    "linux": "Linux command line fundamentals",
    "html": "Semantic, accessible HTML",
    "css": "Responsive layouts with modern CSS",
    "data structures": "Arrays, trees, graphs, and complexity",
}

EXPERIENCE_LEVELS = ["Student / Fresher", "0–2 years", "2–5 years", "5+ years"]


# ------------------------------- LOGIC -------------------------------
def canon(skill):
    s = skill.strip().lower()
    return ALIASES.get(s, s)


def normalize(text):
    text = (text or "").lower()
    for alias, full in ALIASES.items():
        text = re.sub(r"(?<![\w.+#])" + re.escape(alias) + r"(?![\w.+#])", full, text)
    return text


def extract_skills(text):
    text = normalize(text)
    found = set()
    for skill in ALL_SKILLS:
        if re.search(r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])", text):
            found.add(skill)
    return found


def read_resume(file):
    if file is None:
        return ""
    name = file.name.lower()
    try:
        if name.endswith(".pdf") and PdfReader:
            return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)
        if name.endswith(".docx") and Document:
            return "\n".join(p.text for p in Document(file).paragraphs)
        if name.endswith(".txt"):
            return file.getvalue().decode("utf-8", errors="ignore")
    except Exception as exc:
        st.warning(f"Couldn't read that file ({exc}). Paste your resume text instead.")
    return ""


def text_similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    try:
        m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([a, b])
        return round(float(cosine_similarity(m[0:1], m[1:2])[0][0]) * 100, 1)
    except ValueError:
        return 0.0


def fit_label(score):
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Good foundation"
    if score >= 40:
        return "Getting there"
    return "Early stage"


def make_roadmap(missing):
    rows = []
    for i, skill in enumerate(missing[:9]):
        phase, weeks = ("Foundations", "Weeks 1–2") if i < 3 else ("Practice", "Weeks 3–4") if i < 6 else ("Portfolio", "Weeks 5–6")
        rows.append({
            "When": weeks,
            "Focus": phase,
            "Skill": skill.title(),
            "What to learn": COURSES.get(skill, f"Fundamentals and hands-on use of {skill}"),
        })
    return rows


def chips(items, kind):
    return "".join(f'<span class="chip {kind}">{s.title()}</span>' for s in items)


def safe(s):
    return str(s).encode("latin-1", "replace").decode("latin-1")


def build_pdf(d):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "SkillBridge AI - Career Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 7, f"Generated: {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    for line in [
        f"Name: {d['name']}",
        f"Target role: {d['role']}",
        f"Experience: {d['experience']}",
        f"Career fit indicator: {d['overall']}%",
    ]:
        pdf.cell(0, 8, safe(line)[:90], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    sections = [
        ("Your skills", sorted(d["user_skills"])),
        ("Skills the role needs", d["required"]),
        ("Matched", d["matched"]),
        ("Gaps to close", d["missing"]),
        ("Next steps", d["notes"]),
    ]
    for heading, items in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for item in items or ["None found in the text provided."]:
            pdf.multi_cell(0, 6, "- " + safe(item), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    return bytes(pdf.output())


# ------------------------------- HEADER -------------------------------
st.markdown('<div class="brand">♛ SkillBridge AI</div>', unsafe_allow_html=True)
st.markdown('<div class="title">See how your skills match the job you want.</div><div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Tell us about yourself, upload your resume, and pick a role. '
    "You get a match score, the skills you're missing, and a six-week plan to close the gap.</div>",
    unsafe_allow_html=True,
)

# ------------------------------- FORM -------------------------------
with st.form("profile"):
    st.markdown("### About you")
    c1, c2 = st.columns(2)
    name = c1.text_input("Full name", placeholder="Alex Morgan")
    experience = c2.selectbox("Experience level", EXPERIENCE_LEVELS)

    st.markdown("### Resume")
    uploaded = st.file_uploader("Upload your resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    pasted = st.text_area(
        "Or paste your resume text",
        height=140,
        placeholder="Only needed if you didn't upload a file, or if your PDF is a scanned image.",
    )

    st.markdown("### Experience")
    st.markdown('<div class="hint">Jobs, internships, and projects. Include what you built and which tools you used.</div>', unsafe_allow_html=True)
    experience_details = st.text_area(
        "Experience details",
        height=150,
        label_visibility="collapsed",
        placeholder="e.g. Intern at Acme Corp (6 months): built a sales dashboard in Power BI using SQL and Excel...",
    )

    st.markdown("### Skills you know")
    picked = st.multiselect(
        "Pick from the list",
        options=sorted({s.title() for s in ALL_SKILLS}),
        placeholder="Search and select skills",
    )
    other_skills = st.text_input("Other skills (comma separated)", placeholder="Figma, Jira, Spark")

    st.markdown("### Target role")
    role = st.selectbox("Role you're aiming for", list(ROLES.keys()) + [CUSTOM])
    jd = ""
    if role == CUSTOM:
        jd = st.text_area("Job description", height=150, placeholder="Paste the full job description here.")
    extra_required = st.text_input("Extra skills the job needs (optional, comma separated)", placeholder="Docker, FastAPI")

    submitted = st.form_submit_button("Analyze my profile")

# ------------------------------- ANALYSIS -------------------------------
if submitted:
    file_text = read_resume(uploaded)
    resume_text = "\n".join(t for t in [file_text, pasted, experience_details] if t.strip())

    manual = {canon(s) for s in other_skills.split(",") if s.strip()}
    user_skills = extract_skills(resume_text) | {canon(s) for s in picked} | manual

    if not resume_text.strip() and not user_skills:
        st.error("Add at least one thing to analyze: upload a resume, describe your experience, or pick some skills.")
    elif role == CUSTOM and not jd.strip() and not extra_required.strip():
        st.error("Paste a job description or list the skills the job needs.")
    else:
        extra = {canon(s) for s in extra_required.split(",") if s.strip()}
        if role == CUSTOM:
            required = sorted(extract_skills(jd) | extra)
            target_text = jd + " " + " ".join(extra)
        else:
            required = list(dict.fromkeys(ROLES[role]["skills"] + sorted(extra)))
            target_text = " ".join(required) + " " + " ".join(ROLES[role]["projects"])

        matched = [s for s in required if s in user_skills]
        missing = [s for s in required if s not in user_skills]
        coverage = round(len(matched) / len(required) * 100, 1) if required else 0.0
        similarity = text_similarity(resume_text, target_text) if resume_text.strip() else 0.0
        overall = round(0.65 * coverage + 0.35 * similarity, 1) if resume_text.strip() else coverage

        st.session_state["result"] = {
            "name": name.strip() or "Candidate",
            "role": role if role != CUSTOM else "Custom job description",
            "role_key": role,
            "experience": experience,
            "user_skills": user_skills,
            "required": required,
            "matched": matched,
            "missing": missing,
            "coverage": coverage,
            "similarity": similarity,
            "overall": overall,
            "has_text": bool(resume_text.strip()),
            "roadmap": make_roadmap(missing),
            "notes": [
                "Pick the first missing skill and build one small project with it.",
                "Add measurable results and project links to your resume.",
                "Tailor your resume honestly to the wording of the job posting.",
            ],
        }

# ------------------------------- RESULTS -------------------------------
d = st.session_state.get("result")
if d:
    st.markdown("---")
    first = d["name"].split()[0] if d["name"] != "Candidate" else "your"
    heading = f"{first}'s results" if first != "your" else "Your results"
    st.markdown(f"## {heading} for {d['role']}")

    st.markdown(
        f'<div class="score-wrap"><span class="score">{d["overall"]:.0f}%</span>'
        f'<span class="score-label">{fit_label(d["overall"])}</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(int(min(100, d["overall"])))
    st.caption(
        f"{len(d['matched'])} of {len(d['required'])} required skills matched. "
        "This is a rough indicator based on the text you gave, not a hiring prediction."
    )

    tab1, tab2, tab3 = st.tabs(["Skill match", "Learning plan", "Download"])

    with tab1:
        st.markdown("**Skills you have that this role needs**")
        st.markdown(chips(d["matched"], "ok") if d["matched"] else "_None yet. Add skills or more detail above._", unsafe_allow_html=True)
        st.markdown("**Skills to build**")
        st.markdown(chips(d["missing"], "gap") if d["missing"] else "_You cover every skill we looked for._", unsafe_allow_html=True)
        other = sorted(d["user_skills"] - set(d["required"]))
        if other:
            st.markdown("**Your other skills**")
            st.markdown(chips(other, "neutral"), unsafe_allow_html=True)
        if d["has_text"]:
            st.caption(
                f"Score = 65% skill coverage ({d['coverage']}%) + 35% resume text similarity ({d['similarity']}%). "
                f"Experience level ({d['experience']}) is shown for context and doesn't change the score."
            )

    with tab2:
        if d["roadmap"]:
            st.write("A six-week order to work through. Adjust the pace to your schedule.")
            st.dataframe(pd.DataFrame(d["roadmap"]), hide_index=True)
        else:
            st.success("No gaps found. Focus on portfolio depth and interview practice.")
        info = ROLES.get(d["role_key"])
        if info:
            st.markdown("**Project ideas**")
            for p in info["projects"]:
                st.markdown(f"- {p}")
            st.markdown("**Certifications to look at**")
            for c in info["certs"]:
                st.markdown(f"- {c}")
        elif d["missing"]:
            st.markdown("**Project ideas**")
            for s in d["missing"][:3]:
                st.markdown(f"- Build a small project that shows off **{s.title()}**.")

    with tab3:
        st.write("Save your results.")
        st.download_button("Download PDF report", data=build_pdf(d), file_name="skillbridge_report.pdf", mime="application/pdf")
        csv = pd.DataFrame({
            "Skill": [s.title() for s in d["required"]],
            "Status": ["Have" if s in d["matched"] else "Gap" for s in d["required"]],
        }).to_csv(index=False).encode("utf-8")
        st.download_button("Download skill checklist (CSV)", data=csv, file_name="skillbridge_skills.csv", mime="text/csv")

    st.caption("Skills are found by keyword matching, so synonyms and skills outside the built-in list can be missed. Your resume is only processed during this session.")
