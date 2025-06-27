import os
import fitz  #PyMuPDF
import spacy
import tempfile
import docx
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import time

def load_skills(file_path = "skills.txt"):
    try:
        with open(file_path, "r", encoding= "utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    except FileNotFoundError:
        return[]
    
SKILL_KEYWORDS = load_skills()
    
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
   
#extract text from pdf
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path) 
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()  
    return text
   
   
#extract text from docx
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])
   
#extract text from txt
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
        return f.read()
       
#detect file type and extract text accordingly
def extract_text(file_path, file_extension):
    if file_extension == ".pdf":
            return extract_text_from_pdf(file_path)
    elif file_extension == ".docx":
            return extract_text_from_docx(file_path)
    elif file_extension == ".txt":
            return extract_text_from_txt(file_path)
    else:
            return ""
        
#basic skill keywords based extraction
def extract_skills(text):
    extracted = set()
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text.lower():
            extracted.add(skill)
    return extracted
    
    
#semantic similarity score
def get_similarity(resume_text, job_text):
    emb_resume = model.encode(resume_text, convert_to_tensor = True)
    emb_job = model. encode(job_text, convert_to_tensor = True)
    similarity = util.pytorch_cos_sim(emb_resume, emb_job)
    return float(similarity[0][0])    
    
#streamlit UI
st.set_page_config(page_title = "AI Resume Matcher", layout = "wide")
st.title ("📄 AI Resume Matcher with Multi Format Support")
st.markdown("Upload a `.pdf`, `.docx`, `.txt` resume and compare it with the job description.")
    
uploaded_file = st.file_uploader("📎Upload your resume", type = ["pdf", "docx", "txt"])
job_description = st.text_area("📝 Paste the Job Description", height = 200)
    
if uploaded_file and job_description:
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    uploaded_file_bytes = uploaded_file.read()  
    tmp_file.write(uploaded_file_bytes)
    tmp_path = tmp_file.name
    tmp_file.close()
            
    resume_text = extract_text(tmp_path, file_ext)
        
time.sleep(0.2)
if os.path.exists(tmp_path):
    try:
        os.remove(tmp_path)
    except PermissionError:
        time.sleep(1)
        try:
            os.remove(tmp_path)
        except Exception as e:
            st.warning(f"Temporary file could not be deleted even after retry. Error: {e}")
    # else:
    #   st.info("Temporary file already deleted or missing.")
    if resume_text.strip() == "":
            st.error("Could not extract text from the uploaded file.")
    else:
            skills = extract_skills(resume_text)
            relevance_score = get_similarity(resume_text, job_description)
            
            st.subheader("🔍 Extracted Skills")
            st.write(", ".join(skills) if skills else "No matching skills found.")
            
            st.subheader("📊 Relevance Score")
            st.metric(label = "Match %", value = f"{relevance_score * 100:.2f}%")
            
            if relevance_score < 0.5:
                st.warning("This resume may not be a strong match.")
            else:
                st.success("This resume appears relevant.")
else:
    st.info("Please upload a resume and job description to see the results.")
        