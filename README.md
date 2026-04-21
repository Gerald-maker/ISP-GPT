---
title: Career Advisory GPT
emoji: 🎓
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# Career Advisory GPT 🎓🤖

## ✨ Features
- 🔍 **MS Program Recommendation** – tailored suggestions from student profiles  
- 📚 **Knowledge Base Q&A** – career & study guidance via AI  
- 🤝 **Advisory Support** – admissions, exams, relocation tips  
- ⚡ **LLM-powered** – modern LLMs for natural responses  
- 🔗 **Extensible** – connect to student DBs & external APIs

---

## 🏗️ Tech Stack
- **Python** (backend)
- Optional: **LangChain/LlamaIndex** for RAG
- **FastAPI/Flask** (API)
- Optional: **React/Next.js** (frontend)
- **GitHub Actions** (CI/CD)

---

## 🚀 Local Development

```bash
# 1) Clone
git clone https://github.com/Gerald-maker/ISP-GPT.git
cd Career-Advisory-GPT

# 2) Create & activate venv
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Run the app
python app.py
# App runs at http://localhost:8000 (adjust if your code differs)
#
