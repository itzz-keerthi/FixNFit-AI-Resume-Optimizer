# FixNFit-AI-Resume-Optimizer

## Overview
**Fix & Fit** is a personal AI-powered assistant that checks whether your resume aligns with a specific job description and suggests personalized improvements using a chatbot interface.

## Demo Video
[![Watch the video](https://img.youtube.com/vi/hB7amsLXxaM/maxresdefault.jpg)](https://www.youtube.com/watch?v=hB7amsLXxaM)

## 💡 Motivation

Crafting the perfect resume for each job is tedious. I built ResumeGPT to:
- Automatically match my resume with job descriptions.
- Identify missing skills, mismatched content, or gaps.
- Provide smart suggestions for improving the resume using an AI chatbot.

This tool helps me tailor my applications and enhance my chances of landing interviews.

## 🚀 Features

- 📝 **Resume & JD Analyzer**: Upload your resume and paste the job description.
- ⚖️ **Match Score**: Get a compatibility score between your resume and the job description.
- 💬 **Chatbot Recommendations**: Interact with a chatbot that provides improvement suggestions.
- 🎯 **Skill Gap Analysis**: Identify missing or underrepresented skills.
- ✨ **Natural Language Feedback**: Easy-to-understand insights for optimizing your resume.

## 🛠️ Tech Stack

- **Python**
- **OpenAI API** (for language understanding and chatbot)
- **Streamlit** (for interactive UI)
- **LangChain** (optional, for modular prompt logic)
- **PDF/Text parsing libraries**: `PyMuPDF`, `pdfminer`, etc.

## 🖥️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/itzz-keerthi/FixNFit-AI-Resume-Optimizer.git
cd resumegpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:

```bash
streamlit run app.py
```

## 🧠 Future Improvements

- Add support for parsing LinkedIn job postings directly.
- Store improvement history for tracking progress over time.
- Add visual charts for skill matching.
---



