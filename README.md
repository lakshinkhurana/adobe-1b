
# Adobe 1B - Persona-Driven Document Intelligence

This project was built for **Adobe's India Hackathon Challenge 1B** and focuses on developing an intelligent document processing system that surfaces the most relevant content from a collection of PDFs based on a given persona and task.

## 🔍 Challenge Statement

> Given a collection of PDF documents and a defined persona with a job-to-be-done, extract the most relevant sections across documents that help the persona accomplish their goal.

---

## 🧠 Key Features

- 📄 **PDF Sectioning**: Breaks down PDFs into sections using smart heading detection.
- 🚫 **Noise Filtering**: Skips indexes, disclaimers, and other irrelevant sections.
- 🔁 **Deduplication**: Removes repeated or redundant text blocks.
- 🧬 **Sentence Embeddings**: Uses `sentence-transformers` to encode and compare text.
- 🔎 **Semantic Search**: Ranks sections using cosine similarity to a query.
- 🪄 **Summarization (Subsection Refinement)**: First-line summarization for extracted content.
- 🧵 **Parallel Processing**: Multi-threaded PDF parsing for speed.
- 🧪 **Fail-safe Logging**: Reports skipped sections and failed PDFs.

---

## 🧰 Tech Stack

- **Python 3**
- [`pdfplumber`](https://github.com/jsvine/pdfplumber)
- [`sentence-transformers`](https://www.sbert.net/)
- `scikit-learn`, `numpy`, `nltk`, `tqdm`
- ThreadPoolExecutor for concurrency

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run the Script

```bash
python 1b.py --pdf_dir "<path-to-pdfs>" --persona "<persona>" --job "<job-to-be-done>" --output "<output.json>"
```

### Example:

```bash
python 1b.py --pdf_dir "sample-dataset/Collection 3/PDFs" --persona "nutrition-conscious food enthusiast" --job "building a balanced and creative weekly meal plan that includes quick, healthy, and diverse recipes for breakfast, lunch, and dinner" --output "Collection3_output.json"
```

---

## 📂 Repository Structure

```
adobe-1b/
├── 1b.py                        # Main script
├── sample-dataset/
│   ├── Collection 1/
│   │   └── PDFs/
│   ├── Collection 2/
│   └── Collection 3/
├── .embedding-cache/                     # Optional cache
├── .gitignore                     
├── LICENSE                    
├── dockerfile               
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

---

## 📝 Output JSON Format

- `metadata`: input documents, persona, job, timestamp
- `extracted_sections`: top relevant sections
- `subsection_analysis`: brief preview sentences
- `skipped_sections`: filtered noise/short sections

---

## 💡 Use Cases

- Personalized document summarization
- Knowledge retrieval from manuals, recipes, whitepapers
- Role-specific onboarding material extraction

---

## 👤 Author

**Lakshin Khurana and Yashvardhan Nayal**  
GitHub: [@lakshinkhurana](https://github.com/lakshinkhurana) and [@YashvardhanNayal0212](https://github.com/YashvardhanNayal0212)

---

## 🏁 License

This project is open-sourced for evaluation and educational purposes.
