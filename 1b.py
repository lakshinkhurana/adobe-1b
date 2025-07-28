import os
import argparse
import json
import glob
from datetime import datetime
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import time
import logging
import nltk
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize NLTK resources
def initialize_nltk():
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    required_packages = ['punkt', 'punkt_tab']
    
    for package in required_packages:
        try:
            # Try to find existing tokenizer
            nltk.data.find(f'tokenizers/{package}')
            logging.info(f"NLTK {package} tokenizer found")
        except LookupError:
            logging.info(f"Downloading required NLTK data: {package}")
            try:
                # Try downloading with a timeout
                import socket
                socket.setdefaulttimeout(30)  # 30 second timeout
                nltk.download(package, quiet=True, download_dir=nltk_data_dir)
                logging.info(f"NLTK {package} downloaded successfully")
            except Exception as e:
                logging.error(f"Error downloading NLTK {package}: {str(e)}")
                # Try to use local copy if available
                local_data = os.path.join(nltk_data_dir, 'tokenizers', package)
                if os.path.exists(local_data):
                    logging.info(f"Using existing local NLTK data for {package}")
                    continue
                raise RuntimeError(f"Failed to initialize NLTK {package}. Please check your internet connection or manually download the tokenizer.")

# --- Helper Functions ---
NOISE_WORDS = [
    'copyright', 'references', 'table of contents', 'index', 'acknowledgments',
    'disclaimer', 'appendix', 'about the author', 'preface', 'foreword', 'glossary'
]

def is_noise_section(section):
    text = section['text'].lower()
    return any(word in text for word in NOISE_WORDS)

def improved_section_heading(line):
    # Improved heading detection: numbered, all-caps, 'CHAPTER', 'Section', etc.
    if re.match(r'^(\d+\.?\d*)?\s*([A-Z][A-Za-z0-9 ,\-]+)$', line):
        return True
    if re.match(r'^(CHAPTER|SECTION)\s+\d+', line, re.IGNORECASE):
        return True
    if line.isupper() and len(line) > 3:
        return True
    return False

def extract_sections_from_pdf(pdf_path, chunk_size=5):
    sections = []
    if not os.path.exists(pdf_path):
        return [], f"File not found: {pdf_path}"
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return [], "PDF has no pages"
            total_pages = len(pdf.pages)
            # Process pages in chunks for better memory management
            for chunk_start in range(0, total_pages, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pages)
                current_section = None
                
                for i in range(chunk_start, chunk_end):
                    page = pdf.pages[i]
                    text = page.extract_text() or ''
                    lines = text.split('\n')
                    
                    if current_section is None:
                        current_section = {'page_number': i+1, 'section_title': 'Introduction', 'text': ''}
                    
                    for line in lines:
                        line = line.strip()
                        if improved_section_heading(line):
                            if current_section['text'].strip():
                                sections.append(current_section)
                            current_section = {
                                'page_number': i+1,
                                'section_title': line,
                                'text': ''
                            }
                        else:
                            current_section['text'] += line + '\n'
                
                # Save the last section of the chunk
                if current_section and current_section['text'].strip():
                    sections.append(current_section)
                    current_section = None
                
                # Clear page objects to free memory
                for i in range(chunk_start, chunk_end):
                    pdf.pages[i]._objects = {}
                
        return sections, None
    except Exception as e:
        return [], str(e)

def get_pdf_title(pdf_path):
    return os.path.splitext(os.path.basename(pdf_path))[0]

def embed_texts(model, texts, batch_size=32, cache_dir='.embedding_cache'):
    import hashlib
    import pickle
    import os

    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache key from texts
    cache_key = hashlib.md5(''.join(texts).encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f'emb_{cache_key}.pkl')
    
    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass  # If loading fails, compute embeddings
            
    # Compute embeddings
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    # Cache the results
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
    except:
        pass  # If caching fails, just return the embeddings
        
    return embeddings

def filter_sections(sections, min_length=30):
    filtered = []
    skipped = []
    for s in sections:
        if len(s['text']) > min_length and not is_noise_section(s):
            filtered.append(s)
        else:
            skipped.append(s)
    return filtered, skipped

def deduplicate_sections(sections):
    seen = set()
    deduped = []
    for s in sections:
        key = s['text'].strip()
        if key and key not in seen:
            deduped.append(s)
            seen.add(key)
    return deduped

def preprocess_text(text):
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    # Convert to lowercase for better matching
    text = text.lower()
    return text

def compute_weighted_similarity(query_emb, section_emb, title_weight=1.5):
    """Compute similarity with additional weighting for important features"""
    # Base similarity using cosine similarity
    base_sim = cosine_similarity(query_emb, section_emb)[0][0]
    
    # Apply sigmoid function to make similarities more pronounced
    weighted_sim = 1 / (1 + np.exp(-10 * (base_sim - 0.5)))
    
    return weighted_sim

def rank_sections(query_emb, section_embs):
    # Compute similarities with enhanced weighting
    sims = np.array([compute_weighted_similarity(query_emb.reshape(1, -1), 
                                               emb.reshape(1, -1)) 
                     for emb in section_embs])
    
    # Apply threshold to filter out very low similarities
    threshold = 0.1
    sims[sims < threshold] = 0
    
    ranked_indices = np.argsort(sims)[::-1]
    return ranked_indices, sims

def safe_tokenize(text):
    try:
        return sent_tokenize(text)
    except Exception as e:
        logging.warning(f"Error in sentence tokenization: {str(e)}")
        # Fallback to simple period-based splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences if sentences else [text]

def build_output(doc_metadata, all_sections, ranked_indices, sims, top_n=10):
    extracted_sections = []
    subsection_analysis = []
    for rank, idx in enumerate(ranked_indices[:top_n]):
        sec = all_sections[idx]
        extracted_sections.append({
            'document': sec['document'],
            'page_number': sec['page_number'],
            'section_title': sec['section_title'],
            'importance_rank': rank+1,
            'similarity_score': float(sims[idx])  # Add similarity score
        })
        sentences = safe_tokenize(sec['text'])
        refined_text = ' '.join(sentences[:3]).strip()
        subsection_analysis.append({
            'document': sec['document'],
            'refined_text': refined_text,
            'page_number': sec['page_number'],
            'total_sentences': len(sentences)  # Add sentence count
        })
    return extracted_sections, subsection_analysis

def main():
    # Initialize NLTK first
    try:
        initialize_nltk()
    except Exception as e:
        logging.error(str(e))
        return

    parser = argparse.ArgumentParser(description='Persona-Driven Document Intelligence')
    parser.add_argument('--pdf_dir', type=str, required=True, help='Directory containing PDF files')
    parser.add_argument('--persona', type=str, required=True, help='Persona description')
    parser.add_argument('--job', type=str, required=True, help='Job to be done')
    parser.add_argument('--output', type=str, default='output.json', help='Output JSON file')
    parser.add_argument('--min_section_length', type=int, default=30, help='Minimum section length to consider')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top sections to output')
    parser.add_argument('--skipped_sections_output', type=str, default=None, help='Optional: Write skipped sections to this file instead of including in main output')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for processing sections')
    parser.add_argument('--cache_dir', type=str, default='.embedding_cache', help='Directory to cache embeddings')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker threads for PDF processing')
    parser.add_argument('--chunk_size', type=int, default=5, help='Number of PDF pages to process in memory at once')
    args = parser.parse_args()

    start_time = time.time()
    pdf_files = glob.glob(os.path.join(args.pdf_dir, '*.pdf'))
    if not pdf_files:
        logging.error('No PDF files found in directory.')
        return
    logging.info(f'Found {len(pdf_files)} PDF files.')

    # Load model (CPU only)
    logging.info('Loading embedding model...')
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Enhanced query preparation
    # Combine persona and job with better structure
    query = f"A {args.persona} who is {args.job}"
    # Add emphasis on key terms by repetition
    query = f"{query}. Important aspects: {args.persona}, {args.job}"
    # Preprocess the query
    query = preprocess_text(query)
    query_emb = model.encode([query])

    all_sections = []
    doc_metadata = []
    skipped_sections = []
    failed_files = defaultdict(str)

    # Parallel PDF section extraction
    with ThreadPoolExecutor() as executor:
        future_to_pdf = {executor.submit(extract_sections_from_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc='Extracting PDFs'):
            pdf_path = future_to_pdf[future]
            title = get_pdf_title(pdf_path)
            doc_metadata.append({'filename': os.path.basename(pdf_path), 'title': title})
            try:
                sections, error = future.result()
                if error:
                    failed_files[os.path.basename(pdf_path)] = error
                    continue
                for section in sections:
                    section['document'] = os.path.basename(pdf_path)
                    section['title'] = title
                filtered, skipped = filter_sections(sections, min_length=args.min_section_length)
                all_sections.extend(filtered)
                skipped_sections.extend(skipped)
            except Exception as e:
                failed_files[os.path.basename(pdf_path)] = str(e)

    if not all_sections:
        logging.warning('No valid sections found after filtering. Outputting skipped sections for review.')
        output = {
            'metadata': {
                'input_documents': doc_metadata,
                'persona': args.persona,
                'job_to_be_done': args.job,
                'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
                'flag': 'NO_VALID_SECTIONS_FOUND',
                'failed_files': dict(failed_files)
            },
            'extracted_sections': [],
            'subsection_analysis': [],
            'skipped_sections': skipped_sections
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        elapsed = time.time() - start_time
        logging.info(f'Output written to {args.output} in {elapsed:.2f} seconds.')
        return

    # Deduplicate sections
    before_dedup = len(all_sections)
    all_sections = deduplicate_sections(all_sections)
    after_dedup = len(all_sections)
    logging.info(f'{before_dedup} sections after filtering, {after_dedup} after deduplication. {len(skipped_sections)} sections skipped. {len(failed_files)} files failed.')

    # Prepare texts with better weighting and preprocessing
    section_texts = []
    for s in all_sections:
        # Preprocess title and text
        title = preprocess_text(s['section_title'])
        text = preprocess_text(s['text'])
        
        # Create a weighted combination with enhanced title importance
        weighted_text = f"{title} {title} {title}. " # Triple weight for title
        
        # Add first sentence of text with double weight
        first_sentence = safe_tokenize(text)[:1]
        if first_sentence:
            weighted_text += f"{first_sentence[0]} {first_sentence[0]}. "
        
        # Add rest of the text
        weighted_text += text
        
        section_texts.append(weighted_text)

    # Process large document sets efficiently with dynamic batching
    def get_optimal_batch_size(total_size):
        # Adjust batch size based on total dataset size
        if total_size < 100:
            return total_size
        elif total_size < 1000:
            return 256
        else:
            return 512

    batch_size = get_optimal_batch_size(len(section_texts))
    top_n = args.top_n
    section_embs = []
    batch_indices = []

    # Use memory-efficient batching
    for i in tqdm(range(0, len(section_texts), batch_size), desc='Embedding sections'):
        batch = section_texts[i:i+batch_size]
        try:
            embs = embed_texts(model, batch, batch_size=min(32, len(batch)))
            section_embs.append(embs)
            batch_indices.extend(range(i, min(i+batch_size, len(section_texts))))
        except Exception as e:
            logging.warning(f"Error processing batch {i//batch_size}: {str(e)}")
            # Use smaller batch size for retry
            retry_size = len(batch) // 2
            if retry_size > 0:
                for j in range(0, len(batch), retry_size):
                    sub_batch = batch[j:j+retry_size]
                    try:
                        sub_embs = embed_texts(model, sub_batch, batch_size=min(16, len(sub_batch)))
                        section_embs.append(sub_embs)
                        batch_indices.extend(range(i+j, min(i+j+retry_size, len(section_texts))))
                    except Exception as sub_e:
                        logging.error(f"Failed to process sub-batch: {str(sub_e)}")

    if section_embs:
        section_embs = np.vstack(section_embs)
    else:
        raise RuntimeError("No embeddings could be generated")

    # Compute relevance and rank
    ranked_indices, sims = rank_sections(query_emb, section_embs)
    # Keep only top-N in memory
    ranked_indices = ranked_indices[:top_n]

    # Build output
    extracted_sections, subsection_analysis = build_output(doc_metadata, all_sections, ranked_indices, sims, top_n=top_n)

    output = {
        'metadata': {
            'input_documents': doc_metadata,
            'persona': args.persona,
            'job_to_be_done': args.job,
            'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
            'failed_files': dict(failed_files)
        },
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis
    }
    if args.skipped_sections_output:
        with open(args.skipped_sections_output, 'w', encoding='utf-8') as f:
            json.dump(skipped_sections, f, indent=2)
    else:
        output['skipped_sections'] = skipped_sections

    elapsed = time.time() - start_time
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    logging.info(f'Output written to {args.output} in {elapsed:.2f} seconds.')

if __name__ == '__main__':
    main() 