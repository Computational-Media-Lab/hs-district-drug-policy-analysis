import os, re, csv, torch, PyPDF2, nltk, unicodedata

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor

#=======================================#
###        DATA CONFIGURATION         ###   
#=======================================#
nltk.download('punkt')


if os.path.exists("/work/10824/ayang4625/ls6/School_Drug_Policy_Project"):
    PROJECT_DIR = "/work/10824/ayang4625/ls6/School_Drug_Policy_Project"
else:
    PROJECT_DIR = "C:\\Users\\antho\\Computational-Media-Lab\\hs-district-drug-policy-analysis"

BASE_DIR = os.path.join(PROJECT_DIR, "drug-policy-data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "district_analyses_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE = "Texas"

# Define categories and their corresponding terms for analysis
CATEGORIES = [
    "total number of drug or alcohol terms mentioned",
    "number of times suspension or synonyms are mentioned",
    "number of times expulsion or synonyms is mentioned",
    "number of time transfer or referral to disciplinary alternative school is mentioned",
    "number of times transfer or referral to juvenile justice program is mentioned",
    "number of times arrest or synonyms are mentioned",
    "number of times drug sniffing dogs or other drug screening procedures are mentioned",
    "number of times police or school resource officers are mentioned",
    "number of times any other terms are mentioned that relate to disciplinary approaches to student drug and alcohol use",
    "number of times transfer or referral to treatment programs are mentioned",
    "number of times transfer or referral to recovery schools are mentioned",
    "number of times school based health centers or synonyms are mentioned",
    "number of times transfer or referral to mental health clinics or other off campus health facilities are mentioned",
    "number of times trauma-informed care or synonyms are mentioned",
    "number of times restorative justice approaches or synonyms are mentioned",
    "number of times drug and alcohol use prevention programs are mentioned",
    "number of times health-services staff such as school nurses, social workers, psychologists, or substance use counselors are mentioned",
    "number of times any other terms are mentioned that relate to a health-oriented approach to student drug and alcohol use"
]

CATEGORY_TERMS = {
    "total number of drug or alcohol terms mentioned": ["drug", "alcohol", "substance", "marijuana", "cocaine", "opioid", "tobacco", "vape"],
    "number of times suspension or synonyms are mentioned": ["suspension", "in-school suspension", "temporary removal"],
    "number of times expulsion or synonyms is mentioned": ["expulsion", "permanent removal"],
    "number of time transfer or referral to disciplinary alternative school is mentioned": ["transfer", "alternative school", "disciplinary placement"],
    "number of times transfer or referral to juvenile justice program is mentioned": ["juvenile justice", "court referral", "probation"],
    "number of times arrest or synonyms are mentioned": ["arrest", "police action", "detention by authorities"],
    "number of times drug sniffing dogs or other drug screening procedures are mentioned": 
    ["drug dog", "drug screening", "random drug test", "Sniffing dog", "Sniffer dog"],
    "number of times police or school resource officers are mentioned": ["school resource officer", "SRO", "police presence"],
    "number of times any other terms are mentioned that relate to disciplinary approaches to student drug and alcohol use": 
    ["disciplinary approach", "policy enforcement", "student conduct", "disciplinary", "rule violation", "code of conduct"],
    "number of times transfer or referral to treatment programs are mentioned": ["treatment program", "rehabilitation", "substance program"],
    "number of times transfer or referral to recovery schools are mentioned": ["recovery school", "alternative education", "substance recovery school"],
    "number of times school based health centers or synonyms are mentioned": ["school health center", "nurse office", "clinic"],
    "number of times transfer or referral to mental health clinics or other off campus health facilities are mentioned": 
    ["mental health clinic", "therapy center", "counseling center", "rehab"],
    "number of times trauma-informed care or synonyms are mentioned": ["trauma-informed", "supportive care", "emotional support"],
    "number of times restorative justice approaches or synonyms are mentioned": ["restorative justice", "mediation", "conflict resolution"],
    "number of times drug and alcohol use prevention programs are mentioned": ["prevention program", "education program", "awareness program"],
    "number of times health-services staff such as school nurses, social workers, psychologists, or substance use counselors are mentioned": 
    ["nurse", "social worker", "psychologist", "counselor", "substance use counselor", "addiction therapist"],
    "number of times any other terms are mentioned that relate to a health-oriented approach to student drug and alcohol use": 
    ["health approach", "wellness program", "support services", "mental health", "therapy", "counselor", "psychologist", "school nurse", "sick"]
}

#======================================#
###      LLM Model Configuration     ###
#======================================#

token = "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"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
login(token = token)

def choose_llm_model(deepseek = True, llama = False, mistral = False, other = ''):
    if deepseek:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp", token=token)
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp", device_map="auto", token=token)
    elif llama:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", token=token)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", device_map="auto", token=token)
    elif mistral:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Large-3-675B-Instruct-2512", token=token)
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Large-3-675B-Instruct-2512", device_map="auto", token=token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(other, token = token)
        model = AutoModelForCausalLM.from_pretrained(other, device_map="auto", token = token)

    return tokenizer, model


#========================================#
###           PDF Analysis             ###
#========================================#

# Extract pdf text from a given file path
def extract_pdf_text(pdf_path):
    text_fragments = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                # 1. Fix Mojibake/Encoding errors (e.g., individualâ€™s -> individual's)
                try:
                    # Re-encode and decode back to clean UTF-8 strings
                    page_text = page_text.encode('cp1252').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    # Fallback to standard normalization if byte sequence doesn't match
                    page_text = unicodedata.normalize("NFKC", page_text)

                # 2. Re-map specific common leftover corrupted characters
                char_fixes = {
                    "â€”": "—", "â€?": "—", "â€™": "'", "â€œ": '"', "â€": '"', 
                    "â€?": '"', "ând": "and", "em- ployment": "employment",
                    "dis- ability": "disability", "in- dividualized": "individualized"
                }
                for bad, good in char_fixes.items():
                    page_text = page_text.replace(bad, good)

                # 3. Clean page breaks/newlines without breaking words
                cleaned_page_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', page_text)
                text_fragments.append(cleaned_page_text)

    full_text = " ".join(text_fragments)
    raw_sentences = nltk.sent_tokenize(full_text)

    clean_sentences = []
    for sentence in raw_sentences:
        # Normalize whitespace
        s = re.sub(r'\s+', ' ', sentence).strip()
        
        # --- AGGRESSIVE JUNK FILTERS ---
        
        # Skip Table of Contents dot leaders (e.g., Discrimination.................3)
        if re.search(r'\.{4,}\s*\d+', s) or s.count('.') > 5:
            continue
            
        # Skip legal PDF header/footer stamps (e.g., "DATE ISSUED: 1/22/2025 3 of 23")
        if "DATE ISSUED" in s or "UPDATE " in s or re.search(r'\b\d+\s+of\s+\d+\b', s):
            continue
            
        # Skip lines ending abruptly with hanging legal code structures (e.g., DAA(LEGAL)-P)
        if re.search(r'[A-Z]{2,}\(LEGAL\)', s):
            continue
            
        # Basic baseline syntax checks
        if not s or s.isdigit() or len(s.split()) < 4 or s.isupper():
            continue
            
        clean_sentences.append(s)

    return clean_sentences

# Categorize a line of text using a language model
def categorize_line_with_llm(line, category_terms, deepseek = True, llama = False, mistral = False, other = ''):
    prompt = f"""
    Given a sentence from a school district policy document, categorize it into either none, one, or more of the given categories using the key words that go with each category.
    These are the categories and their corresponding key words:
    {category_terms}

    For example, if given the sentence "Student will be suspended if drug use is detected", then categorize it as the following categories:
    total number of drug or alcohol terms mentioned,number of times suspension or synonyms are mentioned

    The output format should be categories separated by commas with no spaces in between.
    Terms with different tenses (past, present, future) or singular vs plural count as the same term. For example, "suspended" and "suspension" should be counted as the same term. Similarly, "drug" and "drugs" should be counted as the same term.
    Ignore any headings, subheadings or other non-sentence text. Only categorize actual sentences that are part of the policy text.

    Categorize this school drug policy line:
    "{line}"

    Return the relevant categories, or "None" if irrelevant.
    """
    tokenizer, model = choose_llm_model(deepseek = deepseek, llama = llama, mistral = mistral, other = other)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=30)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if result == "None" or not result:
        return [] # Return an empty list if no categories or "None"

    # Split by comma and strip whitespace from each category
    predicted_categories = [c.strip() for c in result.split(',') if c.strip()]

    # Filter to only include categories that are actually in CATEGORY_TERMS keys
    valid_categories = [c for c in predicted_categories if c in category_terms.keys()]

    return valid_categories

# Process each school district
def process_district(district_path, district_name, deepseek = True, llama = False, mistral = False, other = ''):
    pdf_files = [os.path.join(root, f) for root, _, files in os.walk(district_path)
                 for f in files if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDFs for {district_name}. Skipping.")
        return

    district_output = os.path.join(OUTPUT_DIR, district_name)
    os.makedirs(district_output, exist_ok=True)

    summary_csv = os.path.join(district_output, f"{district_name}_summary.csv")
    detailed_csv = os.path.join(district_output, f"{district_name}_detailed.csv")

    counts = {cat: 0 for cat in CATEGORIES}
    detailed_rows = []

    # Read PDFs in parallel (returns a list of lists of clean sentences)
    with ThreadPoolExecutor() as executor:
        texts_by_file = list(executor.map(extract_pdf_text, pdf_files))

    # Pair each pre-cleaned sentence with its corresponding source file
    all_sentences = []
    for pdf_path, clean_sentences in zip(pdf_files, texts_by_file):
        filename = os.path.basename(pdf_path)
        for s in clean_sentences:
            # No need to tokenise or clean again; extract_pdf_text already handled it
            all_sentences.append((filename, s))

    # Extract just the text strings for embedding
    just_sentences = [s[1] for s in all_sentences]

    # Precompute embeddings for all sentences (fast)
    sentence_embeddings = embedding_model.encode(just_sentences, convert_to_tensor=True)

    # Cache for results
    llm_cache = {}

    for i, (filename, sentence) in enumerate(tqdm(all_sentences, desc=f"🔍 {district_name}")):
        matched = False
        for category, terms in CATEGORY_TERMS.items():
            if any(re.search(rf"\b{re.escape(term)}\b", sentence, re.IGNORECASE) for term in terms):
                counts[category] += 1
                detailed_rows.append([category, filename, sentence])
                matched = True
                break

        if not matched:
            if sentence in llm_cache:
                cat = llm_cache[sentence]
            else:
                sims = util.cos_sim(sentence_embeddings[i], sentence_embeddings)
                if torch.max(sims) > 0.92:
                    similar_idx = torch.argmax(sims).item()
                    cat = llm_cache.get(just_sentences[similar_idx], None)
                else:
                    cat = categorize_line_with_llm(sentence, CATEGORIES, deepseek = deepseek, llama = llama, mistral = mistral, other = other)
                    llm_cache[sentence] = cat

            if cat and cat != "None":
                counts[cat] += 1
                detailed_rows.append([cat, filename, sentence])

    # Write results
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["district", "state"] + CATEGORIES)
        writer.writerow([district_name, STATE] + [counts[cat] for cat in CATEGORIES])

    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Topic Category", "PDF", "Text"])
        writer.writerows(detailed_rows)

    print(f"✅ Finished {district_name}: {len(all_sentences)} sentences analyzed.")


#=====================================#
###         MAIN EXECUTION          ###
#=====================================#
print("=========================================")
print("HS District Drug Policy LLM Analysis Tool")
print("=========================================\n")

print(f"📂 Project directory: {PROJECT_DIR}")
print(f"📂 Input data folder: {BASE_DIR}")
print(f"📂 Output CSV folder: {OUTPUT_DIR}\n")


TARGET_DISTRICT = None  # Change to None to run all districts
OMIT_DISTRICT = None  # Change to None to include all districts

if TARGET_DISTRICT:
    districts_to_process = [TARGET_DISTRICT]
elif OMIT_DISTRICT:
    districts_to_process = [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and d != OMIT_DISTRICT
    ]
else:
    districts_to_process = [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

print(f"🚀 Starting analysis for {len(districts_to_process)} district(s)...")

for district_name in districts_to_process:
    district_path = os.path.join(BASE_DIR, district_name)
    if not os.path.isdir(district_path):
        print(f"❌ District folder not found: {district_path}. Skipping...")
        continue
    print(f"📂 Processing {district_name}...")
    process_district(district_path, district_name, deepseek = True, llama = False, mistral = False, other = '')
    print(f"✅ Completed processing {district_name}.\n")

print(f"🎉 Finished processing all districts. Output saved to: {OUTPUT_DIR}")