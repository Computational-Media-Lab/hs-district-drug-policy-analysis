import os, re, csv, torch, faiss, PyPDF2
import numpy as np
import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor

#=======================================#
###        DATA CONFIGURATION         ###   
#=======================================#
PROJECT_DIR = "/work/10824/ayang4625/ls6/School_Drug_Policy_Project"
BASE_DIR = os.path.join(PROJECT_DIR, "drug-policy-data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "district_analyses_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE = "Texas"

# Categories and their associated terms for keyword matching
CATEGORIES = [
    "total number of drug or alcohol terms mentioned",
    "total number of disciplinary terms mentioned",
    "total number of health or treatment terms mentioned",
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
    "total number of disciplinary terms mentioned": ["disciplinary", "rule violation", "code of conduct"],
    "total number of health or treatment terms mentioned": ["mental health", "therapy", "counselor", "psychologist", "school nurse", "sick"],
    "number of times suspension or synonyms are mentioned": ["suspension", "in-school suspension", "temporary removal"],
    "number of times expulsion or synonyms is mentioned": ["expulsion", "permanent removal"],
    "number of time transfer or referral to disciplinary alternative school is mentioned": ["transfer", "alternative school", "disciplinary placement"],
    "number of times transfer or referral to juvenile justice program is mentioned": ["juvenile justice", "court referral", "probation"],
    "number of times arrest or synonyms are mentioned": ["arrest", "police action", "detention by authorities"],
    "number of times drug sniffing dogs or other drug screening procedures are mentioned": 
    ["drug dog", "drug screening", "random drug test", "Sniffing dog", "Sniffer dog"],
    "number of times police or school resource officers are mentioned": ["school resource officer", "SRO", "police presence"],
    "number of times any other terms are mentioned that relate to disciplinary approaches to student drug and alcohol use": 
    ["disciplinary approach", "policy enforcement", "student conduct"],
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
    ["health approach", "wellness program", "support services"]
}

#======================================#
###      LLM Model Configuration     ###
#======================================#

token = "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"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
login(token = token)
os.environ["HF_HOME"] = "/work/10824/ayang4625/ls6/hf_cache"

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

# Extract pdf from a a given path
def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# Categorize a line of text using a language model
def categorize_line_with_llm(line, categories, deepseek = True, llama = False, mistral = False, other = ''):
    prompt = f"""
    Categorize this school drug policy line:
    "{line}"

    Categories:
    {', '.join(categories)}

    Return the single best category name, or "None" if irrelevant.
    """
    tokenizer, model = choose_llm_model(deepseek = deepseek, llama = llama, mistral = mistral, other = other)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=30)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return result if result in categories else None

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

    # Read PDFs in parallel (faster IO)
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_pdf_text, pdf_files))

    all_lines = []
    for text in texts:
        all_lines.extend([l.strip() for l in text.split("\n") if l.strip()])

    # Precompute embeddings for all lines (fast)
    line_embeddings = embedding_model.encode(all_lines, convert_to_tensor=True)

    # Cache for results
    llm_cache = {}

    for i, line in enumerate(tqdm(all_lines, desc=f"🔍 {district_name}")):
        matched = False
        for category, terms in CATEGORY_TERMS.items():
            if any(re.search(rf"\b{re.escape(term)}\b", line, re.IGNORECASE) for term in terms):
                counts[category] += 1
                detailed_rows.append([category, os.path.basename(pdf_files[0]), line])
                matched = True
                break

        if not matched:
            # Find if similar line was already processed by Llama
            if line in llm_cache:
                cat = llm_cache[line]
            else:
                # Compare semantic similarity with previous lines
                sims = util.cos_sim(line_embeddings[i], line_embeddings)
                if torch.max(sims) > 0.92:  # reuse similar result if >92% match
                    similar_idx = torch.argmax(sims).item()
                    cat = llm_cache.get(all_lines[similar_idx], None)
                else:
                    cat = categorize_line_with_llm(line, CATEGORIES, deepseek = deepseek, llama = llama, mistral = mistral, other = other)
                    llm_cache[line] = cat

            if cat and cat != "None":
                counts[cat] += 1
                detailed_rows.append([cat, os.path.basename(pdf_files[0]), line])

    # Write results
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["district", "state"] + CATEGORIES)
        writer.writerow([district_name, STATE] + [counts[cat] for cat in CATEGORIES])

    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Topic Category", "PDF", "Text"])
        writer.writerows(detailed_rows)

    print(f"✅ Finished {district_name}: {len(all_lines)} lines analyzed.")


#=====================================#
###         MAIN EXECUTION          ###
#=====================================#
print("=========================================")
print("HS District Drug Policy LLM Analysis Tool")
print("=========================================\n")

print(f"📂 Project directory: {PROJECT_DIR}")
print(f"📂 Input data folder: {BASE_DIR}")
print(f"📂 Output CSV folder: {OUTPUT_DIR}\n")

llm_model = input("Which LLM Model would you like to use? (Type \"deepseek\" for DeepSeek, type 'llama' for LLaMA, type \"mistral\" for Mistral, or provide another model name): ")

if llm_model == "deepseek":
    deepseek = True
    llama = False
    mistral = False
    other = False
elif llm_model == "llama":
    deepseek = False
    llama = True
    mistral = False
    other = False
elif llm_model == "mistral":
    deepseek = False
    llama = False
    mistral = True
    other = False
else:
    deepseek = False
    llama = False
    mistral = False
    other = llm_model

TARGET_DISTRICT = None  # Change to None to run all districts

if TARGET_DISTRICT:
    districts_to_process = [TARGET_DISTRICT]
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
    process_district(district_path, district_name, deepseek = deepseek, llama = llama, mistral = mistral, other = other)

print(f"🎉 Finished processing all districts. Output saved to: {OUTPUT_DIR}")
