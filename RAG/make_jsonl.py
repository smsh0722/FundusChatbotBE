# make_jsonl.py
import os, re, json, uuid
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("data/clean")
OUTPUT_JSONL = Path("data/jsonl/docs.jsonl")

# ------------------------------
# B) ODIR 라벨 자동 태깅 룰 (간단/확장 가능)
# ------------------------------
LABEL_MAP = {
    "Diabetes": [
        "diabetic retinopathy","microaneurysm","npdr","pdr","dot-blot","intraretinal hemorrhage",
        "irma","neovascular","macular edema","4-2-1 rule"
    ],
    "Age-related Macular Degeneration (AMD)": [
        "age-related macular degeneration","amd","drusen","geographic atrophy","cnv",
        "wet amd","neovascular amd","are ds","macular degeneration"
    ],
    "Glaucoma": [
        "glaucoma","cup-to-disc","c/d ratio","optic nerve cupping","nerve fiber layer thinning",
        "visual field defect","iop","open-angle","angle-closure"
    ],
    "Cataract": [
        "cataract","lens opacity","phacoemulsification","intraocular lens","iols",
        "cloudy lens","blurred vision (lens)","nuclear sclerosis","cortical cataract"
    ],
    "Hypertension": [
        "hypertensive retinopathy","av nicking","arteriovenous nicking","flame hemorrhage",
        "cotton-wool spot","papilledema","malignant hypertension","copper wiring","silver wiring"
    ],
    "Myopia": [
        "pathologic myopia","myopic","myopia-related maculopathy","lacquer crack","staphyloma",
        "peripapillary atrophy","tessellated fundus","patchy atrophy","diffuse atrophy","meta-pm"
    ],
    "Normal": [
        "normal fundus","healthy optic disc","normal macula","physiologic cupping",
        "normal foveal reflex","normal vessels","normal retina"
    ],
    "Other Diseases/Abnormalities": [
        "retinal detachment","rhegmatogenous","vein occlusion","artery occlusion",
        "crvo","brvo","c r a o","cherry-red spot","macular hole","epiretinal membrane",
        "tumor","retinoblastoma","central serous","serous retinopathy"
    ]
}

# ------------------------------
# A) 청크 분리: 헤딩+길이 기반, 오버랩 포함
# - 목표 길이: ~2000자(대략 300~500토큰) / overlap: ~300자
# ------------------------------
TARGET_CHARS = 2000
OVERLAP_CHARS = 300

HEAD_PATTERNS = [
    re.compile(r'^(\d+(\.\d+)*)\s+[A-Z][^\n]{3,}$'),   # "3.1 Definition"
    re.compile(r'^[A-Z][A-Za-z0-9 \-/]{5,100}$'),      # ALL CAPS/Title-ish lines
]

def split_by_headings(text:str):
    lines = text.splitlines()
    head_idxs = set([0])
    for i,l in enumerate(lines):
        ls = l.strip()
        if not ls: 
            continue
        for pat in HEAD_PATTERNS:
            if pat.match(ls):
                head_idxs.add(i)
                break
    head_idxs = sorted(list(head_idxs) + [len(lines)])
    sections = []
    for a,b in zip(head_idxs, head_idxs[1:]):
        chunk = "\n".join(lines[a:b]).strip()
        if len(chunk) >= 200:  # 너무 짧은 건 스킵
            sections.append(chunk)
    return sections if sections else [text]

def window_with_overlap(long_text:str, target=TARGET_CHARS, overlap=OVERLAP_CHARS):
    long_text = re.sub(r'\n{3,}', '\n\n', long_text.strip())
    if len(long_text) <= target:
        return [long_text]
    out = []
    start = 0
    while start < len(long_text):
        end = min(start + target, len(long_text))
        out.append(long_text[start:end].strip())
        if end == len(long_text): break
        start = max(0, end - overlap)
    return out

def chunk_text(text:str):
    # 1) 섹션 분리
    sections = split_by_headings(text)
    # 2) 섹션별 길이 분할(오버랩)
    chunks = []
    for sec in sections:
        chunks.extend(window_with_overlap(sec))
    # 가벼운 클린
    clean_chunks = []
    for c in chunks:
        c = re.sub(r'[ \t]+', ' ', c)
        c = re.sub(r'\n +', '\n', c).strip()
        if len(c) >= 300:   # 너무 짧은 조각은 버림
            clean_chunks.append(c)
    return clean_chunks

# ------------------------------
# B) 라벨 자동 태깅
# ------------------------------
def auto_labels(text:str):
    t = text.lower()
    tags = set()
    for label, kws in LABEL_MAP.items():
        for kw in kws:
            if kw in t:
                tags.add(label)
                break
    # 최소 1개 보장: 없으면 Other로 (또는 Unknown)
    return sorted(tags) if tags else ["Other"]

# ------------------------------
# 유틸: 제목/소스 추정(파일명 기반)
# ------------------------------
SOURCE_MAP = {
    "Normal.txt": "https://www.ncbi.nlm.nih.gov/books/NBK11533/?utm_source=chatgpt.com",
    "Diabetic_Retinopathy_PPP.txt": "https://www.aao.org/education/preferred-practice-pattern/diabetic-retinopathy-ppp",
    "Primary_Open-Angle_Glaucoma_PPP.txt": "https://www.aao.org/education/preferred-practice-pattern/primary-open-angle-glaucoma-ppp",
    "Cataract_in_the_Adult_Eye_PPP_7.9.25.txt":"https://www.aao.org/education/preferred-practice-pattern/cataract-in-adult-eye-ppp-2021-in-press",
    "Age-Related_Macular_Degeneration_PPP.txt":"https://www.aao.org/education/preferred-practice-pattern/age-related-macular-degeneration-ppp",
    "Hypertensive_Retinopathy.txt":"https://www.ncbi.nlm.nih.gov/books/NBK525980/",
    "Myopia.txt":"https://www.ncbi.nlm.nih.gov/books/NBK580529/",
}

def guess_title_and_source(path:Path):
    title = path.stem.replace("_", " ").replace("-", " ").title()
    source = SOURCE_MAP.get(path.name, "")
    return title, source

# ------------------------------
# E) JSONL 생성
# ------------------------------
def process_one_file(path:Path, out_f):
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n","\n").replace("\r","\n").strip()
    if not text:
        return 0

    doc_id = str(uuid.uuid4())
    title, source = guess_title_and_source(path)

    chunks = chunk_text(text)
    count = 0
    for ch in chunks:
        labels = auto_labels(ch)
        row = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "labels": labels,                # ▼ 우선 전부 disease_card로 두고 시작
            "category": "disease_card",
            "text": ch
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
    return count

def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as out_f:
        total_chunks = 0
        files = sorted([p for p in INPUT_DIR.glob("**/*.txt")])
        for p in tqdm(files, desc="Ingest"):
            total_chunks += process_one_file(p, out_f)
    print(f"Done. Files: {len(files)}, Chunks: {total_chunks}, Out: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
