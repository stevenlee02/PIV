import os, json, re
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import spacy
import networkx as nx
from collections import defaultdict
from openai import OpenAI

# ---------------- 初始化 ----------------
app = FastAPI()
nlp = spacy.load("en_core_web_sm")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- 工具函数 ----------------
def detect_author_name(text):
    """自动检测小说作者"""
    match = re.search(r'(?:by|author[:\s]+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text[:1500], re.IGNORECASE)
    if match:
        author_name = match.group(1).strip()
        parts = author_name.lower().split()
        print(f"🧾 Detected author name: {author_name}")
        return parts
    return []

def split_text_by_sentence(text):
    """按句子拆分"""
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

def extract_persons(chunk):
    """提取句子中的人名"""
    doc = nlp(chunk)
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]

def clean_name(name: str) -> str:
    """标准化人名"""
    name = re.sub(r"[\r\n]", " ", name)
    name = re.sub(r"[^A-Za-z\s\-']", "", name)
    return name.strip()

def is_valid_name(name: str, skip_words) -> bool:
    """过滤无效人名"""
    if len(name) < 2:
        return False
    return all(k.lower() not in name.lower() for k in skip_words)

# ---------------- 主分析接口 ----------------
@app.post("/analyze")
async def analyze(file: UploadFile):
    text = (await file.read()).decode("utf-8", errors="ignore")

    # 动态跳过封面部分
    match = re.search(r'Chapter\s+1', text, re.IGNORECASE)
    if match:
        text = text[match.start():]

    author_parts = detect_author_name(text)
    skip_words = [
        "copyright", "edition", "chapter", "preface",
        "project", "release", "translator", "gutenberg"
    ] + author_parts

    print("🔹 Splitting sentences and extracting names...")
    sentences = split_text_by_sentence(text)
    all_names, sent_persons = [], []

    for sent in sentences:
        persons = [clean_name(p) for p in extract_persons(sent) if p.strip()]
        persons = [p for p in persons if is_valid_name(p, skip_words)]
        if persons:
            sent_persons.append(persons)
            all_names.extend(persons)

    unique_names = sorted(set(all_names))[:200]
    print(f"Extracted {len(unique_names)} candidate names.")

    prompt = f"""
You are an expert in literary text analysis.

You are given a list of PERSON entities automatically extracted from a novel.
They may include:
- Character names (like "Elizabeth Bennet", "Mr. Darcy")
- Nicknames (like "Lizzy")
- Non-character entities (like "Jane Austen", "Project Gutenberg")

Here is the extracted list (max 200 entries):
{unique_names}

Your job:
1. Identify which names correspond to **fictional characters** from the novel.
2. Merge all name variants that refer to the same character (for example: "Darcy", "Mr. Darcy", "Fitzwilliam Darcy" → "Mr. Darcy").
3. Exclude any of the following:
   - Author names
   - Publishers, editors, illustrators, and translators
   - Non-human entities or locations
   - Words like "Chapter", "Copyright", "Project Gutenberg"
4. Output only valid JSON **in this exact structure**:

{{
  "Canonical Character Name": ["variant1", "variant2", "variant3"]
}}

Rules:
- Output must be a single valid JSON object.
- No markdown, no explanations, no comments, no backticks.
- Keep only names of fictional characters that appear within the story.
"""

    # ---------------- GPT 角色聚合 ----------------
    try:
        print("🔹 Calling GPT model for name normalization...")
        resp = client.responses.create(
            model="gpt-4o-mini",
            temperature=0,
            input=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        mapping_text = resp.output_text.strip()
        print("🔹 GPT raw output preview (first 400 chars):")
        print(mapping_text[:400])
        mapping = json.loads(mapping_text)
        print("GPT mapping parsed successfully.")
    except Exception as e:
        print("GPT fallback:", e)
        # 如果 GPT 失败，给 identity mapping（每个名字映射到自己）
        mapping = {name: [name] for name in unique_names}

    # ---------------- 建立变体 -> canonical 映射（保证完整） ----------------
    variant_to_canon = {}

    # 先把 GPT 提供的 canonical 映射清洗并插入
    skip_words = ["copyright", "edition", "chapter", "preface",
                  "project", "release", "translator", "gutenberg"] + author_parts

    for canon, variants in mapping.items():
        canon_clean = clean_name(canon)
        if not is_valid_name(canon_clean, skip_words):
            continue
        # 将 canonical 本身也映射到 canonical（保证identity）
        variant_to_canon[canon_clean] = canon_clean
        for v in variants:
            v_clean = clean_name(v)
            if is_valid_name(v_clean, skip_words):
                variant_to_canon[v_clean] = canon_clean

    # 对 unique_names 中任何未被 GPT 映射的名字，做 identity 映射（避免遗漏）
    for n in unique_names:
        n_clean = clean_name(n)
        if n_clean not in variant_to_canon and is_valid_name(n_clean, skip_words):
            variant_to_canon[n_clean] = n_clean

    # ---------------- 用 canonical 名称构建共现网络 ----------------
    G = nx.Graph()
    cooccurrence_texts = defaultdict(list)

    for sent, persons in zip(sentences, sent_persons):
        # 将句子中提取的每个名字替换成 canonical（若没有canonical则跳过）
        canon_list = []
        for p in persons:
            p_clean = clean_name(p)
            canon_name = variant_to_canon.get(p_clean)
            if canon_name:
                canon_list.append(canon_name)
        canon_list = list(set(canon_list))  # 去重

        if not canon_list:
            continue

        # 增加节点计数（使用 canonical 名）
        for a in canon_list:
            if a in G.nodes:
                G.nodes[a]["count"] += 1
            else:
                G.add_node(a, count=1)

        # 增加边并保存上下文句子
        for i in range(len(canon_list)):
            for j in range(i + 1, len(canon_list)):
                a, b = canon_list[i], canon_list[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
                # use sorted key so "A|B" and "B|A" map same
                key = "|".join(sorted([a, b]))
                cooccurrence_texts[key].append(sent[:400])

    # ---------------- 清理并生成返回的 nodes/links ----------------
    # 确保 nodes 与 links 的一致性：先取 G 的节点集合（不提前删除）
    all_nodes = list(G.nodes)
    node_set = set(all_nodes)

    # links：只保留两端都在 node_set 的边
    links = []
    for u, v, data in G.edges(data=True):
        if u in node_set and v in node_set and data.get("weight", 0) > 0:
            links.append({"source": u, "target": v, "value": int(data["weight"])})

    # nodes：输出所有在 node_set 中的节点（可以在前端再做阈值隐藏）
    nodes = [{"id": n, "value": int(G.nodes[n].get("count", 1))} for n in all_nodes]

    print(f"Final Network: {len(nodes)} nodes, {len(links)} edges.")
    return {"nodes": nodes, "links": links, "contexts": dict(cooccurrence_texts)}

@app.get("/ping")
async def ping():
    return {"message": "pong", "key_loaded": bool(os.getenv("OPENAI_API_KEY"))}
