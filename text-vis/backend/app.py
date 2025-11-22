import os, json, re
from typing import List, Dict
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

# 泛化黑名单 用于clean_name
BLACKLIST = {
    "project gutenberg", "gutenberg", "ebook", "e-text", "etext", "epub", "html", "ascii", "txt",
    "produced by", "transcribed by", "distributed proofreaders", "formatting", "release date",
    "license", "limited warranty", "liability", "copyright", "literary archive", "archive",
    "foundation", "internal revenue", "irs", "ein", "government",
    "preface", "editor", "translator", "illustrator", "publisher", "chapter", "send", "send:",
    "general information", "terms", "warranty", "possibility", "punitive", "liable", "direct",
    # 常见作者
    "o henry", "o. henry", "michael s hart", "project gutenberg literary archive foundation",
    "mark twain", "jane austen", "charles dickens", "william shakespeare",
    # 比喻or历史人物
    "solomon", "king solomon", "caesar", "napoleon", "homer"
}

# 前缀集合
HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "mme", "mlle", "sir", "lady", "lord", "master", "mx"}

# 用于在prompt中排除的词
SKIP_WORDS = ["copyright", "edition", "chapter", "preface", "project", "release", "translator", "gutenberg"]

def split_text_by_sentence(text):
    """按句子拆分"""
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

def extract_persons(chunk):
    """提取句子中的人名"""
    doc = nlp(chunk)
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]


# 处理名字
def clean_name(name: str) -> str:
    if not name:
        return ""

    name = re.sub(r"[\r\n]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[\[\(\{].*?[\]\)\}]", "", name)
    name = re.sub(r"[^A-Za-z\s']", "", name).strip()

    if not name:
        return ""

    parts = name.split()
    if not parts:
        return ""

    # honorific
    if parts[0].lower().strip(".") in HONORIFICS and len(parts) == 1:
        return ""

    # 重建名字
    name_cleaned = " ".join(p.capitalize() for p in parts)

    # 去掉后缀残留的词 例如 --but said
    name_cleaned = re.sub(r"\b(but|said|says|then|also)\b", "", name_cleaned, flags=re.IGNORECASE).strip()

    # 黑名单过滤
    lname = name_cleaned.lower()
    for bad in BLACKLIST:
        if bad in lname:
            return ""

    return name_cleaned



def clean_illustrations(text: str) -> str:
    """
    Remove Gutenberg illustration blocks such as:
    [Illustration: ...]
    [Illustration ...]
    [_Copyright 1894 by ...]
    """

    text = re.sub(
        r"\[.*?illustration.*?\]",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r"\[_copyright.*?\]",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    """text = re.sub(r"\[[^\]]{0,200}\]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\n\s+\n", "\n\n", text)
    """

    return text


# ---------------- 主分析接口 ----------------
@app.post("/analyze")
async def analyze(file: UploadFile):
    text = (await file.read()).decode("utf-8", errors="ignore")
    # 删掉Gutenberg illustration
    text = clean_illustrations(text)

    skip_words = [
        "copyright", "edition", "chapter", "preface",
        "project", "release", "translator", "gutenberg"
    ]
    
    print("🔹 Splitting sentences and extracting names...")
    sentences = split_text_by_sentence(text)
    all_names, sent_persons = [], []
    sent_persons = []  

    for sent in sentences:
        raw_persons = extract_persons(sent)  
        persons = []
        for p in raw_persons:
            p_clean = clean_name(p)
            if p_clean:
                persons.append(p_clean)
        if persons:
            sent_persons.append(persons)
            all_names.extend(persons)
        else:
            sent_persons.append([])


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
{json.dumps(unique_names, ensure_ascii=False)}

Your job:
1. Identify which names correspond to fictional characters from the novel.
2. Merge all name variants that refer to the same character (for example: "Darcy", "Mr. Darcy", "Fitzwilliam Darcy" → "Mr. Darcy").
3. Exclude any of the following:
   - Author names
   - Publishers, editors, illustrators, and translators
   - Non-human entities or locations
   - Words like "Chapter", "Copyright", "Project Gutenberg"
4. Output only valid JSON in this exact structure:

{{ "Canonical Character Name": ["variant1", "variant2", "variant3"] }}

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

        mapping_text = ""
        if hasattr(resp, "output_text") and resp.output_text:
            mapping_text = resp.output_text.strip()
        else:
            # Fallback: try to extract from resp.output or resp.get("output", ...)
            try:
                # resp.output is often a list of dicts with 'content' items
                if isinstance(resp.output, list):
                    parts = []
                    for item in resp.output:
                        # try a few likely keys
                        if isinstance(item, dict):
                            # some SDKs: item["content"][0]["text"]
                            content = item.get("content") or item.get("text")
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and "text" in c:
                                        parts.append(c["text"])
                                    elif isinstance(c, str):
                                        parts.append(c)
                            elif isinstance(content, str):
                                parts.append(content)
                    mapping_text = "\n".join(parts).strip()
                else:
                    mapping_text = str(resp)
            except Exception:
                mapping_text = str(resp)
        print("🔹 GPT raw output preview (first 400 chars):")
        print(mapping_text[:400])

        # 解析 JSON
        cleaned = mapping_text
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        mapping = json.loads(cleaned)
        print("GPT mapping parsed successfully.")
    except Exception as e:
        print("🔸 GPT fallback: parsing failed or GPT error:", e)
        # 如果GPT失败 每个名字映射到自己
        mapping = {name: [name] for name in unique_names}

    # ---------------- 建立变体 -> canonical 映射（保证完整） ----------------
    variant_to_canon = {}

    # 写入GPT提供的mapping 包括canonical本身
    for canon, variants in mapping.items():
        canon_clean = clean_name(canon)
        if not canon_clean:
            continue
        # 忽略含skip_words的canonical
        if any(k in canon_clean.lower() for k in skip_words):
            continue
        # 把canonical映射到自己 保证canonical出现在mapping
        variant_to_canon[canon_clean] = canon_clean
        # 把变体映射到canonical
        if isinstance(variants, list):
            for v in variants:
                v_clean = clean_name(v)
                if v_clean and not any(k in v_clean.lower() for k in skip_words):
                    variant_to_canon[v_clean] = canon_clean
    
    # 只从现有的名字中提取部分匹配 不创建新名字 
    def enhance_mapping(variant_to_canon, all_extracted_names):
        enhanced_mapping = variant_to_canon.copy()
        canonicals = list(set(enhanced_mapping.values()))
        all_names_set = set(all_extracted_names)  # 所有实际提取到的名字
    
        # 为每个已存在的名字变体 检查其部分是否也在提取的名字列表中
        for existing_variant in list(enhanced_mapping.keys()):
            parts = existing_variant.split()
            if len(parts) <= 1:
                continue
            
            # 检查每个部分是否独立存在于提取的名字中
            for part in parts:
                if (len(part) > 2 and  # 避免太短的匹配
                    part in all_names_set and  
                    part not in enhanced_mapping and
                    not any(k in part.lower() for k in skip_words)):
                    # 将这个部分映射到同一个 canonical
                    enhanced_mapping[part] = enhanced_mapping[existing_variant]
    
        return enhanced_mapping

    variant_to_canon = enhance_mapping(variant_to_canon, unique_names)

    # 对unique_names中任何未被GPT映射的名字做identity映射
    for n in unique_names:
        n_clean = clean_name(n)
        if not n_clean:
            continue

        # 如果尾部是s尝试去掉再匹配
        if n_clean.lower().endswith("s"):
            singular = n_clean[:-1]
            if singular in variant_to_canon:
                variant_to_canon[n_clean] = variant_to_canon[singular]
                continue

        # 如果名字已经在映射中就跳过
        if n_clean in variant_to_canon:
            continue
        
        # 检查是否应该映射到现有的 canonical
        matched = False
        for canon in set(variant_to_canon.values()):
            # 检查n_clean是否是canon的一部分 或者canon是n_clean的一部分
            if (n_clean in canon.split() or canon in n_clean.split() or
                n_clean == canon):
                variant_to_canon[n_clean] = canon
                matched = True
                break
    
        # 如果没有匹配到任何现有的 做identity映射
        if not matched and not any(k in n_clean.lower() for k in skip_words):
            variant_to_canon[n_clean] = n_clean

    
    # ---------------- 用 canonical 名称构建共现网络 ----------------
    def build_cooccurrence_network(sentences, sent_persons, variant_to_canon):
        G = nx.Graph()
        cooccurrence_texts = defaultdict(list)

        for sent, persons in zip(sentences, sent_persons):
            # 将句子中提取的每个名字替换成canonical 若没有canonical则跳过
            canon_list = []
            for p in persons:
                p_clean = clean_name(p)
                if not p_clean:
                    continue
                canon_name = variant_to_canon.get(p_clean)
                if canon_name:
                    canon_list.append(canon_name)
            canon_list = list(set(canon_list))  # 去重

            if not canon_list:
                continue

            # 增加节点计数 使用canonical name
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
                    if len(cooccurrence_texts[key]) < 5:  # 限制上下文条数 
                        ###这里是否需要区分长文本和短文本 对于短文本如果是过长的上下文 导致不同人物被聚合到一起
                        cooccurrence_texts[key].append(sent[:400])

        nodes_to_keep = [n for n in G.nodes if G.nodes[n].get("count", 0) >= 5]
    
        # 创建子图 只保留出现次数>=5的节点
        G_filtered = G.subgraph(nodes_to_keep).copy()
    
        all_nodes = list(G_filtered.nodes)
        node_set = set(all_nodes)


        # links：只保留两端都在node_set的边
        links = []
        for u, v, data in G.edges(data=True):
            if u in node_set and v in node_set and data.get("weight", 0) > 0:
                links.append({"source": u, "target": v, "value": int(data["weight"])})

        # nodes：输出所有在node_set中的节点 按count排序（大到小）
        nodes = [{"id": n, "value": int(G.nodes[n].get("count", 1))} for n in all_nodes]
        nodes = sorted(nodes, key=lambda x: x["value"], reverse=True)

        print(f"Final Network: {len(nodes)} nodes, {len(links)} edges.")
   
        filtered_contexts = {}
        for key, contexts in cooccurrence_texts.items():
            chars = key.split("|")
            if len(chars) == 2 and chars[0] in node_set and chars[1] in node_set:
                filtered_contexts[key] = contexts

        return {"nodes": nodes, "links": links, "contexts": dict(cooccurrence_texts)}
    
    result = build_cooccurrence_network(sentences, sent_persons, variant_to_canon)
    return result

@app.get("/ping")
async def ping():
    return {"message": "pong", "key_loaded": bool(os.getenv("OPENAI_API_KEY"))}
