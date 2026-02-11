# app.py
import os
import csv
import pickle
from pathlib import Path
from typing import List
from datetime import datetime

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import fitz  # PyMuPDF

# Optional imports
try:
    import openai
except Exception:
    openai = None

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ---------- 配置 ----------
INDEX_FILE = "index_store.pkl"
USAGE_LOG = "usage_log.csv"
MODEL_NAME = os.environ.get("SB_MODEL", "all-MiniLM-L6-v2")
MAX_CONTEXT_CHARS = 3000
st.set_page_config(page_title="RAG Chatbot MVP - Integrated", layout="wide")

# 以 USD / per 1k tokens 为单位 (input_price_per_1k, output_price_per_1k)
# 请在生产前核对 OpenAI 官方定价并更新此映射。
MODEL_PRICE_PER_1K = {
    "gpt-3.5-turbo": (0.00050, 0.00150),
    "gpt-4o": (0.00250, 0.01000),
    "gpt-4-0613": (0.03000, 0.06000),
    "gpt-4-32k": (0.06000, 0.12000),
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-3.5-turbo-0613": (0.00150, 0.00200),
}

# ---------- 帮助函数 ----------
@st.cache_resource
def load_embedding_model(name=MODEL_NAME):
    return SentenceTransformer(name)

def read_txt(file_bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except Exception:
        return file_bytes.decode("latin-1")

def read_pdf_bytes(bytes_data) -> str:
    doc = fitz.open(stream=bytes_data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size=300, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def build_index(text_chunks: List[str], model):
    if len(text_chunks) == 0:
        raise ValueError("没有文本块可以索引")
    embs = model.encode(text_chunks, show_progress_bar=False, convert_to_numpy=True)
    nbrs = NearestNeighbors(n_neighbors=min(5, len(text_chunks)), metric="cosine").fit(embs)
    return {"nbrs": nbrs, "embs": embs, "chunks": text_chunks}

def save_index(obj, path=INDEX_FILE):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_index(path=INDEX_FILE):
    if Path(path).exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def retrieve(query, model, nbrs_obj, top_k=3):
    if not nbrs_obj:
        return []
    q_emb = model.encode([query], convert_to_numpy=True)
    dists, idxs = nbrs_obj["nbrs"].kneighbors(q_emb, n_neighbors=min(top_k, len(nbrs_obj["chunks"])))
    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        results.append((float(dist), nbrs_obj["chunks"][idx]))
    return results

def assemble_context(retrieved):
    parts = []
    total = 0
    for dist, txt in retrieved:
        if total + len(txt) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 0:
                parts.append(txt[:remaining])
            break
        parts.append(txt)
        total += len(txt)
    return "\n\n---\n\n".join(parts)

# ---------- Token counting / cost estimate ----------
def count_tokens_with_tiktoken(text: str, model_name: str = "gpt-3.5-turbo"):
    if tiktoken is None:
        return None
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
    toks = enc.encode(text)
    return len(toks)

def estimate_tokens_and_cost(prompt_text: str, expected_completion_tokens: int, model_name: str):
    # Count prompt tokens
    prompt_tokens = None
    if tiktoken is not None:
        try:
            prompt_tokens = count_tokens_with_tiktoken(prompt_text, model_name)
        except Exception:
            prompt_tokens = None
    if prompt_tokens is None:
        prompt_tokens = max(1, int(len(prompt_text) / 4))

    total_tokens = prompt_tokens + expected_completion_tokens

    model_key = model_name.lower()
    cost_estimate = None
    if model_key in MODEL_PRICE_PER_1K:
        iprice, oprice = MODEL_PRICE_PER_1K[model_key]
        cost_estimate = (prompt_tokens / 1000.0) * iprice + (expected_completion_tokens / 1000.0) * oprice
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": expected_completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_estimate,
    }

# ---------- OpenAI helper ----------
def call_openai_generate(prompt: str, system_prompt: str = None, model_name="gpt-3.5-turbo", max_tokens=256, temperature=0.2):
    if openai is None:
        return "OpenAI SDK 未安装（请在 requirements 中加入 openai）", None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "未检测到 OPENAI_API_KEY 环境变量，请先设置。", None

    openai.api_key = api_key
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        out = None
        try:
            out = resp.choices[0].message.get("content") if resp.choices else None
        except Exception:
            out = None
        if out is None:
            try:
                out = resp.choices[0].text if resp.choices else ""
            except Exception:
                out = ""
        usage = None
        try:
            usage = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
        except Exception:
            usage = None
        return out.strip(), usage
    except Exception as e:
        return f"调用 OpenAI 出错：{e}", None

# ---------- Usage logging ----------
def append_usage_log(row: dict, path=USAGE_LOG):
    header = ["timestamp", "model", "prompt_tokens_est", "completion_tokens_est", "total_tokens_est", "estimated_cost_usd", "prompt_tokens_actual", "completion_tokens_actual", "total_tokens_actual"]
    write_header = not Path(path).exists()
    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            row.get("timestamp"),
            row.get("model"),
            row.get("prompt_tokens_est"),
            row.get("completion_tokens_est"),
            row.get("total_tokens_est"),
            row.get("estimated_cost_usd"),
            row.get("prompt_tokens_actual"),
            row.get("completion_tokens_actual"),
            row.get("total_tokens_actual"),
        ])

# ---------- UI ----------

st.title("📚 RAG Chatbot MVP — Integrated")
st.write("上传文件 → 构建索引 → 提问 → 可选择生成（OpenAI / 本地 HF）。首次加载模型会从 HuggingFace 下载。")

col_left, col_main = st.columns([1, 2])

with col_left:
    st.header("上传 & 索引")
    uploaded = st.file_uploader("上传 TXT 或 PDF（可多选）", type=["txt", "pdf"], accept_multiple_files=True)
    chunk_size = st.number_input("分块大小（词）", min_value=100, max_value=2000, value=300, step=50)
    overlap = st.number_input("分块重叠（词）", min_value=0, max_value=1000, value=50, step=10)
    if st.button("构建索引（或覆盖）"):
        if not uploaded:
            st.warning("请先上传至少一个文件")
        else:
            all_chunks = []
            for f in uploaded:
                try:
                    b = f.read()
                    if f.type == "text/plain" or f.name.lower().endswith(".txt"):
                        text = read_txt(b)
                    else:
                        text = read_pdf_bytes(b)
                except Exception as e:
                    st.error(f"读取文件 {f.name} 失败：{e}")
                    text = ""
                if text:
                    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                    all_chunks.extend(chunks)
            if not all_chunks:
                st.error("没有可用文本片段")
            else:
                with st.spinner("加载 embedding 模型并向量化（第一次会慢）..."):
                    emb_model = load_embedding_model()
                    try:
                        idx = build_index(all_chunks, emb_model)
                        st.session_state["index"] = idx
                        save_index(idx)
                        st.success(f"索引构建完成，片段数量：{len(all_chunks)}，已保存到 {INDEX_FILE}")
                    except Exception as e:
                        st.error(f"索引构建失败：{e}")

    st.write("---")
    st.header("生成选项")
    gen_method = st.selectbox("生成器", options=["None", "OpenAI API", "Local HF (transformers)"])
    if gen_method == "OpenAI API":
        st.caption("需要设置环境变量 OPENAI_API_KEY；调用前会显示 token 与费用估算。")
    if gen_method == "Local HF (transformers)":
        st.caption("本地生成需要 transformers + torch；CPU 上可能很慢。")
    openai_model = st.text_input("OpenAI 模型名（若使用 OpenAI）", value="gpt-3.5-turbo")
    gen_max_tokens = st.slider("生成最大 tokens（completion 最大长度）", 64, 1024, 256)
    gen_temperature = st.slider("生成温度（temperature）", 0.0, 1.0, 0.2, 0.05)

with col_main:
    st.header("问答 / 聊天")
    if "index" not in st.session_state:
        st.info("请先上传并构建索引，或加载示例。")
    query = st.text_input("请输入问题：", key="query_input")
    top_k = st.slider("检索片段数 (top_k)", 1, 5, 3)

    if st.button("查询") and query.strip():
        emb_model = load_embedding_model()
        idx = st.session_state.get("index", None)
        if not idx:
            st.error("未找到索引，请先构建索引")
        else:
            results = retrieve(query, emb_model, idx, top_k=top_k)
            st.write("### 检索到的片段（按相似度）")
            for i, (dist, txt) in enumerate(results):
                st.markdown(f"**片段 {i+1}（相似度距离={dist:.3f}）**")
                st.write(txt[:1500] + ("..." if len(txt) > 1500 else ""))
            assembled = assemble_context(results)
            st.write("### 合并后的上下文（已截断）")
            st.write(assembled[:4000] + ("..." if len(assembled) > 4000 else ""))

            prompt_template = (
                "你是一个知识库问答助手。请基于下面从知识库检索到的上下文，"
                "以及用户的问题，给出简洁、准确、可引用来源的回答。\n\n"
                "上下文:\n{context}\n\n用户问题:\n{question}\n\n回答:"
            )
            prompt = prompt_template.format(context=assembled, question=query)

            if gen_method == "OpenAI API":

                est = estimate_tokens_and_cost(prompt, expected_completion_tokens=gen_max_tokens, model_name=openai_model)
                st.write("**估算（仅供参考）**")
                st.write(f"- prompt tokens (估计): {est['prompt_tokens']}")
                st.write(f"- completion tokens (设定上限): {est['completion_tokens']}")
                st.write(f"- total tokens (估计): {est['total_tokens']}")
                if est["estimated_cost_usd"] is not None:
                    st.write(f"- 估算费用 (USD): ${est['estimated_cost_usd']:.6f}")
                else:
                    st.write("- 估算费用: 模型价格未配置，无法估算（请在代码的 MODEL_PRICE_PER_1K 添加条目）")

                if st.button("Proceed with OpenAI call (will incur cost)"):
                    with st.spinner("调用 OpenAI 生成中..."):
                        out, usage = call_openai_generate(prompt, system_prompt=None, model_name=openai_model, max_tokens=gen_max_tokens, temperature=gen_temperature)
                        st.write("### 生成回答")
                        st.write(out)
                        # 写入 usage log（若有）
                        row = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "model": openai_model,
                            "prompt_tokens_est": est['prompt_tokens'],
                            "completion_tokens_est": est['completion_tokens'],
                            "total_tokens_est": est['total_tokens'],
                            "estimated_cost_usd": est['estimated_cost_usd'],
                            "prompt_tokens_actual": None,
                            "completion_tokens_actual": None,
                            "total_tokens_actual": None,
                        }
                        if usage:
                            try:
                                pt = usage.get('prompt_tokens') or usage.get('input_tokens')
                                ct = usage.get('completion_tokens') or usage.get('output_tokens')
                                tt = usage.get('total_tokens')
                                row['prompt_tokens_actual'] = pt
                                row['completion_tokens_actual'] = ct
                                row['total_tokens_actual'] = tt
                            except Exception:
                                pass
                        try:
                            append_usage_log(row)
                            st.success('已将 usage 保存到 ' + USAGE_LOG)
                        except Exception as e:
                            st.error('保存 usage 失败：' + str(e))

            elif gen_method == "Local HF (transformers)":
                st.info("本地生成：将 prompt 传入 transformers pipeline（若已安装）。")
                if st.button("Local generate"):
                    if pipeline is None:
                        st.error("transformers 未安装或不可用，请安装 transformers + torch。")
                    else:
                        with st.spinner("本地生成中（CPU 可能慢）..."):
                            try:
                                gen = pipeline("text2text-generation", model="google/flan-t5-small")
                                out = gen(prompt, max_length=gen_max_tokens, do_sample=False)
                                st.write("### 本地生成结果")
                                st.write(out[0].get("generated_text", str(out)))
                            except Exception as e:
                                st.error(f"本地生成失败：{e}")
            else:
                st.info("生成关闭：仅返回检索片段。")

st.caption("Token counting via tiktoken (preferred). If tiktoken absent, a heuristic (1 token ≈ 4 chars) is used.")
