from __future__ import annotations
import os, io, re, json, hashlib, textwrap, tempfile
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from neo4j import GraphDatabase, Driver
from pyvis.network import Network

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# ──────────────────────────────────────────────────────────────────────────
# Streamlit page config: must be first Streamlit command
st.set_page_config(layout="wide", page_title="Log Analysis & Problem Solver")

# ──────────────────────────────────────────────────────────────────────────
# Env & Clients
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in environment.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# ── Neo4j connection helper ─────────────────────────────────────────────
def get_driver() -> Driver:
    uri      = os.getenv("NEO4J_URI")
    user     = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))

dr: Driver = get_driver()
DB_NAME = "logindex"  # hardcoded database

# ──────────────────────────────────────────────────────────────────────────
# Prompts definitions
TRIPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert log analyst. Convert each log entry to concise JSON triples capturing key entities and relations. Format: [{{\"subject\": s, \"predicate\": p, \"object\": o}}]."),
    ("human", "{log_line}")
])
INVESTIGATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a forensic log investigator. Use the provided context to:\n1. Identify the timeframe of events.\n2. Highlight key events.\n3. Trace event sequence.\n4. Identify patterns.\n5. Note impacted services.\n6. Check for resolutions.\n7. Summarize findings."),
    ("human", "Context:\n{context}\n\nQuestion: {query}")
])
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert log summarizer. Given extracted triples, produce a concise summary focusing on key events, errors, and root causes."),
    ("human", "Triples:\n{triples}")
])

# ──────────────────────────────────────────────────────────────────────────
# Helpers for Neo4j (store triples & metadata)
MERGE_TRIPLE_CYPHER = textwrap.dedent("""
MERGE (s:Entity {value:$subject})
MERGE (o:Entity {value:$object})
MERGE (s)-[r:%s]->(o)
SET r.log_id=$log_id
""")

METADATA_CYPHER = textwrap.dedent("""
MERGE (l:Log {id:$log_id})
SET l.start=$start, l.end=$end
""")

def save_triples(log_id: str, triples: List[Dict]):
    with dr.session(database=DB_NAME) as sess:
        for t in triples:
            pred = re.sub(r"[^A-Za-z0-9]", "_", t["predicate"].upper())
            sess.run(MERGE_TRIPLE_CYPHER % pred,
                     subject=t["subject"], object=t["object"], log_id=log_id)


def save_metadata(log_id: str, start: str, end: str):
    with dr.session(database=DB_NAME) as sess:
        sess.run(METADATA_CYPHER, log_id=log_id, start=start, end=end)


def save_solution(error_hash: str, solution: str):
    with dr.session(database=DB_NAME) as sess:
        sess.run(
            "MERGE (e:Error {hash:$hash}) SET e.solution=$solution, e.solved=true",
            hash=error_hash, solution=solution,
        )


def fetch_triples(log_id: str) -> List[Tuple[str, str, str]]:
    with dr.session(database=DB_NAME) as sess:
        result = sess.run(
            "MATCH (s)-[r]->(o) WHERE r.log_id=$log_id RETURN s.value,type(r),o.value LIMIT 2000",
            log_id=log_id,
        )
        return [(r[0], r[1], r[2]) for r in result]

# ──────────────────────────────────────────────────────────────────────────
# TF-IDF store (in-memory) for raw lines and triples
class TextStore:
    def __init__(self):
        self.docs: List[str] = []
        self.ids:  List[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None

    def add(self, doc_id: str, text: str):
        self.ids.append(doc_id)
        self.docs.append(text)
        self.matrix = self.vectorizer.fit_transform(self.docs)

    def query(self, q: str, k: int = 10) -> List[str]:
        if not self.docs:
            return []
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix).flatten()
        top = sims.argsort()[::-1][:k]
        return [self.docs[i] for i in top if sims[i] > 0.05]

text_store = TextStore()

# ──────────────────────────────────────────────────────────────────────────
# Log parsing & ingestion (triples + raw lines + metadata)
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} [\d:,]+) (\w+) +\[(.*?)\] +\((.*?)\) (.*)$")

def infer_triples(line: str) -> List[Dict]:
    resp = LLMChain(llm=llm, prompt=TRIPLE_PROMPT).run({"log_line": line})
    try:
        triples = json.loads(resp)
        return triples if isinstance(triples, list) else []
    except Exception:
        return []


def process_log(text: str, filename: str):
    # Derive a stable log ID
    log_id = hashlib.sha1(filename.encode()).hexdigest()[:12]
    timestamps: List[str] = []
    triples_all: List[Dict] = []

    for idx, line in enumerate(tqdm(text.splitlines(), desc="Extracting triples")):
        if not line.strip():
            continue
        # Raw context ingestion
        text_store.add(f"{log_id}:raw:{idx}", line)
        # Timestamp capture
        m = TS_RE.match(line)
        if m:
            timestamps.append(m.group(1))
        # Triple extraction
        triples = infer_triples(line)
        triples_all.extend(triples)

    # Persist triples and raw context
    save_triples(log_id, triples_all)
    for t in triples_all:
        text_store.add(f"{log_id}:triple:{hashlib.sha1(json.dumps(t).encode()).hexdigest()}", json.dumps(t))

    # Compute and save metadata timeframe
    start = min(timestamps) if timestamps else ""
    end   = max(timestamps) if timestamps else ""
    save_metadata(log_id, start, end)

    return log_id, triples_all

# ──────────────────────────────────────────────────────────────────────────
# Summarization using triples

def summarize_log(triples: List[Tuple[str, str, str]]) -> str:
    flat = "\n".join([f"{s} {p} {o}" for s, p, o in triples])
    return LLMChain(llm=llm, prompt=SUMMARY_PROMPT).run({"triples": flat})

# ──────────────────────────────────────────────────────────────────────────
# Network visualization

def build_network(triples: List[Tuple[str, str, str]]):
    net = Network(height="600px", width="100%", directed=True)
    for s, p, o in triples:
        net.add_node(s, label=s)
        net.add_node(o, label=o)
        net.add_edge(s, o, label=p)
    return net


def show_network(net: Network):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp.name)
    st.components.v1.html(open(tmp.name).read(), height=650, scrolling=True)

# ──────────────────────────────────────────────────────────────────────────
# Context retrieval now includes timeframe

def retrieve_context(query: str) -> str:
    # TF-IDF raw and triple context
    tfidf_ctx   = text_store.query(query)
    # Graph-based context
    graph_ctx: List[str] = []
    with dr.session(database=DB_NAME) as sess:
        res = sess.run(
            "MATCH (s)-[r]->(o) WHERE s.value CONTAINS $q OR o.value CONTAINS $q RETURN s.value, type(r), o.value LIMIT 10",
            q=query,
        )
        graph_ctx = [f"{r[0]} {r[1]} {r[2]}" for r in res]
        # Fetch global timeframe
        tr = sess.run("MATCH (l:Log) RETURN min(l.start) AS start, max(l.end) AS end")
        rec = tr.single()
        if rec:
            timeframe = f"Timeframe: {rec['start']} to {rec['end']}"
        else:
            timeframe = ""
    return "\n".join([timeframe] + tfidf_ctx + graph_ctx)

# ──────────────────────────────────────────────────────────────────────────
# Streamlit UI layout
tabs = st.tabs(["📈 Log Graph Admin", "🕵️ Log Investigator"])

with tabs[0]:
    st.header("Log Graph Admin")
    uploaded = st.sidebar.file_uploader("Upload .log file", type=["log", "txt"])
    if uploaded:
        content = uploaded.read().decode(errors="ignore")
        with st.spinner("Processing log …"):
            log_id, triples = process_log(content, uploaded.name)
        st.success(f"Ingested {len(triples)} triples for log {log_id}")

    if "log_ids" not in st.session_state:
        with dr.session(database=DB_NAME) as sess:
            vals = sess.run("MATCH (l:Log) RETURN l.id LIMIT 1000")
            st.session_state.log_ids = [v[0] for v in vals]
    sel = st.sidebar.selectbox("Select log to explore", st.session_state.log_ids)
    if sel:
        data = fetch_triples(sel)
        df = pd.DataFrame(data, columns=["Subject","Predicate","Object"])
        st.subheader("Triples Table")
        st.table(df)
        st.subheader("Knowledge Graph")
        show_network(build_network(data))
        if st.button("Summarize log"):
            with st.spinner("Summarizing …"):
                summary = summarize_log(data)
            st.markdown("### Log Summary")
            st.write(summary)

with tabs[1]:
    st.header("Log Investigator")
    query = st.text_input("Ask anything about the logs")
    if st.button("Investigate") and query.strip():
        with st.spinner("Investigating …"):
            context = retrieve_context(query)
            report = LLMChain(llm=llm, prompt=INVESTIGATOR_PROMPT).run({"context": context, "query": query})
        st.markdown("### Investigation Report")
        st.markdown(report)
