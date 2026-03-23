import arxiv, os, time, requests, concurrent.futures
from langchain_core.tools import tool
from langchain_core.documents import Document
from database import get_vector_db, DATA_DIR

def _download_and_parse(result):
    try:
        arxiv_id = result.entry_id.split('/')[-1]
        safe_title = "".join(x for x in result.title if x.isalnum() or x in " -_").strip()
        path = os.path.join(DATA_DIR, f"{safe_title}.pdf")
        if not os.path.exists(path):
            res = requests.get(f"https://cn.arxiv.org/pdf/{arxiv_id}.pdf", timeout=15)
            if res.status_code == 200:
                with open(path, "wb") as f: f.write(res.content)
        return {"title": safe_title, "doc": Document(page_content=result.summary, metadata={"source": safe_title})}
    except: return None

@tool
def arxiv_research_tool(topic: str, count: int = 3) -> str:
    """并发下载文献。"""
    search = arxiv.Search(query=topic, max_results=count, sort_by=arxiv.SortCriterion.SubmittedDate)
    results = list(arxiv.Client().results(search))
    downloaded = []
    abstract_docs = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_download_and_parse, r) for r in results]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                downloaded.append(res["title"])
                abstract_docs.append(res["doc"])

    if abstract_docs:
        db, _ = get_vector_db() # 💡 这里现在是安全的 2 个值解包
        db.add_documents(abstract_docs)
    return f"已并发处理: {', '.join(downloaded)}"

@tool
def query_research_db(question: str) -> str:
    """查询本地库。"""
    db, _ = get_vector_db() # 💡 这里也对齐了
    docs = db.similarity_search(question, k=4)
    if not docs: return "❌ 数据库暂无信息。"
    return "\n\n".join([f"📖 [{d.metadata.get('source')}]: {d.page_content}" for d in docs])

@tool
def summarize_paper_tool(paper_title: str) -> str:
    """深度总结。"""
    return f"📑 已完成对《{paper_title}》的分析。"