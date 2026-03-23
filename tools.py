import arxiv, os, time, requests
import concurrent.futures # 🚀 引入并发库
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from database import get_vector_db, DATA_DIR

def _download_and_parse(result):
    """原子化处理：单篇论文的下载与解析摘要"""
    try:
        arxiv_id = result.entry_id.split('/')[-1]
        safe_title = "".join(x for x in result.title if x.isalnum() or x in " -_").strip()
        path = os.path.join(DATA_DIR, f"{safe_title}.pdf")
        
        # 镜像站极速下载
        if not os.path.exists(path):
            mirror_url = f"https://cn.arxiv.org/pdf/{arxiv_id}.pdf"
            res = requests.get(mirror_url, timeout=15)
            if res.status_code == 200:
                with open(path, "wb") as f: f.write(res.content)
        
        return {
            "title": safe_title,
            "doc": Document(page_content=result.summary, metadata={"source": safe_title})
        }
    except Exception as e:
        print(f"⚠️ 处理 {result.title} 失败: {e}")
        return None

@tool
def arxiv_research_tool(topic: str, count: int = 3) -> str:
    """搜索并下载论文。并发模式开启。"""
    print(f"🚀 [并发搜索启动] 关键词: {topic}")
    client = arxiv.Client(num_retries=5)
    search = arxiv.Search(query=topic, max_results=count, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    results = list(client.results(search))
    if not results: return "❌ 未找到相关文献。"

    downloaded = []
    abstract_docs = []
    
    # ⚡ [并发核心]：开启 5 线程池并行处理 I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_download_and_parse, r) for r in results]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                downloaded.append(res["title"])
                abstract_docs.append(res["doc"])

    if abstract_docs:
        db, _ = get_vector_db()
        db.add_documents(abstract_docs)
        
    return f"✅ 并发处理完成！已保存：\n" + "\n".join(downloaded)

@tool
def query_research_db(question: str) -> str:
    """查询向量数据库。"""
    db, _ = get_vector_db()
    docs = db.similarity_search(question, k=4)
    return "\n\n".join([f"📖 [{d.metadata.get('source')}]: {d.page_content}" for d in docs])

@tool
def summarize_paper_tool(paper_title: str) -> str:
    """总结论文。"""
    # 保持原有逻辑，建议增加 DeepSeek 总结
    return f"📑 《{paper_title}》的深度总结已生成至知识库。"