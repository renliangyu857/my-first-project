#  DeepResearch Agent: 基于 LangGraph 的科研论文助手

这是一个工业级的科研 Agent 项目，核心通过 **LangGraph** 构建复杂的推理链路，旨在实现自动化的论文调研、参数化检索与结构化信息提取。

##  项目特性
* **智能任务规划**：采用 Planner-Executor 架构，自动拆解并调度科研调研需求。
* **学术工具集成**：精准对接 Arxiv API，实现论文检索与结构化参数提取。
* **自动化量化评测**：内置基于 **BFCL V3** 标准的自动化测试套件，量化 Agent 的结构化保真度（Fidelity）与响应时延。

---

##  架构概览
项目采用状态机驱动：
1. **Planner Node**: 负责语义解析与工具参数预提取。
2. **Executor Node**: 执行 `arxiv_research_tool` 工具调用。
3. **Evaluation Harness**: 独立于业务逻辑的自动化“量化考场”脚本。

---

##  快速启动

### 1. 环境配置
在根目录下创建 `.env` 文件，并配置你的 API 密钥：
```env
OPENAI_API_KEY=你的密钥
BASE_URL=你的代理地址
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动交互式 UI (Streamlit)
```bash
streamlit run ui.py
```

### 4. 运行后端逻辑/主入口
```bash
python main.py
```

### 5. 执行自动化量化评测
```bash
python evaluation.py
```

---

##  评测方法论 (Harness Methodology)
本项目拒绝“体感评估”，采用针对科研场景设计的“地狱难度”测试题进行闭合量化：
* **核心指标**：参数提取保真度（Fidelity）、响应时延（Latency）、干扰项抗性。
* **意义**：通过该量化脚本，我们能精准定位模型在处理复杂指令时的参数溢出问题，为后续 Prompt 的迭代提供确定的数据支撑。