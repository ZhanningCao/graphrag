# GraphRAG 焊接知识图谱问答系统 — 环境搭建与运行指南

> 本指南同时覆盖 **Windows 本机** 和 **Linux 服务器（AutoDL 等）** 两种环境。

## 一、前置条件

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.11.x | 推荐 3.11.9，不支持 3.14 |
| Ollama | 最新版 | 本地 LLM 推理服务 |

---

## 二、项目结构说明

```
graphrag/                          ← 工作根目录
├── QA.jsonl                       ← QA 数据集（问题+标准答案）
├── run_one_query.py               ← 单条查询脚本
├── run_queries.py                 ← 预定义多条查询脚本
├── batch_local_search.py          ← 批量 Local Search（核心脚本）
├── local_search_qa_ollama_batch.py← 两阶段批量查询（优化版）
├── analyze_results.py             ← 结果质量分析
├── evaluate.py                    ← 评估脚本（关键词召回 + ROUGE-L）
├── test_pcst*.py                  ← PCST 子图检索测试脚本
├── graph_database/                ← GraphRAG 数据库目录
│   ├── settings.yaml              ← ⭐ 核心配置文件
│   ├── input/                     ← 输入数据（all.chunks.json）
│   ├── output/                    ← 索引输出（graph.graphml 等）
│   │   └── lancedb/               ← 向量数据库
│   └── prompts/                   ← 提示词模板
├── graphrag-main/                 ← GraphRAG v3.0.5 源码
│   └── packages/                  ← 各子包源码
│       ├── graphrag/
│       ├── graphrag-cache/
│       ├── graphrag-chunking/
│       ├── graphrag-common/
│       ├── graphrag-input/
│       ├── graphrag-llm/
│       ├── graphrag-storage/
│       └── graphrag-vectors/
└── pcst_results/                  ← PCST 查询结果输出
```

---

## 三、环境搭建步骤

---

### 步骤 1：安装 Ollama 并拉取模型

<details>
<summary><b>Windows 本机</b></summary>

```powershell
# 从 https://ollama.com 下载 Windows 安装包安装
# 安装完成后自动启动服务，如需手动启动：
ollama serve

# 拉取模型（另开终端）
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:4b
```

</details>

<details>
<summary><b>⭐ Linux 服务器（AutoDL 等）</b></summary>

```bash
# ====== 1) 安装 Ollama ======
curl -fsSL https://ollama.com/install.sh | sh

# ====== 2) 后台启动 Ollama 服务 ======
# 方法一：用 nohup 后台运行（推荐）
nohup ollama serve > /tmp/ollama.log 2>&1 &

# 方法二：用 systemctl（如果有 systemd）
# sudo systemctl start ollama

# 等几秒让服务启动，然后验证
sleep 3
curl http://localhost:11434/api/tags
# 应该返回 {"models":[]}

# ====== 3) 拉取模型 ======
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:4b

# 验证模型已就绪
curl http://localhost:11434/api/tags
# 应看到两个模型
```

**AutoDL 注意事项**：
- 如果服务器**没有外网**，无法直接 `curl` 安装。需要在有网时提前下载，或开学术加速。
- AutoDL 默认没有 systemd，用 `nohup` 方式启动。
- 模型文件默认存在 `~/.ollama/models`，首次拉取需要下载约 4GB。
- 如果 GPU 显存不够，可换用更小的模型（如 `qwen3:0.6b`），同时修改 `settings.yaml`。

</details>

验证 Ollama 正常运行：
```bash
curl http://localhost:11434/api/tags
# 应该能看到 qwen3:1.7b 和 qwen3-embedding:4b 两个模型
```

---

### 步骤 2：上传项目文件（仅服务器）

如果在服务器上运行，需要把项目文件上传到服务器：

```bash
# 方式一：在本地压缩后通过 AutoDL 文件管理上传
# 本地：把 graphrag 文件夹打成 zip
# AutoDL 网页：点"文件" → 上传到 /root/autodl-tmp/

# 方式二：scp 上传
scp graphrag.zip root@<服务器IP>:/root/autodl-tmp/

# 服务器上解压
cd /root/autodl-tmp
unzip graphrag.zip
```

---

### 步骤 3：创建 Python 虚拟环境

<details>
<summary><b>Windows 本机</b></summary>

```powershell
cd d:\SEU\SRTP\graphrag\graphrag
& "C:\Users\18428\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
```

</details>

<details>
<summary><b>⭐ Linux 服务器</b></summary>

```bash
cd ~/autodl-tmp/graphrag

# 查看系统 Python 版本
python3 --version
# 如果是 3.11.x 就直接用，否则需要安装 3.11

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate
```

如果系统 Python 不是 3.11：
```bash
# Ubuntu/Debian：
sudo apt update && sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate

# AutoDL 一般已有 conda，也可用 conda：
conda create -n graphrag python=3.11 -y
conda activate graphrag
```

</details>

---

### 步骤 4：安装 GraphRAG 及所有子包

必须**按依赖顺序**安装所有子包，最后装主包。

<details>
<summary><b>Windows 本机（PowerShell）</b></summary>

```powershell
$pip = ".\.venv\Scripts\pip.exe"
$pkg = ".\graphrag-main\packages"

& $pip install -e "$pkg\graphrag-common"
& $pip install -e "$pkg\graphrag-cache"
& $pip install -e "$pkg\graphrag-chunking"
& $pip install -e "$pkg\graphrag-input"
& $pip install -e "$pkg\graphrag-llm"
& $pip install -e "$pkg\graphrag-storage"
& $pip install -e "$pkg\graphrag-vectors"
& $pip install -e "$pkg\graphrag"
```

</details>

<details>
<summary><b>⭐ Linux 服务器（Bash）</b></summary>

```bash
# 确保已激活虚拟环境
source .venv/bin/activate
# 或者 conda activate graphrag

PKG=./graphrag-main/packages

pip install -e $PKG/graphrag-common
pip install -e $PKG/graphrag-cache
pip install -e $PKG/graphrag-chunking
pip install -e $PKG/graphrag-input
pip install -e $PKG/graphrag-llm
pip install -e $PKG/graphrag-storage
pip install -e $PKG/graphrag-vectors
pip install -e $PKG/graphrag
```

</details>

### 步骤 5：验证安装

```bash
python -c "import graphrag; print('graphrag OK')"
python -c "import lancedb; import litellm; import networkx; print('All OK')"
```

---

## 四、配置文件修改（⚠️ 关键步骤）

核心配置在 `graph_database/settings.yaml`，**换机器后必须修改路径**：

### 4.1 修改向量数据库路径

```bash
# 查看你的 lancedb 实际路径
ls graph_database/output/lancedb/
# 应该看到 default-entity-description.lance/ 和 default-text_unit-text.lance/
```

编辑 `graph_database/settings.yaml`，找到 `vector_store` 部分：

**Windows 路径格式**：
```yaml
vector_store:
  type: lancedb
  db_uri: "d:\\SEU\\SRTP\\graphrag\\graphrag\\graph_database\\output\\lancedb"
```

**Linux 路径格式**：
```yaml
vector_store:
  type: lancedb
  db_uri: "/root/autodl-tmp/graphrag/graph_database/output/lancedb"
```

### 4.2 修改脚本中的硬编码路径（仅 Linux）

`run_one_query.py` 和 `run_queries.py` 中有 Windows 路径硬编码，需要改：

```bash
# run_one_query.py 中修改这两行：
# os.chdir(r"d:\SEU\SRTP\graphrag\graphrag\graph_database")
#   → 改为
# os.chdir("/root/autodl-tmp/graphrag/graph_database")

# root = r"d:\SEU\SRTP\graphrag\graphrag\graph_database"
#   → 改为
# root = "/root/autodl-tmp/graphrag/graph_database"

# 用 sed 快速替换（一条命令搞定）：
sed -i 's|d:\\\\SEU\\\\SRTP\\\\graphrag\\\\graphrag\\\\graph_database|/root/autodl-tmp/graphrag/graph_database|g' run_one_query.py
```

> **batch_local_search.py 不需要改**，因为它的路径通过命令行参数传入。

### 4.3 LLM 模型配置（一般不用改）

```yaml
completion_models:
  default_completion_model:
    model: qwen3:1.7b                    # Ollama 聊天模型
    api_base: http://localhost:11434/v1   # Ollama OpenAI 兼容接口
    api_key: Ollama                       # 随意填，Ollama 不验证

embedding_models:
  default_embedding_model:
    model: qwen3-embedding:4b             # Ollama Embedding 模型
    api_base: http://localhost:11434/v1
```

### 4.4 PCST 子图检索（在 local_search 配置段）

```yaml
local_search:
  use_pcst: true           # true=启用 PCST 子图检索
  pcst_top_k_nodes: 10
  pcst_top_k_edges: 10
  pcst_cost_per_edge: 0.5
```

---

## 五、运行方式

<details>
<summary><b>Windows 本机（PowerShell）</b></summary>

```powershell
# 定义快捷变量
$py = "d:\SEU\SRTP\graphrag\graphrag\.venv\Scripts\python.exe"
cd d:\SEU\SRTP\graphrag\graphrag

# 单条查询
& $py run_one_query.py "QIROX焊接机器人哪些型号属于QRC系列？" result.txt

# 批量查询
& $py batch_local_search.py `
  --root graph_database `
  --queries_file QA.jsonl `
  --out_dir pcst_results `
  --query_col input `
  --max_questions 0

# 分析结果
& $py analyze_results.py
```

</details>

<details>
<summary><b>⭐ Linux 服务器（Bash）</b></summary>

```bash
# 激活环境
cd ~/autodl-tmp/graphrag
source .venv/bin/activate

# ---- 单条查询 ----
python run_one_query.py "QIROX焊接机器人哪些型号属于QRC系列？" result.txt

# ---- 批量查询（⭐ 主要方式）----
python batch_local_search.py \
  --root graph_database \
  --queries_file QA.jsonl \
  --out_dir pcst_results \
  --query_col input \
  --max_questions 0

# ---- 分析结果 ----
python analyze_results.py

# ---- PCST 对比测试（可选）----
python test_pcst.py "你的问题"
```

**后台运行长时间任务**（防止 SSH 断开后中断）：
```bash
# 使用 nohup 后台运行批量查询
nohup python batch_local_search.py \
  --root graph_database \
  --queries_file QA.jsonl \
  --out_dir pcst_results \
  --query_col input \
  --max_questions 0 \
  > batch_output.log 2>&1 &

# 查看进度
tail -f batch_output.log
```

</details>

---

## 六、常见问题排错

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `ollama: command not found` | 服务器没装 Ollama | `curl -fsSL https://ollama.com/install.sh \| sh` |
| `ModuleNotFoundError: No module named 'graphrag'` | 没激活虚拟环境 | `source .venv/bin/activate` 或用完整路径 |
| `'LanceDBVectorStore' ... no attribute 'document_collection'` | `settings.yaml` 中 `db_uri` 路径不对 | 改为实际 lancedb 目录的绝对路径 |
| `Connection refused localhost:11434` | Ollama 没启动 | `nohup ollama serve > /tmp/ollama.log 2>&1 &` |
| 查询超时/无响应 | Ollama 模型推理慢 | 耐心等待（30-120 秒），或换更强 GPU |
| `Activate.ps1 禁止运行`（Windows） | PowerShell 执行策略 | 不用 activate，直接用完整路径调 python |
| 换电脑/服务器后跑不起来 | 路径硬编码 | 修改 `settings.yaml` 的 `db_uri` 和脚本中的路径 |
| SSH 断了任务中断 | 前台进程被杀 | 用 `nohup` 或 `tmux`/`screen` 运行 |

---

## 七、完整命令序列

### A. Linux 服务器（AutoDL）— 从零到跑起来

```bash
# ====== 0. 安装 Ollama ======
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 3
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:4b

# ====== 1. 解压项目（如已上传 zip）======
cd ~/autodl-tmp
unzip graphrag.zip
cd graphrag

# ====== 2. 创建虚拟环境 ======
python3 -m venv .venv
source .venv/bin/activate

# ====== 3. 安装依赖 ======
PKG=./graphrag-main/packages
pip install -e $PKG/graphrag-common
pip install -e $PKG/graphrag-cache
pip install -e $PKG/graphrag-chunking
pip install -e $PKG/graphrag-input
pip install -e $PKG/graphrag-llm
pip install -e $PKG/graphrag-storage
pip install -e $PKG/graphrag-vectors
pip install -e $PKG/graphrag

# ====== 4. 验证 ======
python -c "import graphrag; print('OK')"

# ====== 5. 修改配置路径 ======
# 编辑 graph_database/settings.yaml，把 db_uri 改为：
#   "/root/autodl-tmp/graphrag/graph_database/output/lancedb"
# 可以用 sed 一键替换：
sed -i 's|d:\\\\SEU\\\\SRTP\\\\graphrag\\\\graphrag\\\\graph_database\\\\output\\\\lancedb|/root/autodl-tmp/graphrag/graph_database/output/lancedb|g' graph_database/settings.yaml

# 修改 run_one_query.py 的路径（如需使用该脚本）：
sed -i 's|d:\\\\SEU\\\\SRTP\\\\graphrag\\\\graphrag\\\\graph_database|/root/autodl-tmp/graphrag/graph_database|g' run_one_query.py

# ====== 6. 运行批量查询 ======
python batch_local_search.py \
  --root graph_database \
  --queries_file QA.jsonl \
  --out_dir pcst_results \
  --query_col input \
  --max_questions 0

# ====== 7. 分析结果 ======
python analyze_results.py
```

### B. Windows 本机 — 从零到跑起来

```powershell
# ====== 0. 启动 Ollama（另开终端）======
ollama serve
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:4b

# ====== 1. 创建虚拟环境 ======
cd d:\SEU\SRTP\graphrag\graphrag
& "C:\Users\18428\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv

# ====== 2. 安装依赖 ======
$pip = ".\.venv\Scripts\pip.exe"
$pkg = ".\graphrag-main\packages"
& $pip install -e "$pkg\graphrag-common"
& $pip install -e "$pkg\graphrag-cache"
& $pip install -e "$pkg\graphrag-chunking"
& $pip install -e "$pkg\graphrag-input"
& $pip install -e "$pkg\graphrag-llm"
& $pip install -e "$pkg\graphrag-storage"
& $pip install -e "$pkg\graphrag-vectors"
& $pip install -e "$pkg\graphrag"

# ====== 3. 验证 ======
$py = ".\.venv\Scripts\python.exe"
& $py -c "import graphrag; print('OK')"

# ====== 4. 运行批量查询 ======
& $py batch_local_search.py `
  --root graph_database `
  --queries_file QA.jsonl `
  --out_dir pcst_results `
  --query_col input `
  --max_questions 0

# ====== 5. 分析结果 ======
& $py analyze_results.py
```
