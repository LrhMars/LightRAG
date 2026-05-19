# 基于超图的交通基建多智能体检索增强系统 — 项目交接说明

> **仓库定位**：在 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 开源框架上的**深度定制版本**，面向港航/水运工程规范类文档的知识构建与智能问答。  
> **交接读者**：接手开发、部署、运维或二次扩展的同事。  
> **上游说明**：通用 LightRAG 能力见仓库内 `README-zh.md`；本文只描述**本项目的增量与运行方式**。

---

## 1. 项目做什么

| 维度 | 说明 |
|------|------|
| **业务场景** | 港口与航道工程相关的施工规范、安全技术规程、合同与管理条款等文本的智能问答 |
| **核心痛点** | 传统向量 RAG 难以稳定表达 **数值阈值、AND/OR 联合条件、多因一果、条款溯源** |
| **技术路线** | 离线：层级切片 + 并行 LLM 抽取（实体/关系/超边）+ 三层超图索引；在线：ReAct 多 Agent + 动态路由 + 向量/图谱/超图多模式检索 |
| **交付形态** | FastAPI + Neo4j + JSON KV + NanoVectorDB + React WebUI；可选 MCP Server（Cursor/Claude Desktop） |

---

## 2. 系统架构（简图）

```
文档上传 → KG Pipeline（分块→抽取→合并/消歧→超边物化→入库）
                ↓
    data/rag_storage（KV + VDB + kg_pipeline_runs）
    Neo4j（实体/关系 + Hub 超边）
                ↓
用户提问 → Orchestrator 路由 → Fast / KG / Hyper Agent（或 MultiMode 融合）
                ↓
         答案 + 引用证据（chunk / section_path / 图路径 / hub）
```

更细的工程说明见：`docs/架构实现方案.md`。

---

## 3. 目录结构（接手必看）

```
LightRAG/                          # 本项目根目录（下文路径均相对此目录）
├── .env                           # 环境变量（含 LLM/Embedding/Neo4j，勿提交密钥到公网）
├── docker-compose.yml             # lightrag + neo4j 编排与热挂载清单
├── config.ini                     # 服务配置
├── mcp_server.py                  # FastMCP 对外 MCP Server（stdio）
├── data/
│   ├── inputs/                    # 文档入队/上传目录
│   └── rag_storage/               # 工作目录：KV、VDB、日志、pipeline 留痕
├── lightrag/                      # 核心 Python 包（大量定制）
│   ├── api/                       # FastAPI：lightrag_server.py、各 router
│   ├── agent/                     # Orchestrator、Fast/KG/Hyper、MultiMode、tools
│   ├── kg_pipeline/               # 嵌入式 KG 流水线 runner、merge、storage_sink
│   ├── operate.py                 # naive_query、kg_query、local/global 检索
│   ├── chunking.py                # Markdown 层级感知切片
│   └── kg/                        # Neo4j、JsonKV、NanoVectorDB 等存储实现
├── lightrag_webui/                # 前端源码（需 build 到 lightrag/api/webui）
├── lightrag_kg_pipeline_direct/     # 独立 KG 流水线（可选，与嵌入式 pipeline 并存）
├── scripts/                       # 运维/测试脚本（清库、报告、评测等）
└── docs/                          # 架构、演讲稿、面试清单、模拟面试等文档
```

---

## 4. 环境要求

- **Docker Desktop**（Windows 上需先启动，否则 Neo4j 未就绪会导致 lightrag 反复重启）
- **Python 3.10+**（宿主机跑脚本时）
- **Node.js**（仅在前端需要重新 build WebUI 时）
- **外部服务**：OpenAI 兼容的 **LLM API**、**Embedding API**（当前 `.env` 示例为硅基流动 bge-m3 + Qwen2.5，请按实际替换）

---

## 5. 快速启动（推荐：Docker Compose）

### 5.1 配置环境变量

1. 复制并编辑 `.env`（勿将真实 API Key 提交到 Git）。
2. **容器内路径**必须与 compose 一致：
   - `WORKING_DIR=/app/data/rag_storage`
   - `INPUT_DIR=/app/data/inputs`
3. **Neo4j**：compose 内 lightrag 使用 `NEO4J_URI=neo4j://neo4j:7687`（由 `docker-compose.yml` 的 `environment` 注入，覆盖 `.env` 里指向 `127.0.0.1` 的配置）。
4. **LLM / Embedding**：修改 `LLM_BINDING_HOST`、`LLM_MODEL`、`EMBEDDING_BINDING_HOST`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`（bge-m3 为 1024）。

### 5.2 构建镜像与前端（首次）

```powershell
cd LightRAG
# 可选：编译 WebUI 到 lightrag/api/webui
.\scripts\prepare_local_docker_build.ps1
# 或：cd lightrag_webui && npm install && npm run build

docker compose build lightrag
docker compose up -d
```

### 5.3 访问入口

| 入口 | 地址 |
|------|------|
| WebUI / API | http://localhost:9621 |
| Swagger | http://localhost:9621/docs |
| Neo4j Browser | http://localhost:7474 |

### 5.4 开发时改代码

`docker-compose.yml` 已对 `kg_pipeline/`、`agent/`、`operate.py`、`chunking.py` 等做了**选择性 volume 挂载**。改 Python 后：

```powershell
docker compose restart lightrag
```

**注意**：不要挂载整个 `./lightrag`，会覆盖镜像内已构建的 `api/webui`。

---

## 6. 核心业务流程

### 6.1 离线：文档入库（KG Pipeline）

1. 将 Markdown/文本放入 `data/inputs/` 或通过 WebUI/API 上传。
2. 触发文档处理 / KG Pipeline（前端或 `POST` 文档相关接口，见 Swagger）。
3. 流水线阶段（嵌入式 `lightrag/kg_pipeline/runner.py`）：
   - **Stage 0**：`chunking_by_markdown_hierarchy`（章节优先 + token 兜底 + `section_path` + overlap）
   - **Stage 1**：并行 LLM 抽取（`asyncio.gather` + `Semaphore` 限并发），输出 Entity / Relation / Hyperedge
   - **合并/消歧**：`merge.py` + 可选 `_run_entity_disambiguation`（opensource / llm / hybrid）
   - **超边物化**：Neo4j Hub + `kv_store_hyperedge_index.json` + `vdb_hyperedges.json`
   - **Sink**：`storage_sink.py` 写入图与多路向量库
4. **留痕目录**：`data/rag_storage/kg_pipeline_runs/{doc_id}/`  
   生成 HTML 报告：`python scripts/pipeline_report.py`

默认 **跳过本体统计与精炼**（`skip_ontology_refine=True`），无需单独维护「本体库」即可跑通主链路。

### 6.2 在线：问答

| 方式 | 说明 |
|------|------|
| **WebUI** | 上传文档、查看图谱、对话（支持流式） |
| **Agent API** | `POST /agent/chat`、`POST /agent/chat/stream`（SSE） |
| **通用查询** | `query_routes`：`mode=naive|local|global|hybrid|mix` |
| **MCP** | 独立进程 `python mcp_server.py`，供 Cursor 等客户端配置 |

**Agent 路由（简述）**：

- **FastAgent**：向量召回为主，简单事实/定义。
- **KGReasoningAgent**：`kg_query` / `entity_lookup`，多跳与溯源。
- **HyperAgent**：优先 `hyperedge_query` / `hyper_query`，联合约束类问题。
- **MultiMode（Universal）**：`QueryPlanner` 选模式 → 并行 Naive / KG Local / KG Global / HyperRAG → 融合答案。

ReAct 循环：Thought → 工具调用（Function Calling）→ Observation；各 Agent 有 **max_steps** 与工具白名单。

---

## 7. 数据与存储说明

### 7.1 工作目录 `data/rag_storage`

| 类型 | 典型文件 |
|------|----------|
| KV | `kv_store_text_chunks.json`、`kv_store_doc_status.json`、`kv_store_hyperedge_index.json` 等 |
| 向量库 | `vdb_chunks.json`、`vdb_entities.json`、`vdb_relationships.json`、`vdb_hyperedges.json` |
| 日志 | `lightrag.log` |
| 流水线留痕 | `kg_pipeline_runs/` |

### 7.2 Neo4j

- 实体、二元关系、**Hub 超边**（星扩展：`HUB_MEMBER` / `HUB_RESULT`）。
- 数据卷：`neo4j_data`（compose 命名卷）。

### 7.3 路径一致性（常见坑）

| 场景 | 要求 |
|------|------|
| 容器内服务 | `WORKING_DIR=/app/data/rag_storage`，对应宿主 `./data/rag_storage` 挂载 |
| 宿主直接跑 `scripts/*.py` | `.env` 中 `WORKING_DIR` 应指向 **同一物理目录**（如 `./data/rag_storage`），不要用与 compose 不一致的绝对路径 |

否则会出现「脚本写入成功、容器内查不到向量库」的假成功。

---

## 8. 配置项速查

| 变量 | 含义 |
|------|------|
| `LLM_BINDING_HOST` / `LLM_MODEL` | 抽取与问答用大模型 |
| `EMBEDDING_BINDING_HOST` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 向量化（须与模型维度一致） |
| `LIGHTRAG_GRAPH_STORAGE` | 默认 `Neo4JStorage` |
| `LIGHTRAG_VECTOR_STORAGE` | 默认 `NanoVectorDBStorage` |
| `LLM_NO_THINK` | Qwen 系关闭 thinking 时设为 `true` |
| `kg_pipeline_disambiguation_mode` | `opensource`（默认）/ `llm` / `hybrid` |
| `SUMMARY_LANGUAGE` | 建议 `Chinese` |

存储与并发相关实现见：`lightrag/kg/nano_vector_db_impl.py`（批量 embedding）、`lightrag/utils.py`（`safe_vdb_operation_with_exception`）。

---

## 9. 常用运维脚本

```powershell
cd LightRAG

# 预览/清空全部数据（Neo4j + KV + VDB + pipeline_runs）
python scripts/clear_all_data.py              # 仅预览
python scripts/clear_all_data.py --confirm    # 执行删除

# 流水线 HTML 报告（需已有 kg_pipeline_runs）
python scripts/pipeline_report.py

# 工具与 Agent 单测（开发调试用）
python scripts/test_tools.py
python scripts/test_orchestrator.py

# 评测（若有评测集配置）
python scripts/eval_pipeline.py
```

---

## 10. MCP Server 交接

- **文件**：`mcp_server.py`（FastMCP，**stdio** 传输）。
- **对外工具（聚合型，非 Agent 内全部细粒度工具）**：
  - `search` — 综合检索（可传 `mode`: hybrid/local/global 等）
  - `get_regulation` — 按章节路径取原文
  - `query_entity` — 实体属性与邻接关系
  - `list_knowledge_bases` — 列出数据集
- **启动**：`python mcp_server.py`（需能加载 `.env` 与 `WORKING_DIR`）。
- **Cursor 配置示例**（路径改为本机实际路径）：

```json
{
  "mcpServers": {
    "hyperlograg": {
      "command": "python",
      "args": ["D:/path/to/LightRAG/mcp_server.py"],
      "env": {}
    }
  }
}
```

Agent 内部另有完整 Function Calling 工具集，见 `lightrag/agent/tools.py`（向量、KG、章节、超图、计算、mode_* 等）。

---

## 11. API 一览（定制相关）

| 前缀 | 模块 | 说明 |
|------|------|------|
| `/documents` | `document_routes` | 上传、处理状态、KG Pipeline 触发与进度 |
| `/query` | `query_routes` | LightRAG 标准查询（naive/local/global/…） |
| `/graph` | `graph_routes` | 图数据查询与可视化 |
| `/datasets` | `dataset_routes` | 数据集管理（若启用） |
| `/agent` | `agent_routes` | 多智能体对话、SSE、会话管理 |

详细参数以运行中 **Swagger** 为准：`http://localhost:9621/docs`。

---

## 12. 测试与文档索引

| 文档 | 用途 |
|------|------|
| `docs/架构实现方案.md` | 工程架构、风险、演进路线 |
| `docs/项目介绍清单.md` | 流程图 + 分模块说明 |
| `docs/十分钟演讲稿.md` | 对外介绍口径 |
| `docs/面试_项目实现细节清单.md` | 实现细节清单 |
| `docs/模拟面试_问题清单.md` / `docs/模拟面试_回答清单.md` | 面试问答 |
| `qa_dataset.jsonl` | 示例问答集（评测/演示） |

自动化测试：`tests/test_kg_pipeline_runner.py` 等；端到端脚本 `scripts/test_e2e_ganghang.py`。

---

## 13. 已知问题与排障

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| lightrag 容器反复重启 | Docker/Neo4j 未启动 | 先开 Docker Desktop，`docker compose up neo4j` 等 healthy |
| 入库成功但查询无结果 | `WORKING_DIR` 路径不一致 | 统一宿主与容器挂载目录 |
| LLM 502/超时 | 外部 API 不可用 | 检查 `.env` 中 LLM/Embedding 地址与 Key |
| WebUI 只有 Swagger | `api/webui` 为空 | 执行 `prepare_local_docker_build.ps1` 或 npm build |
| 向量库文件不存在 | 未触发落盘回调 | 确认 pipeline 结束有 flush；参考历史修复 `_insert_done` / `index_done_callback` |
| 前端仍显示旧「本体」 | 仅 UI 缓存或旧 datasets 文件 | 清库时注意是否保留 `kv_store_datasets.json`；默认 pipeline 已 skip 本体阶段 |

日志：`data/rag_storage/lightrag.log`。

---

## 14. 交接清单（Checklist）

- [ ] 获取 `.env` 与 Neo4j 账号（勿泄露到公网仓库）
- [ ] 确认 LLM/Embedding 服务可用及配额
- [ ] `docker compose up -d` 能访问 9621 与 Neo4j
- [ ] 上传 1 份样例文档跑通 Pipeline，检查 `kg_pipeline_runs` 与 `vdb_*.json`
- [ ] WebUI 或 `/agent/chat` 能问答，联合约束类问题走 Hyper 路径
- [ ] 阅读 `docs/架构实现方案.md` 与本文第 7 节「路径一致性」
- [ ] 知晓清库脚本 `scripts/clear_all_data.py` 的影响范围
- [ ] （可选）配置 MCP 与 Cursor 联调

---

## 15. 联系与版本

- **基础框架**：LightRAG（HKUDS），本仓库为 fork/定制延续，合并上游时注意 `operate.py`、`lightrag.py` 等冲突。
- **定制主线**：超图索引 + KG Pipeline 重构 + 多 Agent 路由 + MCP 聚合接口 + 流水线留痕与评测脚本。

如有疑问，优先查 `docs/` 下文档与 Swagger；代码入口建议从 `lightrag/api/lightrag_server.py`、`lightrag/kg_pipeline/runner.py`、`lightrag/agent/orchestrator.py` 读起。

---

*最后更新：交接文档初版。部署密钥、内网 IP 以团队内部配置为准，请勿写入公开 README。*
