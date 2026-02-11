# RAG Chatbot MVP (Integrated)

## 快速运行（本地）
1. 创建/进入虚拟环境并激活（推荐）
2. 安装依赖： `pip install -r requirements.txt`
3. 运行： `streamlit run app.py`
4. 打开浏览器访问 `http://localhost:8501`

## OpenAI 配置
- 若要使用 OpenAI 生成：请在环境变量中设置 `OPENAI_API_KEY`。
  - Linux/macOS: `export OPENAI_API_KEY="sk-..."`
  - Windows(PS): `setx OPENAI_API_KEY "sk-..."`

## 部署（Docker）
- 构建镜像并运行：
  - `docker build -t rag-chatbot-mvp:latest .`
  - `docker run -p 8501:8501 rag-chatbot-mvp:latest`

## 线上运行（PaaS）
- 本项目容器已支持读取平台注入的 `PORT` 环境变量（例如 Railway/Render/Fly.io）。
- 若平台自动分配端口，无需手动改 `Dockerfile`，直接部署即可。
- 若本地模拟动态端口，可执行：
  - `docker run -e PORT=7860 -p 7860:7860 rag-chatbot-mvp:latest`

## 费用估算
- 本项目会在调用 OpenAI 前估算 token 与费用（基于 `tiktoken` 或经验估算与 `MODEL_PRICE_PER_1K`）。
- 请定期核对 OpenAI 官方定价并更新 `MODEL_PRICE_PER_1K`。
