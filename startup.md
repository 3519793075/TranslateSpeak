# [cite_start]项目需求文档：中文口播文本转英文翻译+语音小工具 [cite: 2]

## 1. 项目背景与目标
[cite_start]实现一个基于 Python 和 Streamlit 的 Web 小工具。运营人员将输入中文带货或短视频口播文案，系统需要快速生成地道的英文翻译，并调用语音接口生成可直接使用的英文音频 [cite: 9, 10]。

## 2. 技术栈
- **前端与框架**: `streamlit`
- **翻译大模型 API**: `openai` (用于文本翻译)
- [cite_start]**语音合成 API**: `requests` (直接调用 ElevenLabs REST API [cite: 20])
- **环境变量管理**: `python-dotenv`

## 3. 核心功能需求

### 3.1 输入模块
- [cite_start]提供一个文本输入框（`st.text_area`），支持用户输入中文口播文本 [cite: 14]。

### 3.2 翻译处理与输出
- 调用 OpenAI API，将中文翻译为英文。
- **Translation System Prompt 要求**: 翻译必须打破机器味。需运用地道的英语口语表达、恰当的俚语（Slang）以及符合特定受众群体的社会方言（Sociolects），使其读起来像真正的 Native Speaker 在做短视频带货。
- [cite_start]在页面上展示翻译后的英文文本，并利用 Streamlit 的原生代码块功能或 `st.code` 实现“一键复制” [cite: 16]。

### 3.3 语音生成与展示
- [cite_start]接收翻译好的英文文本，调用 ElevenLabs API 生成音频 [cite: 20]。
- [cite_start]**进阶：多音色并发对比 [cite: 24]**：
  - 预设 2 个风格迥异的 `voice_id`（例如一个热情洋溢的女声，一个沉稳有说服力的男声）。
  - 使用 Python 的 `concurrent.futures.ThreadPoolExecutor` 并发请求 ElevenLabs API，同时生成这两个音色的音频。
- [cite_start]**进阶：语音自然度优化 [cite: 23]**：在 API 请求参数中，指定模型为 `eleven_multilingual_v2`，并适度调低 `stability` 参数以增强口播的语气起伏。
- [cite_start]在前端并排（使用 `st.columns`）展示生成的两个音频播放器。Streamlit 的 `st.audio` 原生支持播放和下载功能 [cite: 17]。

### 3.4 历史记录功能
- [cite_start]使用 Streamlit 的 `st.session_state` 实现历史记录状态管理 [cite: 18]。
- 每次成功生成后，将原始中文、英文翻译及音频数据（bytes）保存到 Session State 中的列表中。
- [cite_start]在页面下方遍历该列表，渲染历史记录的文本和音频组件 [cite: 18]。

## 4. 文件结构与部署配置
请为我生成以下文件结构及代码：

1. **`app.py`**: 主程序逻辑，包含 UI 构建、API 请求函数（加入错误重试机制）和并发处理逻辑。
2. **`.env.example`**: 环境变量示例文件，包含 `OPENAI_API_KEY` 和 `ELEVENLABS_API_KEY` 字段。
3. **`requirements.txt`**: 项目依赖清单。
4. **`Dockerfile`** (用于云端容器化部署):
   - 基于 `python:3.10-slim` 构建。
   - 暴露 8501 端口。
   - 包含运行命令 `streamlit run app.py --server.address=0.0.0.0`。

## 5. 测试数据参考
请在代码调试阶段，使用以下示例文本验证功能是否正常闭环：
> [cite_start]"还在为割草头疼吗？太阳底下忙活大半天，又累又费劲儿，清理碎草还麻烦！给你推荐这个懒人神器：全自动割草机器人... 解放双手，不用再为割草浪费时间，轻松拥有整洁庭院，闭眼入不踩雷。" [cite: 31, 32]

## 6. AI 助手执行指令
请读取上述需求，一步步为我生成所需的所有代码文件。先输出 `requirements.txt` 和 `.env.example`，然后输出核心的 `app.py` 代码，最后输出 `Dockerfile`。确保代码包含详尽的中文注释，并妥善处理可能出现的 API 网络异常。