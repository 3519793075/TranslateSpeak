# TranslateSpeak

中文口播文案 → 英文翻译 + 双音色语音合成，助力海外短视频内容快速产出。

## 功能

### 🌐 智能翻译

中文口播文案 → 地道英文脚本，由 **DeepSeek-V3** 调用专用 System Prompt 驱动：

- 输出 Native Speaker 级别的带货口播，包含地道的俚语（Slang）和社会方言（Sociolects）
- 保留原文语气节奏——痛点描述、价值主张、行动号召一步到位
- 有重试机制（最多 3 次，指数退避），网络波动不中断

### 🔊 双音色语音合成

调用 **ElevenLabs** `eleven_multilingual_v2` 模型，**并发**生成两路音频：

- **女声 + 男声** 同时请求（`ThreadPoolExecutor`），无需等待
- 可切换音色播放，对比不同风格的口播效果
- 支持 **在线播放 + 一键下载** MP3（44100 Hz / 128 kbps）

### 📖 逐句高亮播放

音频播放时，英文文本与语音 **实时同步**：

- 按句子拆分英文，根据字符数比例估算每句时间区间
- 播放时当前句子 **蓝色高亮 + 自动滚动**，已播放句子置灰
- **点击任意句子** 跳转到对应播放位置，便于精听反复打磨
- 语音切换时播放进度不丢失

### 📚 历史记录

每次生成自动保存，记录上限 **20 条**，可随时回溯：

- 保留：中文原文、英文翻译、双音色音频
- 支持历史音频在线播放和下载
- 一键清空全部历史

### 🛡 按钮防重复

防止误操作和重复调用 API：

- **10 秒冷却**：两次点击间隔必须 ≥ 10 秒，页面上显示倒计时
- **生成中锁定**：翻译/语音生成期间按钮禁用，防止并发冲突
- 两个限制为 **AND 关系**，任一触发即锁定按钮

## 技术栈

| 层 | 组件 |
|---|------|
| UI | Streamlit |
| 翻译 | DeepSeek API（OpenAI 兼容接口） |
| 语音 | ElevenLabs SDK（`eleven_multilingual_v2`） |
| 部署 | Docker + GitHub Actions（CI/CD 待接入） |

## 快速开始

### 1. 克隆仓库

```bash
git clone git@github.com:3519793075/TranslateSpeak.git
cd TranslateSpeak
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```ini
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key

# 可选：覆盖默认语音
# ELEVENLABS_VOICE_A_ID=xxx  # 女声
# ELEVENLABS_VOICE_B_ID=xxx  # 男声
```

- DeepSeek API Key → https://platform.deepseek.com/api_keys
- ElevenLabs API Key → https://elevenlabs.io/app/settings/api-keys

### 4. 启动

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

## Docker 部署

```bash
docker build -t translatespeak .
docker run -d -p 8501:8501 --env-file .env --name translatespeak translatespeak
```

## 项目结构

```
TranslateSpeak/
├── app.py              # 主程序（Streamlit UI + API 请求 + 逐句高亮）
├── requirements.txt    # Python 依赖
├── Dockerfile          # 容器化部署
├── .env.example        # 环境变量模板
└── logs/               # 运行日志（按时间戳命名）
```

## 使用说明

1. 在 **🎙️ 语音生成** 标签页粘贴中文口播文案
2. 点击 **翻译并生成语音**，等待翻译和双音色语音并发生成
3. 在播放器中切换女声/男声，跟随逐句高亮预览效果，或点击句子精听
4. 切换到 **📚 历史记录** 标签页回溯过往结果，或下载音频文件
5. 同一文案 10 秒内无需重复点击，按钮会自动进入冷却保护
