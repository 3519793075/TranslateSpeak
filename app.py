"""
中文口播文案 → 英文翻译 + 语音合成 小工具
============================================
基于 Streamlit 构建的 Web 应用。
运营人员输入中文带货/短视频口播文案，
系统通过 DeepSeek 翻译为地道英文，
再通过 ElevenLabs 并发生成双音色英文音频。
"""

import base64
import json
import os
import re
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from openai import OpenAI

# ---------------------------------------------------------------------------
# 日志配置（控制台 + 按启动时间戳命名的日志文件）
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"app_{_start_timestamp}.log"

log_format = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 文件 Handler
_file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(log_format)

# 控制台 Handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(log_format)

# 根 logger 配置
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)
_root_logger.addHandler(_file_handler)
_root_logger.addHandler(_console_handler)

logger = logging.getLogger(__name__)

# 启动签名
logger.info("=" * 60)
logger.info("TranslateSpeak 启动 — 日志文件: %s", LOG_FILE.name)
logger.info("=" * 60)

# ---------------------------------------------------------------------------
# 环境变量加载
# ---------------------------------------------------------------------------
load_dotenv()

DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

# —— ElevenLabs 语音配置 ——
# 通过环境变量覆盖 voice_id，未设置时使用默认值。
# 示例: 在 .env 中添加 ELEVENLABS_VOICE_A_ID=xxx / ELEVENLABS_VOICE_B_ID=xxx
DEFAULT_VOICE_A = "3KRnY9b7OxRLT0Nf4gGj"   # 默认女声
DEFAULT_VOICE_B = "j6n2GrcbiXsTbSd0LnFI"   # 默认男声
VOICE_A_ID = os.getenv("ELEVENLABS_VOICE_A_ID", DEFAULT_VOICE_A)
VOICE_B_ID = os.getenv("ELEVENLABS_VOICE_B_ID", DEFAULT_VOICE_B)

ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"

# —— DeepSeek 翻译配置 ——
# DeepSeek API 兼容 OpenAI SDK，只需替换 base_url 和 api_key
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_TRANSLATION_MODEL = "deepseek-chat"  # DeepSeek-V3 最新对话模型

# —— 历史记录配置 ——
MAX_HISTORY = 20  # 最多保留的历史记录条数，防止内存无限增长

# —— 重试配置 ——
MAX_RETRIES = 3
RETRY_DELAY_SEC = 1.5  # 指数退避的基数

# —— 测试用示例文本 ——
SAMPLE_TEXT = (
    "还在为割草头疼吗？太阳底下忙活大半天，又累又费劲儿，"
    "清理碎草还麻烦！给你推荐这个懒人神器：全自动割草机器人... "
    "解放双手，不用再为割草浪费时间，轻松拥有整洁庭院，闭眼入不踩雷。"
)

# —— 翻译 System Prompt ——
TRANSLATION_SYSTEM_PROMPT = (
    "You are a world-class copywriter and translator specializing in short-form video "
    "commerce scripts (like TikTok Shop, Instagram Reels, live-selling).\n\n"
    "Translate the given Chinese script into English that sounds like a NATIVE English "
    "speaker is doing a live-selling pitch. The output must:\n"
    "- Sound 100% natural and conversational — as if spoken, not read.\n"
    "- Use authentic colloquial expressions, appropriate slang, and sociolects that "
    "match the target audience (e.g., Gen Z / millennial shoppers).\n"
    "- Keep the energy and rhythm of short-video sales: punchy openers, vivid pain-point "
    "descriptions, strong value propositions, and a clear call-to-action.\n"
    "- NEVER sound like a literal translation or textbook English.\n"
    "- Preserve emojis if the source contains them; otherwise add a few fitting emojis "
    "sparingly to match the platform vibe.\n\n"
    "Output ONLY the translated English script. No explanations, no notes, no markdown fences."
)


# ===================================================================
# API 请求函数（含错误重试机制）
# ===================================================================

@st.cache_resource
def _get_deepseek_client() -> OpenAI:
    """获取缓存的 DeepSeek 客户端实例（复用连接池）。

    DeepSeek API 完全兼容 OpenAI SDK，仅需指定 base_url。
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 文件中设置。")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


@st.cache_resource
def _get_elevenlabs_client() -> ElevenLabs:
    """获取缓存的 ElevenLabs 客户端实例（复用连接池）。"""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("未配置 ELEVENLABS_API_KEY，请在 .env 文件中设置。")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)


def translate_cn_to_en(text: str) -> str:
    """调用 DeepSeek API，将中文口播文案翻译为地道英文。

    Args:
        text: 用户输入的中文文本。

    Returns:
        翻译后的英文文本。

    Raises:
        RuntimeError: 所有重试均失败时抛出。
    """
    client = _get_deepseek_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("翻译请求 第 %d 次…", attempt)
            response = client.chat.completions.create(
                model=DEEPSEEK_TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.9,  # 稍高温度增加表达多样性
            )
            translated = response.choices[0].message.content
            if not translated:  # 同时检查 None 和空字符串
                raise RuntimeError("DeepSeek 返回了空翻译结果。")
            logger.info("翻译成功，共 %d 字符。", len(translated))
            return translated.strip()

        except Exception as exc:
            last_error = exc
            logger.warning("翻译失败（第 %d/%d 次）: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)  # 指数退避

    raise RuntimeError(f"翻译失败（已重试 {MAX_RETRIES} 次）: {last_error}")


def _generate_audio_single(
    text: str,
    voice_id: str,
    voice_label: str,
) -> Tuple[str, bytes]:
    """使用 ElevenLabs SDK 生成单路语音（内部函数）。

    Args:
        text: 英文文本。
        voice_id: ElevenLabs 语音 ID。
        voice_label: 语音标签（用于日志）。

    Returns:
        (voice_label, audio_bytes) 元组。

    Raises:
        RuntimeError: 所有重试均失败时抛出。
    """
    client = _get_elevenlabs_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("TTS [%s] 第 %d 次请求…", voice_label, attempt)
            audio_iter = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=ELEVENLABS_MODEL,
                output_format=ELEVENLABS_OUTPUT_FORMAT,
            )
            # SDK 返回 bytes 迭代器，拼接为完整音频
            audio_bytes = b"".join(audio_iter)
            if not audio_bytes:
                raise RuntimeError("ElevenLabs 返回了空音频数据。")
            logger.info("TTS [%s] 成功，音频大小 %d bytes。", voice_label, len(audio_bytes))
            return voice_label, audio_bytes

        except Exception as exc:
            last_error = exc
            logger.warning("TTS [%s] 失败（第 %d/%d 次）: %s", voice_label, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC * attempt)

    raise RuntimeError(f"语音生成失败 [{voice_label}]（已重试 {MAX_RETRIES} 次）: {last_error}")


def generate_dual_audio(text: str) -> Dict[str, Optional[bytes]]:
    """使用 ThreadPoolExecutor 并发请求 ElevenLabs，同时生成两个音色。

    Args:
        text: 英文文本。

    Returns:
        Dict[str, Optional[bytes]]，key 为语音标签，value 为音频 bytes（失败时为 None）。
        示例: {"Voice A (女声)": b"...", "Voice B (男声)": b"..."}
    """
    voice_configs = [
        (VOICE_A_ID, "Voice A (女声)"),
        (VOICE_B_ID, "Voice B (男声)"),
    ]

    results: Dict[str, Optional[bytes]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_map = {
            executor.submit(_generate_audio_single, text, vid, label): label
            for vid, label in voice_configs
        }
        for future in as_completed(future_map):
            label = future_map[future]
            try:
                voice_label, audio_bytes = future.result()
                results[voice_label] = audio_bytes
            except Exception as exc:
                logger.error("并发 TTS 任务失败 [%s]: %s", label, exc)
                # 即使一个语音失败，也不影响另一个；收集错误信息
                results[label] = None
                st.error(f"语音「{label}」生成失败: {exc}")

    return results


# ===================================================================
# 逐句高亮辅助
# ===================================================================

def split_sentences(text: str) -> List[str]:
    """将英文文本按句子拆分。

    按句号、感叹号、问号后的空格拆分，保留标点附着在句尾。

    Args:
        text: 英文文本。

    Returns:
        句子列表。若无法拆分则返回包含全文的单元素列表。
    """
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences if sentences else [text.strip()]


def _build_audio_player_html(
    english_text: str,
    audio_results: Dict[str, Optional[bytes]],
) -> str:
    """构建带逐句高亮的自包含音频播放器 HTML。

    通过字符数比例估算每句的时间区间，音频播放时
    JavaScript 端根据 currentTime 实时高亮当前句子。

    Args:
        english_text: 英文翻译文本。
        audio_results: {voice_label: audio_bytes_or_None} 字典。

    Returns:
        完整的 HTML 字符串，可直接传给 st.components.v1.html。
    """
    sentences = split_sentences(english_text)
    sentences_json = json.dumps(sentences)

    # 将音频 bytes 转为 base64 data URI
    audio_b64: Dict[str, str] = {}
    for label, audio_bytes in audio_results.items():
        if audio_bytes is not None:
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_b64[label] = f"data:audio/mp3;base64,{b64}"

    voice_a_b64 = audio_b64.get("Voice A (女声)", "")
    voice_b_b64 = audio_b64.get("Voice B (男声)", "")

    # 默认选中第一个可用的音色
    default_label = "Voice A (女声)" if voice_a_b64 else "Voice B (男声)"
    default_src = voice_a_b64 or voice_b_b64

    # ── HTML / CSS / JS（自包含） ──────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0e1117;
    color: #e0e0e0;
    padding: 12px 16px;
}}
.voice-tabs {{
    display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap;
}}
.voice-tab {{
    padding: 7px 18px;
    border: 1px solid #444;
    background: transparent;
    color: #aaa;
    border-radius: 20px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
}}
.voice-tab.active {{
    background: #1a5fb4; border-color: #3584e4; color: #fff;
}}
.voice-tab:hover:not(.active) {{ border-color: #888; color: #fff; }}
.audio-container {{ margin-bottom: 18px; }}
audio {{ width: 100%; border-radius: 6px; outline: none; }}
.text-container {{
    max-height: 420px;
    overflow-y: auto;
    padding: 14px 16px;
    background: #16191f;
    border-radius: 8px;
    border: 1px solid #2a2d35;
    line-height: 2;
    font-size: 15px;
}}
.sentence {{
    display: inline;
    padding: 2px 5px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.25s, color 0.25s, box-shadow 0.25s;
}}
.sentence.played {{ color: #aaa; }}
.sentence.active {{
    background: #1a5fb4;
    color: #fff;
    font-weight: 600;
    box-shadow: 0 0 8px rgba(53,132,228,0.35);
}}
.sentence.pending {{ color: #777; }}
.sentence:hover:not(.active) {{ background: #252830; }}
.download-row {{
    display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap;
}}
.download-btn {{
    padding: 6px 14px;
    border: 1px solid #444;
    background: transparent;
    color: #ccc;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    text-decoration: none;
    transition: all 0.2s;
}}
.download-btn:hover {{ border-color: #888; color: #fff; }}
</style>
</head>
<body>

<div class="voice-tabs" id="voiceTabs"></div>
<div class="audio-container">
    <audio id="ap" controls preload="auto"></audio>
</div>
<div class="text-container" id="textContainer"></div>
<div class="download-row">
    <a class="download-btn" id="dlA" download="voiceover_Voice_A.mp3">⬇ Voice A (女声)</a>
    <a class="download-btn" id="dlB" download="voiceover_Voice_B.mp3">⬇ Voice B (男声)</a>
</div>

<script>
const sentences = {sentences_json};
const sources = {{
    "Voice A (女声)": {json.dumps(voice_a_b64)},
    "Voice B (男声)": {json.dumps(voice_b_b64)}
}};
let boundaries = [];
let currentLabel = {json.dumps(default_label)};

const ap = document.getElementById("ap");
const tc = document.getElementById("textContainer");
const vt = document.getElementById("voiceTabs");
const dlA = document.getElementById("dlA");
const dlB = document.getElementById("dlB");

// ── voice tabs ──────────────────────────────────
Object.entries(sources).forEach(([label, src], idx) => {{
    if (!src) return;
    const btn = document.createElement("button");
    btn.className = "voice-tab" + (label === currentLabel ? " active" : "");
    btn.textContent = label;
    btn.onclick = function() {{ switchVoice(label); }};
    vt.appendChild(btn);
}});

// ── download links ──────────────────────────────
dlA.href = sources["Voice A (女声)"] || "#";
dlB.href = sources["Voice B (男声)"] || "#";

// ── sentence spans ──────────────────────────────
function buildSpans() {{
    tc.innerHTML = "";
    sentences.forEach(function(s, i) {{
        const span = document.createElement("span");
        span.className = "sentence pending";
        span.textContent = s + " ";
        span.dataset.idx = i;
        span.onclick = function() {{ seekTo(i); }};
        tc.appendChild(span);
    }});
}}
buildSpans();

// ── boundaries (character-count proportional) ───
function calcBoundaries() {{
    const dur = ap.duration;
    if (!dur || dur <= 0) return;
    const total = sentences.reduce(function(acc, s) {{ return acc + s.length; }}, 0);
    let off = 0;
    boundaries = sentences.map(function(s) {{
        const st = (off / total) * dur;
        const en = ((off + s.length) / total) * dur;
        off += s.length;
        return {{ start: st, end: en }};
    }});
}}

// ── voice switch ────────────────────────────────
function switchVoice(label) {{
    const src = sources[label];
    if (!src) return;
    const wasPlaying = !ap.paused;
    const saved = ap.currentTime;
    currentLabel = label;
    ap.src = src;

    vt.querySelectorAll(".voice-tab").forEach(function(tab) {{
        tab.classList.toggle("active", tab.textContent === label);
    }});

    ap.addEventListener("loadedmetadata", function() {{
        calcBoundaries();
        if (wasPlaying && saved < ap.duration) {{
            ap.currentTime = saved;
            ap.play();
        }}
    }}, {{ once: true }});
    ap.load();
}}

// ── seek on sentence click ──────────────────────
function seekTo(idx) {{
    if (boundaries.length === 0) calcBoundaries();
    if (boundaries.length === 0) return;
    ap.currentTime = boundaries[idx].start;
    if (ap.paused) ap.play();
    updateHighlight();
}}

// ── highlight current sentence ──────────────────
function updateHighlight() {{
    if (boundaries.length === 0) calcBoundaries();
    if (boundaries.length === 0) return;
    const ct = ap.currentTime;
    let active = -1;
    for (var i = 0; i < boundaries.length; i++) {{
        if (ct >= boundaries[i].start && ct < boundaries[i].end) {{
            active = i; break;
        }}
    }}
    if (active === -1 && ct >= boundaries[boundaries.length - 1].end) {{
        active = boundaries.length - 1;
    }}

    const spans = tc.querySelectorAll(".sentence");
    spans.forEach(function(sp, i) {{
        sp.classList.remove("active", "played", "pending");
        if (i < active) sp.classList.add("played");
        else if (i === active) sp.classList.add("active");
        else sp.classList.add("pending");
    }});

    if (active >= 0) {{
        spans[active].scrollIntoView({{ behavior: "smooth", block: "center" }});
    }}
}}

// ── events ──────────────────────────────────────
ap.addEventListener("loadedmetadata", calcBoundaries);
ap.addEventListener("timeupdate", updateHighlight);
ap.addEventListener("play", function() {{ calcBoundaries(); updateHighlight(); }});
ap.addEventListener("seeked", updateHighlight);
ap.addEventListener("ended", function() {{
    tc.querySelectorAll(".sentence").forEach(function(sp) {{
        sp.classList.remove("active"); sp.classList.add("played");
    }});
}});

// ── initial load ────────────────────────────────
ap.src = sources[currentLabel] || "";
</script>
</body>
</html>"""
    return html


# ===================================================================
# Streamlit UI
# ===================================================================

def setup_page() -> None:
    """配置 Streamlit 页面元数据与布局。"""
    st.set_page_config(
        page_title="中文口播 → 英文翻译 + 语音",
        page_icon="🎙️",
        layout="wide",
    )
    st.title("🎙️ 中文口播文案 → 英文翻译 + 语音合成")
    st.markdown(
        "输入中文带货/短视频口播文案，一键生成**地道英文翻译**和**双音色英文音频**，"
        "让运营同学快速产出海外短视频内容。"
    )
    st.divider()


def init_session_state() -> None:
    """初始化 session_state 中用于历史记录的数据结构。"""
    if "history" not in st.session_state:
        st.session_state.history = []  # 每一项: {cn, en, audios: {label: bytes}}


def render_input_section() -> str:
    """渲染输入区，返回用户输入的文本。

    Returns:
        用户输入的中文文本。
    """
    st.subheader("📝 第一步：输入中文口播文案")
    user_input = st.text_area(
        label="中文口播文本",
        value=SAMPLE_TEXT,
        height=150,
        placeholder="请在此粘贴中文口播文案…",
        key="input_text",
    )
    char_count = len(user_input.strip())
    if char_count > 0:
        st.caption(f"已输入 {char_count} 个字符")
    return user_input


def render_translation_section(english_text: str) -> None:
    """渲染翻译结果区，包含一键复制功能。

    Args:
        english_text: 翻译后的英文文本。
    """
    st.subheader("📖 英文翻译结果")
    st.success("翻译完成！以下是地道英文口播文案：")
    st.code(english_text, language=None)  # st.code 自带一键复制按钮

    # 额外提供一个纯文本复制区，确保兼容性
    with st.expander("📋 纯文本（可选中复制）", expanded=False):
        st.text(english_text)


def render_audio_section(audio_results: Dict[str, Optional[bytes]], english_text: str) -> None:
    """渲染带逐句高亮的音频播放区。

    使用自定义 HTML 组件实现：
    - 双音色切换
    - 播放时英文文本随进度逐句高亮
    - 点击句子跳转到对应播放位置
    - 下载按钮

    Args:
        audio_results: {voice_label: audio_bytes_or_None} 字典。
        english_text: 英文翻译文本，用于逐句拆分和高亮。
    """
    st.subheader("🔊 第二步：双音色语音（并发生成 + 逐句高亮）")
    st.caption(
        f"使用 ElevenLabs `{ELEVENLABS_MODEL}` 模型，"
        f"输出格式 `{ELEVENLABS_OUTPUT_FORMAT}`。"
    )

    if not audio_results:
        st.warning("没有生成任何音频。")
        return

    # ── 逐句高亮播放器 ──
    player_html = _build_audio_player_html(english_text, audio_results)
    st.components.v1.html(player_html, height=600, scrolling=False)

    # ── 下载按钮（Streamlit 原生，更可靠） ──
    labels = list(audio_results.keys())
    if len(labels) >= 2:
        col_left, col_right = st.columns(2)
        columns = [col_left, col_right]
    else:
        columns = [st]

    for idx, label in enumerate(labels):
        audio_bytes = audio_results[label]
        col = columns[idx] if idx < len(columns) else st
        with col:
            if audio_bytes is not None:
                safe_label = re.sub(r"[^\w.-]", "_", label)
                st.download_button(
                    label=f"⬇ 下载 {label}.mp3",
                    data=audio_bytes,
                    file_name=f"voiceover_{safe_label}.mp3",
                    mime="audio/mpeg",
                    key=f"dl_{safe_label}",
                )
            else:
                st.error(f"「{label}」生成失败，无音频数据。")


def render_history() -> None:
    """渲染历史记录区，遍历 session_state.history 渲染所有历史条目。"""
    if len(st.session_state.history) == 0:
        return

    st.divider()

    col_hist_title, col_hist_clear = st.columns([4, 1])
    with col_hist_title:
        st.subheader(f"📚 历史记录（共 {len(st.session_state.history)} 条 / 上限 {MAX_HISTORY} 条）")
    with col_hist_clear:
        if st.button("🗑 清空历史", type="secondary", use_container_width=True):
            st.session_state.history.clear()
            st.rerun()

    for display_num, record in enumerate(st.session_state.history[::-1], start=1):
        with st.expander(f"#{display_num}  {record['cn'][:50]}…", expanded=(display_num == 1)):
            st.markdown("**中文原文**")
            st.text(record["cn"])
            st.markdown("**英文翻译**")
            st.code(record["en"], language=None)

            audios = record.get("audios", {})
            if audios:
                st.markdown("**音频**")
                audio_labels = list(audios.keys())
                if len(audio_labels) >= 2:
                    c1, c2 = st.columns(2)
                    cols = [c1, c2]
                else:
                    cols = [st]
                for j, label in enumerate(audio_labels):
                    with cols[j]:
                        st.caption(label)
                        if audios[label] is not None:
                            st.audio(audios[label], format="audio/mp3")


# ===================================================================
# 主流程
# ===================================================================

def main() -> None:
    """Streamlit 应用主入口。"""
    setup_page()
    init_session_state()

    # ----- 步骤 1：输入 -----
    user_input = render_input_section()

    # ----- 步骤 2：翻译 + 语音 -----
    st.divider()
    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        translate_clicked = st.button(
            "🚀 翻译并生成语音",
            type="primary",
            use_container_width=True,
            disabled=not user_input.strip(),
        )

    if translate_clicked:
        # --- 翻译 ---
        with st.status("🤖 正在进行地道英文翻译…", expanded=True) as status:
            try:
                english_text = translate_cn_to_en(user_input.strip())
                status.update(label="✅ 翻译完成！", state="running")
            except Exception as exc:
                status.update(label=f"❌ 翻译失败: {exc}", state="error")
                st.error(f"翻译失败，请重试。错误详情：{exc}")
                return  # 提前返回，保留历史记录渲染

        # --- 并发语音合成 ---
        with st.status("🔊 正在并发生成双音色语音…", expanded=True) as status:
            try:
                audio_results = generate_dual_audio(english_text)
                success_count = sum(1 for v in audio_results.values() if v is not None)
                status.update(
                    label=f"✅ 语音生成完成！成功 {success_count}/{len(audio_results)} 路。",
                    state="complete",
                )
            except Exception as exc:
                status.update(label=f"❌ 语音生成失败: {exc}", state="error")
                audio_results = {}

        # --- 展示结果 ---
        render_translation_section(english_text)
        if audio_results:
            render_audio_section(audio_results, english_text)

        # --- 写入历史记录 ---
        st.session_state.history.append({
            "cn": user_input.strip(),
            "en": english_text,
            "audios": audio_results,
        })
        # 超出上限时移除最旧的记录
        while len(st.session_state.history) > MAX_HISTORY:
            st.session_state.history.pop(0)
        logger.info("历史记录已更新，当前共 %d 条。", len(st.session_state.history))

    # ----- 历史记录 -----
    render_history()


if __name__ == "__main__":
    main()
