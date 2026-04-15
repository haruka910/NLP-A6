import platform
import random
import re
import math
from collections import Counter
from typing import Dict

import pandas as pd
import streamlit as st

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoTokenizer,
    )
except Exception:
    AutoModelForCausalLM = None
    AutoModelForMaskedLM = None
    AutoTokenizer = None


PAGE_TITLE = "Language Model Training & Comparison Lab"
PROJECT_DESCRIPTION = (
    "本平台用于在统一的教学实验界面中，对比统计语言模型、"
    "自训练 RNN 语言模型、预训练语言模型，以及基于困惑度的评价方式。"
)


def apply_academic_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --paper: #f4f0e8;
          --panel: #fffdf8;
          --ink: #1f2a37;
          --muted: #5f6874;
          --line: #d8cdbd;
          --accent: #8a4b14;
          --accent-soft: #efe2cf;
          --gold-soft: #f5ead2;
        }

        .stApp {
          background: var(--paper);
          color: var(--ink);
          font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        }

        .block-container {
          padding-top: 1.2rem;
          padding-bottom: 2.2rem;
          max-width: 1240px;
        }

        h1, h2, h3 {
          font-family: Georgia, "Palatino Linotype", "Noto Serif SC", serif;
          color: var(--ink);
          letter-spacing: 0;
        }

        p, li, label, .stMarkdown, .stCaption {
          color: var(--ink);
          line-height: 1.75;
        }

        .hero-shell {
          background: linear-gradient(135deg, #fffaf2 0%, #fffdf8 58%, #f4eee4 100%);
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 1.35rem 1.5rem;
          box-shadow: 0 18px 40px rgba(31, 42, 55, 0.05);
          margin-bottom: 1rem;
        }

        .hero-kicker {
          color: var(--accent);
          font-size: 0.82rem;
          font-weight: 600;
          margin-bottom: 0.35rem;
        }

        .hero-title {
          font-family: Georgia, "Palatino Linotype", "Noto Serif SC", serif;
          color: var(--ink);
          font-size: 2.2rem;
          line-height: 1.15;
          margin: 0 0 0.45rem 0;
        }

        .hero-subtitle {
          color: var(--muted);
          font-size: 0.98rem;
          line-height: 1.8;
          margin: 0;
        }

        .chip-row {
          margin-top: 0.85rem;
          display: flex;
          gap: 0.45rem;
          flex-wrap: wrap;
        }

        .chip {
          background: var(--accent-soft);
          border: 1px solid var(--line);
          border-radius: 999px;
          color: var(--accent);
          font-size: 0.78rem;
          padding: 0.18rem 0.55rem;
        }

        .concept-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: 0.95rem 1rem;
          height: 100%;
          box-shadow: 0 10px 24px rgba(31, 42, 55, 0.03);
        }

        .concept-kicker {
          color: var(--accent);
          font-size: 0.76rem;
          margin-bottom: 0.25rem;
          font-weight: 600;
        }

        .concept-title {
          color: var(--ink);
          font-size: 1rem;
          font-weight: 600;
          margin-bottom: 0.25rem;
        }

        .concept-body {
          color: var(--muted);
          font-size: 0.92rem;
          line-height: 1.75;
        }

        .step-shell {
          margin-top: 0.9rem;
          margin-bottom: 0.75rem;
        }

        .step-tag {
          display: inline-block;
          background: var(--accent-soft);
          color: var(--accent);
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 0.14rem 0.52rem;
          font-size: 0.76rem;
          margin-bottom: 0.35rem;
        }

        .step-title {
          font-family: Georgia, "Palatino Linotype", "Noto Serif SC", serif;
          color: var(--ink);
          font-size: 1.25rem;
          margin-bottom: 0.15rem;
        }

        .step-desc {
          color: var(--muted);
          font-size: 0.93rem;
          line-height: 1.75;
        }

        section[data-testid="stSidebar"] {
          background: #f7f2e9;
          border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input,
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          color: var(--ink);
        }

        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
        }

        .stSlider [data-baseweb="slider"] {
          padding-top: 0.2rem;
          padding-bottom: 0.2rem;
        }

        .stButton > button {
          background: var(--accent-soft);
          color: var(--ink);
          border: 1px solid var(--line);
          border-radius: 8px;
          box-shadow: none;
        }

        .stButton > button:hover {
          border-color: var(--accent);
          color: var(--accent);
        }

        [data-testid="stMetric"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: 0.85rem 0.95rem;
          box-shadow: 0 10px 24px rgba(31, 42, 55, 0.03);
        }

        [data-testid="stMetricLabel"] {
          color: var(--muted);
        }

        [data-testid="stMetricValue"] {
          color: var(--ink);
          font-family: Georgia, "Palatino Linotype", "Noto Serif SC", serif;
        }

        .stAlert {
          border-radius: 10px;
          border: 1px solid var(--line);
        }

        .streamlit-expanderHeader {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
        }

        div[data-testid="stDataFrame"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: 0.2rem;
        }

        button[data-baseweb="tab"] {
          background: #f7f1e7;
          border: 1px solid var(--line);
          border-radius: 8px;
          color: var(--muted);
          margin-right: 0.35rem;
          padding: 0.4rem 0.8rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
          background: var(--panel);
          color: var(--ink);
          border-color: var(--accent);
        }

        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
          gap: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-kicker">课程实验平台</div>
          <div class="hero-title">{PAGE_TITLE}</div>
          <p class="hero-subtitle">{PROJECT_DESCRIPTION}</p>
          <div class="chip-row">
            <span class="chip">统计语言模型</span>
            <span class="chip">RNN / LSTM</span>
            <span class="chip">BERT / GPT-2</span>
            <span class="chip">Perplexity</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_concept_cards() -> None:
    col1, col2, col3 = st.columns(3)
    cards = [
        (
            "Concept",
            "先理解任务目标",
            "每个页面都先解释模型在解决什么问题，再进入输入与结果，避免初学者一开始就被指标淹没。",
        ),
        (
            "Reading",
            "先看整体，再看细节",
            "页面会先给出核心结论与阅读提示，再展示表格、候选词、生成文本或困惑度结果。",
        ),
        (
            "Teaching",
            "把结果和概念对应起来",
            "你不仅可以看到输出，还可以结合说明理解为什么会出现这些结果，以及方法容易在哪些地方失败。",
        ),
    ]
    for col, (kicker, title, body) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f"""
                <div class="concept-card">
                  <div class="concept-kicker">{kicker}</div>
                  <div class="concept-title">{title}</div>
                  <div class="concept-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_step_header(step: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="step-shell">
          <div class="step-tag">{step}</div>
          <div class="step-title">{title}</div>
          <div class="step-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_device_info() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return f"CUDA available ({torch.cuda.get_device_name(0)})"
        return "CPU only (PyTorch detected)"
    except Exception:
        system = platform.system()
        machine = platform.machine()
        return f"CPU only ({system}, {machine})"


def apply_seed(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def render_sidebar() -> Dict[str, object]:
    st.sidebar.header("控制区")

    seed = st.sidebar.number_input(
        "随机种子",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help="用于让实验结果尽量可复现。",
    )
    device_info = detect_device_info()
    show_teaching_tips = st.sidebar.checkbox(
        "显示教学提示",
        value=True,
    )

    st.sidebar.text_input("设备信息", value=device_info, disabled=True)

    return {
        "seed": int(seed),
        "device_info": device_info,
        "show_teaching_tips": show_teaching_tips,
    }


def render_intro() -> None:
    render_hero()
    render_concept_cards()


def render_teaching_tips() -> None:
    st.caption("教学提示")
    st.markdown(
        "- 先从语料直觉出发，再理解平滑为什么必要。\n"
        "- 把计数、条件概率和失败案例分开看，会更容易建立概念。\n"
        "- 当 `n` 变大时，最需要留意的是数据稀疏问题。"
    )


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def parse_corpus_sentences(corpus_text: str) -> list[list[str]]:
    sentences = []
    for line in corpus_text.splitlines():
        tokens = tokenize_text(line)
        if tokens:
            sentences.append(tokens)
    return sentences


def add_boundary_tokens(tokens: list[str], n: int) -> list[str]:
    bos_count = max(n - 1, 1)
    return ["<BOS>"] * bos_count + tokens + ["<EOS>"]


def build_ngram_model(
    sentences: list[list[str]], n: int
) -> tuple[Counter, Counter, list[str], set[str]]:
    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()
    processed_tokens: list[str] = []
    vocabulary: set[str] = set()

    for sentence in sentences:
        augmented = add_boundary_tokens(sentence, n)
        processed_tokens.extend(augmented)
        vocabulary.update(augmented)

        if n == 1:
            for token in augmented[1:]:
                ngram_counts[(token,)] += 1
            continue

        for idx in range(n - 1, len(augmented)):
            ngram = tuple(augmented[idx - n + 1 : idx + 1])
            context = ngram[:-1]
            ngram_counts[ngram] += 1
            context_counts[context] += 1

    return ngram_counts, context_counts, processed_tokens, vocabulary


def get_prediction_vocabulary(vocabulary: set[str]) -> list[str]:
    return sorted(token for token in vocabulary if token != "<BOS>")


def format_history(history: tuple[str, ...]) -> str:
    if not history:
        return "(none)"
    return " ".join(history)


def compute_conditional_probability(
    *,
    n: int,
    history: tuple[str, ...],
    target: str,
    ngram_counts: Counter,
    context_counts: Counter,
    prediction_vocab_size: int,
    use_smoothing: bool,
    delta: float,
) -> tuple[int, int, float]:
    if n == 1:
        count_hw = ngram_counts.get((target,), 0)
        count_h = sum(ngram_counts.values())
    else:
        count_hw = ngram_counts.get(history + (target,), 0)
        count_h = context_counts.get(history, 0)

    if use_smoothing:
        numerator = count_hw + delta
        denominator = count_h + delta * prediction_vocab_size
        probability = numerator / denominator if denominator > 0 else 0.0
    else:
        probability = count_hw / count_h if count_h > 0 else 0.0

    return count_hw, count_h, probability


def analyze_sentence(
    sentence_text: str,
    n: int,
    ngram_counts: Counter,
    context_counts: Counter,
    prediction_vocab_size: int,
    *,
    use_smoothing: bool,
    delta: float,
) -> dict[str, object]:
    tokens = tokenize_text(sentence_text)
    if not tokens:
        return {
            "joint_probability": 0.0,
            "rows": [],
            "warning": "请输入至少包含一个有效 token 的英文句子。",
            "zero_probability": False,
        }

    augmented = add_boundary_tokens(tokens, n)
    probability = 1.0
    rows: list[dict[str, object]] = []
    zero_probability = False

    for idx in range(n - 1, len(augmented)):
        target = augmented[idx]
        history = tuple(augmented[idx - n + 1 : idx]) if n > 1 else tuple()
        count_hw, count_h, conditional_probability = compute_conditional_probability(
            n=n,
            history=history,
            target=target,
            ngram_counts=ngram_counts,
            context_counts=context_counts,
            prediction_vocab_size=prediction_vocab_size,
            use_smoothing=use_smoothing,
            delta=delta,
        )

        is_zero_step = conditional_probability == 0.0
        if is_zero_step:
            zero_probability = True
            probability = 0.0
        elif not zero_probability:
            probability *= conditional_probability

        rows.append(
            {
                "history": format_history(history),
                "target word": target,
                "count(history, word)": count_hw,
                "count(history)": count_h,
                "conditional probability": conditional_probability,
                "zero_step": is_zero_step,
            }
        )

    warning = None
    if zero_probability and not use_smoothing:
        warning = (
            "检测到零概率：句子中至少有一个所需的 n-gram 没在训练语料里出现过，因此整句联合概率会变成 0。这就是典型的数据稀疏问题。"
        )

    return {
        "joint_probability": probability,
        "rows": rows,
        "warning": warning,
        "zero_probability": zero_probability,
    }


def make_breakdown_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "history",
                "target word",
                "count(history, word)",
                "count(history)",
                "conditional probability",
            ]
        )

    dataframe = pd.DataFrame(rows)
    return dataframe[
        [
            "history",
            "target word",
            "count(history, word)",
            "count(history)",
            "conditional probability",
            "zero_step",
        ]
    ]


def style_breakdown_dataframe(dataframe: pd.DataFrame):
    if dataframe.empty:
        return dataframe

    display_df = dataframe.drop(columns=["zero_step"])

    def highlight_zero_step(row: pd.Series) -> list[str]:
        if bool(dataframe.loc[row.name, "zero_step"]):
            return ["background-color: #fde8e8"] * len(display_df.columns)
        return [""] * len(row)

    styler = display_df.style.format(
        {
            "conditional probability": "{:.6g}",
        }
    )
    return styler.apply(highlight_zero_step, axis=1)


def get_top_word_dataframe(sentences: list[list[str]], top_k: int = 10) -> pd.DataFrame:
    word_counts: Counter = Counter()
    for sentence in sentences:
        word_counts.update(sentence)

    rows = [
        {"word": word, "count": count}
        for word, count in word_counts.most_common(top_k)
    ]
    return pd.DataFrame(rows)


def get_top_ngram_dataframe(ngram_counts: Counter, top_k: int = 10) -> pd.DataFrame:
    rows = [
        {"n-gram": " ".join(ngram), "count": count}
        for ngram, count in ngram_counts.most_common(top_k)
    ]
    return pd.DataFrame(rows)


def render_probability_comparison(
    unsmoothed_result: dict[str, object],
    smoothed_result: dict[str, object],
    delta: float,
) -> None:
    st.markdown("**平滑前后对比**")
    comparison_df = pd.DataFrame(
        [
            {
                "setting": "未使用平滑",
                "joint probability": unsmoothed_result["joint_probability"],
            },
            {
                "setting": f"使用 Laplace smoothing (delta={delta:.2f})",
                "joint probability": smoothed_result["joint_probability"],
            },
        ]
    )
    st.dataframe(
        comparison_df.style.format({"joint probability": "{:.12g}"}),
        use_container_width=True,
        hide_index=True,
    )


def render_teaching_expander() -> None:
    with st.expander("教学说明：为什么稀疏与平滑重要", expanded=False):
        st.markdown(
            "- `n` 越大，条件历史越长，语料就需要覆盖更多不同的上下文组合，因此未见事件会更常出现。\n"
            "- 训练语料里没见过某个事件，并不等于它在真实语言中概率就是 0，它也可能只是稀有、但没有刚好出现在有限数据里。\n"
            "- Additive smoothing 会给每个候选词分配一点概率余量，也就是在计数上加上 `delta`，从而避免一个未见事件直接把整句概率压成 0。"
        )


def render_tab_ngram(show_teaching_tips: bool) -> None:
    st.subheader("Tab1 · n-gram & smoothing")
    render_step_header(
        "Step 1",
        "理解统计语言模型如何给句子赋概率",
        "这一页用于展示最基础的 n-gram 语言模型。你可以输入英文语料、选择 `unigram / bigram / trigram`，然后观察模型如何把一句话分解成若干步条件概率。",
    )

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        corpus_text = st.text_area(
            "英文语料",
            value=(
                "The cat sat on the mat.\n"
                "The cat saw the mat.\n"
                "The dog sat near the cat."
            ),
            height=140,
        )
        sentence_text = st.text_input(
            "待计算概率的句子",
            value="the cat sat on the mat",
        )
        n_label = st.selectbox(
            "n-gram 阶数",
            options=["unigram", "bigram", "trigram"],
            index=2,
        )
        use_smoothing = st.checkbox(
            "启用 Add-one / Laplace smoothing",
            value=False,
        )
        delta = st.slider(
            "delta",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
        )

    n_map = {"unigram": 1, "bigram": 2, "trigram": 3}
    n = n_map[n_label]
    sentences = parse_corpus_sentences(corpus_text)

    if not sentences:
        token_total = 0
        vocabulary_size = 0
        joint_probability = 0.0
        warning_message = (
            "分词后没有得到有效语料，请至少输入一条英文句子。"
        )
        unsmoothed_result = {
            "joint_probability": 0.0,
            "rows": [],
            "warning": warning_message,
            "zero_probability": False,
        }
        smoothed_result = unsmoothed_result
        top_word_df = pd.DataFrame(columns=["word", "count"])
        top_ngram_df = pd.DataFrame(columns=["n-gram", "count"])
    else:
        ngram_counts, context_counts, processed_tokens, vocabulary = build_ngram_model(
            sentences, n
        )
        token_total = len(processed_tokens)
        vocabulary_size = len(vocabulary)
        prediction_vocab_size = len(get_prediction_vocabulary(vocabulary))
        unsmoothed_result = analyze_sentence(
            sentence_text=sentence_text,
            n=n,
            ngram_counts=ngram_counts,
            context_counts=context_counts,
            prediction_vocab_size=prediction_vocab_size,
            use_smoothing=False,
            delta=0.0,
        )
        smoothed_result = analyze_sentence(
            sentence_text=sentence_text,
            n=n,
            ngram_counts=ngram_counts,
            context_counts=context_counts,
            prediction_vocab_size=prediction_vocab_size,
            use_smoothing=True,
            delta=delta,
        )
        active_result = smoothed_result if use_smoothing else unsmoothed_result
        joint_probability = float(active_result["joint_probability"])
        warning_message = active_result["warning"]
        top_word_df = get_top_word_dataframe(sentences)
        top_ngram_df = get_top_ngram_dataframe(ngram_counts)

    active_result = smoothed_result if use_smoothing else unsmoothed_result
    breakdown_df = make_breakdown_dataframe(active_result["rows"])

    with right_col:
        st.markdown("**如何阅读这一页**")
        st.info(
            "先看右侧的三个统计量：`token 总数` 反映语料规模，`vocabulary size` 反映词表大小，`句子联合概率` 反映当前模型认为这句话有多常见。"
        )
        st.markdown("**结果总览**")
        st.metric(
            "token 总数",
            token_total,
            help="已包含小写化、分词，以及 `<BOS>` / `<EOS>` 边界符号。",
        )
        st.metric("vocabulary size", vocabulary_size)
        st.metric("句子联合概率", f"{joint_probability:.12g}")

        if warning_message:
            st.warning(warning_message)
        else:
            if use_smoothing:
                st.success(
                    f"当前结果使用了 Additive smoothing，`delta={delta:.2f}`。"
                )
            else:
                st.success(
                    "当前结果直接来自原始 n-gram 计数，未使用平滑。"
                )

        if show_teaching_tips:
            render_teaching_tips()

    render_step_header(
        "Step 2",
        "先看逐步概率分解",
        "下面的表格把整句概率拆成一步一步的条件概率。建议从 `history`、`target word`、`conditional probability` 三列开始读，再回头看计数列。",
    )
    if breakdown_df.empty:
        st.info("输入非空语料和句子后，这里会展示逐步概率分解。")
    else:
        if not use_smoothing and bool(active_result["zero_probability"]):
            st.error(
                "下表中至少有一步的条件概率为 0，已经高亮显示。由于联合概率是连乘，这一步会把整句概率直接压成 0。"
            )

        st.dataframe(
            style_breakdown_dataframe(breakdown_df),
            use_container_width=True,
            hide_index=True,
        )

    if use_smoothing and sentences:
        render_probability_comparison(unsmoothed_result, smoothed_result, delta)

    render_step_header(
        "Step 3",
        "最后看语料分布",
        "高频词和高频 n-gram 可以帮助你理解模型主要从哪些局部模式中学习。它们不是最终结论，但能帮助你解释概率为什么会高或低。",
    )
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.markdown("**高频词 Top 列表**")
        st.dataframe(top_word_df, use_container_width=True, hide_index=True)
    with stats_col2:
        st.markdown(f"**高频 {n_label} 列表**")
        st.dataframe(top_ngram_df, use_container_width=True, hide_index=True)

    render_teaching_expander()


if nn is not None:
    class CharSequenceLM(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_size: int,
            model_type: str,
        ) -> None:
            super().__init__()
            self.model_type = model_type
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if model_type == "RNN":
                self.sequence_model = nn.RNN(
                    embedding_dim,
                    hidden_size,
                    batch_first=True,
                )
            else:
                self.sequence_model = nn.LSTM(
                    embedding_dim,
                    hidden_size,
                    batch_first=True,
                )
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, x, hidden=None):
            embedded = self.embedding(x)
            output, hidden = self.sequence_model(embedded, hidden)
            logits = self.output(output)
            return logits, hidden
else:
    class CharSequenceLM:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("使用 Tab2 需要 PyTorch。")


def init_tab2_state() -> None:
    defaults = {
        "tab2_model_artifact": None,
        "tab2_loss_history": [],
        "tab2_generated_text": "",
        "tab2_seed_text": "The ",
        "tab2_generate_length": 120,
        "tab2_experiment_records": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_torch_device():
    if torch is None:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    vocab = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(vocab)}
    itos = {idx: char for char, idx in stoi.items()}
    return stoi, itos


def encode_char_text(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[char] for char in text]


def create_sequence_pairs(encoded: list[int], seq_length: int) -> list[tuple[list[int], list[int]]]:
    sequence_pairs: list[tuple[list[int], list[int]]] = []
    if len(encoded) < 2:
        return sequence_pairs

    for start in range(0, len(encoded) - 1, seq_length):
        end = min(start + seq_length, len(encoded) - 1)
        input_seq = encoded[start:end]
        target_seq = encoded[start + 1 : end + 1]
        if input_seq and target_seq:
            sequence_pairs.append((input_seq, target_seq))

    return sequence_pairs


def make_training_tensors(text: str, stoi: dict[str, int], seq_length: int, device):
    encoded = encode_char_text(text, stoi)
    sequence_pairs = create_sequence_pairs(encoded, seq_length)
    tensor_pairs = []
    for input_seq, target_seq in sequence_pairs:
        input_tensor = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)
        target_tensor = torch.tensor(target_seq, dtype=torch.long, device=device).unsqueeze(0)
        tensor_pairs.append((input_tensor, target_tensor))
    return tensor_pairs


def count_model_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def serialize_state_dict(model) -> dict[str, object]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_char_rnn_model(
    corpus_text: str,
    model_type: str,
    embedding_dim: int,
    hidden_size: int,
    epochs: int,
    seq_length: int,
    learning_rate: float,
    gradient_clipping: bool,
    *,
    epoch_slot,
    loss_slot,
    progress_bar,
    status_slot,
    chart_slot,
) -> dict[str, object]:
    if torch is None or nn is None:
        raise RuntimeError("当前环境没有可用的 PyTorch。")

    device = get_torch_device()
    stoi, itos = build_char_vocab(corpus_text)
    vocab_size = len(stoi)
    training_samples = make_training_tensors(corpus_text, stoi, seq_length, device)
    sample_count = len(training_samples)

    if sample_count == 0:
        raise RuntimeError("当前语料对于所选 `seq length` 来说过短，无法形成训练样本。")

    model = CharSequenceLM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        model_type=model_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_history: list[float] = []
    parameter_count = count_model_parameters(model)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_total = 0.0

        for input_tensor, target_tensor in training_samples:
            optimizer.zero_grad()
            logits, _ = model(input_tensor)
            loss = criterion(logits.reshape(-1, vocab_size), target_tensor.reshape(-1))
            loss.backward()

            if gradient_clipping:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss_total += float(loss.item())

        avg_loss = epoch_loss_total / sample_count
        loss_history.append(avg_loss)

        epoch_slot.metric("当前 epoch", f"{epoch}/{epochs}")
        loss_slot.metric("当前平均 loss", f"{avg_loss:.6f}")
        progress_bar.progress(epoch / epochs)
        status_slot.info(f"正在 {device.type} 上训练...")
        chart_slot.line_chart(
            pd.DataFrame(
                {"loss": loss_history},
                index=range(1, len(loss_history) + 1),
            )
        )

    artifact = {
        "state_dict": serialize_state_dict(model),
        "stoi": stoi,
        "itos": itos,
        "model_type": model_type,
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "epochs": epochs,
        "seq_length": seq_length,
        "learning_rate": learning_rate,
        "gradient_clipping": gradient_clipping,
        "sample_count": sample_count,
        "parameter_count": parameter_count,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "training_text": corpus_text,
    }
    return artifact


def load_char_rnn_model(artifact: dict[str, object]):
    if torch is None:
        raise RuntimeError("当前环境没有可用的 PyTorch。")

    device = get_torch_device()
    model = CharSequenceLM(
        vocab_size=int(artifact["vocab_size"]),
        embedding_dim=int(artifact["embedding_dim"]),
        hidden_size=int(artifact["hidden_size"]),
        model_type=str(artifact["model_type"]),
    ).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, device


def generate_text_from_model(
    artifact: dict[str, object],
    seed_text: str,
    generate_length: int,
    strategy: str,
    temperature: float,
) -> str:
    if torch is None:
        raise RuntimeError("当前环境没有可用的 PyTorch。")

    stoi = artifact["stoi"]
    itos = artifact["itos"]

    if not seed_text:
        raise ValueError("seed 输入不能为空。")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0。")

    unknown_chars = sorted({char for char in seed_text if char not in stoi})
    if unknown_chars:
        raise ValueError(
            "seed 中包含训练阶段未见过的字符："
            + ", ".join(repr(char) for char in unknown_chars)
        )

    model, device = load_char_rnn_model(artifact)
    generated_text = seed_text

    with torch.no_grad():
        seed_ids = [stoi[char] for char in seed_text]
        seed_tensor = torch.tensor(seed_ids, dtype=torch.long, device=device).unsqueeze(0)
        _, hidden = model(seed_tensor)

        last_id = seed_ids[-1]
        current_input = torch.tensor([[last_id]], dtype=torch.long, device=device)

        for _ in range(generate_length):
            logits, hidden = model(current_input, hidden)
            next_token_logits = logits[:, -1, :] / temperature
            if strategy == "greedy":
                next_id = int(torch.argmax(next_token_logits, dim=-1).item())
            else:
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            next_char = itos[next_id]
            generated_text += next_char
            current_input = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return generated_text


def count_training_samples(text: str, seq_length: int) -> int:
    if not text:
        return 0
    stoi, _ = build_char_vocab(text)
    encoded = encode_char_text(text, stoi)
    return len(create_sequence_pairs(encoded, seq_length))


def suggest_seed_text(corpus_text: str) -> str:
    for line in corpus_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[: min(4, len(stripped))]
    return corpus_text[: min(4, len(corpus_text))]


def estimate_model_parameter_count(
    vocab_size: int,
    embedding_dim: int,
    hidden_size: int,
    model_type: str,
) -> int:
    if torch is None or nn is None or vocab_size == 0:
        return 0

    model = CharSequenceLM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        model_type=model_type,
    )
    return count_model_parameters(model)


def append_experiment_record(artifact: dict[str, object]) -> None:
    records = list(st.session_state["tab2_experiment_records"])
    records.insert(
        0,
        {
            "model type": artifact["model_type"],
            "hidden size": artifact["hidden_size"],
            "epochs": artifact["epochs"],
            "final loss": artifact["final_loss"],
        },
    )
    st.session_state["tab2_experiment_records"] = records[:8]


def render_tab2_teaching_note() -> None:
    st.markdown("**教学说明**")
    st.markdown(
        "- RNN 会维护一个 hidden state。每读入一个字符，这个状态都会更新，因此模型可以把前面历史压缩成一个动态表示并传到后面。\n"
        "- 当序列很长时，梯度在时间维度反复传播，容易逐步衰减或放大，于是出现梯度消失或梯度爆炸。\n"
        "- LSTM 通常更稳定，因为它用门控机制控制“保留什么、遗忘什么、输出什么”。"
    )


def render_tab_rnn() -> None:
    st.subheader("Tab2 · train RNN")
    render_step_header(
        "Step 1",
        "理解 character-level 自回归训练在做什么",
        "这一页演示最基础的 character-level 语言模型训练。模型读取前面的字符序列，预测下一个字符，因此你可以直观看到 RNN / LSTM 如何从局部字符模式中学习。",
    )

    init_tab2_state()

    if torch is None or nn is None:
        st.error("当前环境没有可用的 PyTorch，因此 Tab2 不能训练模型。")
        return

    default_corpus = (
        "hello world\n"
        "this is a tiny character level corpus\n"
        "language models learn to predict the next character\n"
    )

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        corpus_text = st.text_area(
            "训练语料",
            value=default_corpus,
            height=180,
            key="tab2_corpus_text",
        )
        model_type = st.selectbox(
            "模型类型",
            options=["RNN", "LSTM"],
            index=0,
        )
        embedding_dim = st.number_input(
            "embedding dim",
            min_value=4,
            max_value=512,
            value=32,
            step=4,
        )
        hidden_size = st.number_input(
            "hidden size",
            min_value=8,
            max_value=512,
            value=64,
            step=8,
        )
        seq_length = st.number_input(
            "seq length",
            min_value=2,
            max_value=512,
            value=32,
            step=1,
        )
        epochs = st.number_input(
            "epochs",
            min_value=1,
            max_value=1000,
            value=50,
            step=1,
        )
        learning_rate = st.number_input(
            "learning rate",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.0005,
            format="%.4f",
        )
        gradient_clipping = st.checkbox(
            "启用 gradient clipping",
            value=True,
        )
        start_training = st.button("开始训练", use_container_width=True)

    with right_col:
        artifact = st.session_state["tab2_model_artifact"]
        current_vocab_size = len(set(corpus_text))
        current_sample_count = count_training_samples(corpus_text, int(seq_length))
        current_parameter_count = estimate_model_parameter_count(
            vocab_size=current_vocab_size,
            embedding_dim=int(embedding_dim),
            hidden_size=int(hidden_size),
            model_type=model_type,
        )
        st.markdown("**如何阅读这一页**")
        st.info(
            "先看 `character vocabulary size`、`training sample count` 和 `model parameter count`，了解任务规模；训练后再看 `final loss` 和 loss 曲线，判断模型是否逐渐学到稳定模式。"
        )
        st.markdown("**结果总览**")
        st.metric("字符表大小", current_vocab_size)
        st.metric("训练样本数", current_sample_count)
        st.metric("模型参数量", current_parameter_count)
        if artifact is None:
            st.info("还没有训练结果。")
        else:
            st.metric("最终 loss", f"{float(artifact['final_loss']):.6f}")
            st.caption(
                f"{artifact['model_type']} | embedding = {artifact['embedding_dim']} | "
                f"hidden = {artifact['hidden_size']} | epochs = {artifact['epochs']}"
            )

    training_status = st.container()
    with training_status:
        epoch_slot, loss_slot = st.columns(2)
        progress_bar = st.progress(0.0)
        status_slot = st.empty()
        chart_slot = st.empty()

    existing_loss_history = st.session_state["tab2_loss_history"]
    if existing_loss_history:
        epoch_slot.metric(
            "当前 epoch",
            f"{len(existing_loss_history)}/{len(existing_loss_history)}",
        )
        loss_slot.metric(
            "当前平均 loss",
            f"{float(existing_loss_history[-1]):.6f}",
        )
        progress_bar.progress(1.0)
        chart_slot.line_chart(
            pd.DataFrame(
                {"loss": existing_loss_history},
                index=range(1, len(existing_loss_history) + 1),
            )
        )

    if start_training:
        if len(corpus_text) < 2:
            status_slot.error("训练语料至少需要包含两个字符。")
        else:
            artifact = train_char_rnn_model(
                corpus_text=corpus_text,
                model_type=model_type,
                embedding_dim=int(embedding_dim),
                hidden_size=int(hidden_size),
                epochs=int(epochs),
                seq_length=int(seq_length),
                learning_rate=float(learning_rate),
                gradient_clipping=bool(gradient_clipping),
                epoch_slot=epoch_slot,
                loss_slot=loss_slot,
                progress_bar=progress_bar,
                status_slot=status_slot,
                chart_slot=chart_slot,
            )
            st.session_state["tab2_model_artifact"] = artifact
            st.session_state["tab2_loss_history"] = artifact["loss_history"]
            st.session_state["tab2_generated_text"] = ""
            st.session_state["tab2_seed_text"] = suggest_seed_text(corpus_text)
            st.session_state["tab2_generate_length"] = 80
            append_experiment_record(artifact)
            status_slot.success("训练完成，模型已保存到 session_state。")

    artifact = st.session_state["tab2_model_artifact"]

    render_step_header(
        "Step 2",
        "用训练后的模型生成文本",
        "生成结果不一定完全通顺，但你可以观察模型是否学到了局部拼写模式、空格位置、常见词片段，以及不同采样策略带来的差异。",
    )
    st.caption(
        "如果结果只是在重复训练语料前缀，例如只生成到 `hello wor`，通常是因为 seed 太长、语料过短、生成长度太小，或者模型只学到了很局部的模式。建议缩短 seed、把生成长度调大，并优先使用 `multinomial sampling`。"
    )
    if artifact is None:
        st.info("请先训练模型，再使用下面的生成控件。")
        return

    seed_text = st.text_input(
        "seed 输入",
        key="tab2_seed_text",
    )
    generate_length = st.number_input(
        "生成长度",
        min_value=1,
        max_value=1000,
        step=1,
        key="tab2_generate_length",
    )
    generation_strategy = st.selectbox(
        "生成策略",
        options=["greedy", "multinomial sampling"],
        index=1,
        key="tab2_generation_strategy",
    )
    temperature = st.slider(
        "temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        key="tab2_temperature",
    )
    generate_button = st.button("生成文本", use_container_width=True)

    if generate_button:
        try:
            generated_text = generate_text_from_model(
                artifact=artifact,
                seed_text=seed_text,
                generate_length=int(generate_length),
                strategy="greedy" if generation_strategy == "greedy" else "multinomial",
                temperature=float(temperature),
            )
            st.session_state["tab2_generated_text"] = generated_text
        except ValueError as exc:
            st.error(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))

    st.text_area(
        "生成结果",
        value=st.session_state["tab2_generated_text"],
        height=200,
        disabled=True,
    )

    records = st.session_state["tab2_experiment_records"]
    render_step_header(
        "Step 3",
        "回看实验记录",
        "实验记录表可以帮助你比较不同模型、隐藏层大小和训练轮数对最终 loss 的影响。对于初学者来说，这能帮助建立“超参数会改变学习结果”的直觉。",
    )
    st.markdown("**实验对比记录**")
    if records:
        records_df = pd.DataFrame(records)
        st.dataframe(
            records_df.style.format({"final loss": "{:.6f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("还没有实验记录。")

    render_tab2_teaching_note()


@st.cache_resource(show_spinner=False)
def load_bert_resources():
    if AutoTokenizer is None or AutoModelForMaskedLM is None or torch is None:
        raise RuntimeError("运行 BERT 推理需要 transformers 和 PyTorch。")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    device = get_torch_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_gpt2_resources():
    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        raise RuntimeError("运行 GPT-2 推理需要 transformers 和 PyTorch。")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = get_torch_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_bert_mask(masked_sentence: str, top_k: int = 5) -> list[dict[str, object]]:
    tokenizer, model, device = load_bert_resources()
    encoded = tokenizer(masked_sentence, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        raise ValueError("请输入包含 `[MASK]` 的句子。")

    mask_index = int(mask_positions[0].item())

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, mask_index]
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probabilities, k=top_k)

    predictions = []
    for token_id, probability in zip(top_ids.tolist(), top_probs.tolist()):
        token = tokenizer.convert_ids_to_tokens(token_id)
        cleaned_token = token.replace("##", "")
        predictions.append(
            {
                "token": cleaned_token,
                "probability": probability,
            }
        )

    return predictions


def generate_gpt2_text(prompt: str, max_new_tokens: int = 60) -> str:
    results = generate_gpt2_samples(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    return results[0]


def generate_gpt2_samples(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    num_return_sequences: int,
) -> list[str]:
    tokenizer, model, device = load_gpt2_resources()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    return [
        tokenizer.decode(sequence, skip_special_tokens=True)
        for sequence in generated_ids
    ]


def compute_gpt2_sentence_metrics(sentence: str) -> dict[str, object]:
    tokenizer, model, device = load_gpt2_resources()
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = float(outputs.loss.item())

    token_count = int(attention_mask.sum().item())
    perplexity = float(math.exp(loss))
    return {
        "sentence": sentence,
        "token count": token_count,
        "loss": loss,
        "perplexity": perplexity,
    }


def load_tab4_examples() -> None:
    st.session_state["tab4_sentence_lines"] = (
        "The cat is sleeping on the warm sofa.\n"
        "Sofa green quickly planets under because sleeping triangle."
    )


def render_tab_transformers() -> None:
    st.subheader("Tab3 · BERT vs GPT-2")
    render_step_header(
        "Step 1",
        "对比 Masked LM 和 Causal LM",
        "这一页将 BERT 和 GPT-2 放在同一个主题下观察。BERT 负责填空，GPT-2 负责续写。建议先比较二者的任务形式，再阅读各自输出。",
    )

    if AutoTokenizer is None or torch is None:
        st.error("使用 Tab3 需要 transformers 和 PyTorch。")
        return

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**BERT：Masked LM**")
        st.info("建议先看 `[MASK]` 位置的 top-5 候选词及其概率。概率越高，表示模型越相信这个词最适合当前上下文。")
        bert_input = st.text_input(
            "包含 [MASK] 的句子",
            value="The capital of France is [MASK].",
            key="tab3_bert_input",
        )
        bert_button = st.button("预测掩码词", use_container_width=True)

        if bert_button:
            if "[MASK]" not in bert_input:
                st.warning("请在句子中加入 `[MASK]`。")
            else:
                try:
                    predictions = predict_bert_mask(bert_input, top_k=5)
                    prediction_df = pd.DataFrame(predictions)
                    st.dataframe(
                        prediction_df.style.format({"probability": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as exc:
                    st.error(
                        "加载或运行 `bert-base-uncased` 失败。通常这意味着模型尚未缓存到本地，或者下载过程失败。\n\n"
                        f"详细信息：{exc}"
                    )

    with right_col:
        st.markdown("**GPT-2：Causal LM**")
        st.info("建议把注意力放在续写方向上。不同的 `temperature`、`top_k`、`top_p` 会改变输出的稳定性与发散程度。")
        gpt2_prompt = st.text_area(
            "续写 Prompt",
            value="Once upon a time",
            height=120,
            key="tab3_gpt2_prompt",
        )
        max_new_tokens = st.number_input(
            "max_new_tokens",
            min_value=1,
            max_value=200,
            value=60,
            step=1,
            key="tab3_max_new_tokens",
        )
        temperature = st.slider(
            "temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="tab3_temperature",
        )
        top_k = st.number_input(
            "top_k",
            min_value=1,
            max_value=200,
            value=50,
            step=1,
            key="tab3_top_k",
        )
        top_p = st.slider(
            "top_p",
            min_value=0.1,
            max_value=1.0,
            value=0.95,
            step=0.05,
            key="tab3_top_p",
        )
        num_return_sequences = st.selectbox(
            "生成结果条数",
            options=[1, 2, 3],
            index=0,
            key="tab3_num_return_sequences",
        )
        gpt2_button = st.button("生成续写", use_container_width=True)

        if gpt2_button:
            if not gpt2_prompt.strip():
                st.warning("请输入非空的 GPT-2 prompt。")
            else:
                try:
                    generated_results = generate_gpt2_samples(
                        prompt=gpt2_prompt,
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        num_return_sequences=int(num_return_sequences),
                    )
                    for index, generated_text in enumerate(generated_results, start=1):
                        st.text_area(
                            f"生成结果 {index}",
                            value=generated_text,
                            height=180,
                            disabled=True,
                            key=f"tab3_generated_{index}",
                        )
                except Exception as exc:
                    st.error(
                        "加载或运行 `gpt2` 失败。通常这意味着模型尚未缓存到本地，或者下载过程失败。\n\n"
                        f"详细信息：{exc}"
                    )

    render_step_header(
        "Step 2",
        "在同一主题下比较“填空”和“续写”",
        "下面的示例帮助你把同一语义主题拆成两类任务。对于 NLP 初学者来说，这一步很重要，因为它能直观看出 BERT 和 GPT-2 的训练目标为什么不同。",
    )
    st.markdown("**同主题对比**")
    st.info(
        "BERT: `The capital of France is [MASK].`\n\n"
        "GPT-2: `The capital of France is`\n\n"
        "前者要求模型结合左右两侧上下文来填空；后者要求模型只依赖左侧历史继续往后续写。"
    )

    st.markdown("**简短解释**")
    st.markdown(
        "- BERT 是双向上下文填空模型，会同时利用目标词左边和右边的信息。\n"
        "- GPT-2 是从左到右的 Causal LM，每一步只根据前面的 token 继续生成后面的 token。"
    )


def render_tab_perplexity() -> None:
    st.subheader("Tab4 · perplexity")
    render_step_header(
        "Step 1",
        "理解 perplexity 在衡量什么",
        "这一页使用 GPT-2 对每一句英文分别计算 loss 与 perplexity。你可以把它理解为：模型看到这句话时到底有多意外。通常 PPL 越小，说明模型越不困惑。",
    )

    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        st.error("使用 Tab4 需要 transformers 和 PyTorch。")
        return

    example_button = st.button(
        "加载示例句子",
        on_click=load_tab4_examples,
    )
    _ = example_button

    sentence_block = st.text_area(
        "英文句子列表（每行一句）",
        value=(
            "The weather is pleasant and the sky is clear.\n"
            "Blue guitar sleeps rapidly under seven clouds."
        ),
        height=180,
        key="tab4_sentence_lines",
    )
    sort_by_ppl = st.checkbox("按 PPL 排序", value=True)

    if st.button("计算困惑度", use_container_width=True):
        sentences = [line.strip() for line in sentence_block.splitlines() if line.strip()]
        if not sentences:
            st.warning("请至少输入一句英文句子。")
            return

        try:
            rows = [compute_gpt2_sentence_metrics(sentence) for sentence in sentences]
        except Exception as exc:
            st.error(
                "加载或运行 `gpt2` 失败，因此无法计算 perplexity。通常这意味着模型尚未缓存到本地，或者下载过程失败。\n\n"
                f"详细信息：{exc}"
            )
            return

        results_df = pd.DataFrame(rows)
        if sort_by_ppl:
            results_df = results_df.sort_values("perplexity", ascending=True).reset_index(drop=True)

        st.info("建议先看 `perplexity` 排序，再看 `loss` 与 `token count`。如果一句话更自然、更接近 GPT-2 熟悉的语言分布，通常会得到更低的 PPL。")
        st.dataframe(
            results_df.style.format(
                {
                    "loss": "{:.4f}",
                    "perplexity": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**PPL 条形图**")
        st.caption("条形越短，表示模型对这句话越不困惑。")
        chart_df = results_df.copy()
        chart_df["label"] = [
            sentence if len(sentence) <= 40 else sentence[:37] + "..."
            for sentence in chart_df["sentence"]
        ]
        st.bar_chart(chart_df.set_index("label")["perplexity"])

    st.markdown("**简短解释**")
    st.markdown(
        "- PPL 越小，说明模型越不困惑。\n"
        "- 通顺、自然的句子通常会比随机拼接句子得到更低的 PPL。"
    )


def render_tab_placeholder(name: str) -> None:
    st.subheader(name)
    st.info("功能建设中。")


def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_academic_theme()

    sidebar_state = render_sidebar()
    apply_seed(int(sidebar_state["seed"]))

    render_intro()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "n-gram & smoothing",
            "train RNN",
            "BERT vs GPT-2",
            "perplexity",
        ]
    )

    with tab1:
        render_tab_ngram(bool(sidebar_state["show_teaching_tips"]))

    with tab2:
        render_tab_rnn()

    with tab3:
        render_tab_transformers()

    with tab4:
        render_tab_perplexity()


if __name__ == "__main__":
    main()
