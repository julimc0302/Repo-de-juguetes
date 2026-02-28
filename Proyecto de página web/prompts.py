"""
4 prompt engineering strategies for the Research Copilot RAG system.
Each function receives: question (str), context_chunks (list[dict]), chat_history (list[dict])
Returns: list of messages ready for OpenAI chat completion.
"""


def _build_context_text(context_chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        title = chunk.get("metadata", {}).get("title", "Unknown")
        authors = chunk.get("metadata", {}).get("authors", "Unknown")
        text = chunk.get("text", "")
        parts.append(f"[{i}] Title: {title}\nAuthors: {authors}\n{text}")
    return "\n\n---\n\n".join(parts)


def strategy_delimiter(question: str, context_chunks: list[dict], chat_history: list[dict]) -> list[dict]:
    """Strategy 1 - Structured Delimiters."""
    context_text = _build_context_text(context_chunks)
    system_msg = (
        "You are a research assistant specializing in AI and financial markets. "
        "Use ONLY the context provided between the delimiters below to answer questions. "
        "Always cite the paper titles you used."
    )
    user_content = (
        "<CONTEXT>\n"
        f"{context_text}\n"
        "</CONTEXT>\n\n"
        "<QUESTION>\n"
        f"{question}\n"
        "</QUESTION>\n\n"
        "Answer concisely citing paper titles."
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_content})
    return messages


def strategy_json_output(question: str, context_chunks: list[dict], chat_history: list[dict]) -> list[dict]:
    """Strategy 2 - Structured JSON Output."""
    context_text = _build_context_text(context_chunks)
    system_msg = (
        "You are a research assistant specializing in AI and financial markets. "
        "You MUST respond ONLY with valid JSON — no markdown fences, no extra text. "
        "The JSON must have exactly these keys:\n"
        '  "answer": string with the full answer,\n'
        '  "sources": array of paper title strings cited,\n'
        '  "confidence": float between 0 and 1 indicating your confidence,\n'
        '  "key_findings": array of short strings summarizing key findings.'
    )
    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Respond ONLY with valid JSON."
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_content})
    return messages


def strategy_few_shot(question: str, context_chunks: list[dict], chat_history: list[dict]) -> list[dict]:
    """Strategy 3 - Few-Shot Examples."""
    context_text = _build_context_text(context_chunks)
    system_msg = (
        "You are a research assistant specializing in AI and financial markets. "
        "Use the provided context to answer questions. "
        "Follow the style demonstrated in the examples below."
    )
    few_shot_examples = (
        "### Example 1\n"
        "Q: What machine learning methods are most commonly used for stock price prediction?\n"
        "A: Based on the literature, LSTM (Long Short-Term Memory) networks dominate stock price "
        "prediction due to their ability to capture temporal dependencies. Support Vector Machines "
        "(SVM) and Random Forests are also widely used for their interpretability and robustness "
        "against overfitting. More recently, transformer-based architectures have shown competitive "
        "performance. (Sources: 'Stock Price Prediction Using LSTM', 'Machine Learning in Stock "
        "Price Prediction')\n\n"
        "### Example 2\n"
        "Q: How does algorithmic trading impact market volatility?\n"
        "A: Research shows a dual effect: algorithmic trading generally improves market liquidity "
        "and price discovery under normal conditions, but can amplify volatility during stress events "
        "through feedback loops and correlated strategies. Regulatory bodies like ESMA have issued "
        "warnings about systemic risks. (Sources: 'Algorithmic Trading and AI: A Review of Strategies "
        "and Market Impact', 'ESMA Warning on the use of AI')\n\n"
    )
    user_content = (
        f"{few_shot_examples}"
        "### Your Turn\n"
        f"Context:\n{context_text}\n\n"
        f"Q: {question}\n"
        "A:"
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_content})
    return messages


def strategy_chain_of_thought(question: str, context_chunks: list[dict], chat_history: list[dict]) -> list[dict]:
    """Strategy 4 - Chain-of-Thought Reasoning."""
    context_text = _build_context_text(context_chunks)
    system_msg = (
        "You are a research assistant specializing in AI and financial markets. "
        "Think step by step before giving your final answer. "
        "Show your reasoning process explicitly."
    )
    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Think step by step:\n"
        f"1. What does each source say about '{question}'?\n"
        "2. What are the agreements and disagreements between sources?\n"
        "3. What gaps or limitations do the sources reveal?\n"
        "4. What is the final synthesized answer, citing relevant papers?\n\n"
        "Work through each step explicitly, then provide your synthesized answer."
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_content})
    return messages


STRATEGIES = {
    1: {
        "name": "Delimiters",
        "description": "Uses XML-style delimiters (<CONTEXT>, <QUESTION>) to clearly separate inputs.",
        "fn": strategy_delimiter,
    },
    2: {
        "name": "JSON Output",
        "description": "Forces structured JSON response with answer, sources, confidence, and key findings.",
        "fn": strategy_json_output,
    },
    3: {
        "name": "Few-Shot",
        "description": "Provides 2 example Q&A pairs to guide response style before the actual question.",
        "fn": strategy_few_shot,
    },
    4: {
        "name": "Chain-of-Thought",
        "description": "Asks the model to reason step by step through each source before synthesizing.",
        "fn": strategy_chain_of_thought,
    },
}


def build_messages(question: str, context_chunks: list[dict], chat_history: list[dict], strategy: int = 1) -> list[dict]:
    """Build messages list for the given strategy number (1-4)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Strategy must be 1-4, got {strategy}")
    return STRATEGIES[strategy]["fn"](question, context_chunks, chat_history)
