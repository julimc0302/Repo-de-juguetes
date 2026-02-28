# Prompt strategy modules for the Research Copilot RAG system
from prompts.strategy1_delimiters import build_prompt as delimiters_prompt
from prompts.strategy2_json import build_prompt as json_prompt
from prompts.strategy3_fewshot import build_prompt as fewshot_prompt
from prompts.strategy4_cot import build_prompt as cot_prompt

__all__ = [
    "delimiters_prompt",
    "json_prompt",
    "fewshot_prompt",
    "cot_prompt",
]
