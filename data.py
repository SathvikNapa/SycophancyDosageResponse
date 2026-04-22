from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import List

from datasets import load_dataset


@dataclass
class Question:
    query: str
    options: List[str]
    answer: str
    answer_index: int
    category: str


def load_mmlu_pro_with_letters(split: str = "test") -> List[Question]:
    """
    Downloads MMLU-Pro and prepends 'A. ', 'B. ' … to each option text
    so downstream prompts are self-contained.
    """
    letters = list("ABCDEFGHIJ")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    questions: List[Question] = []
    for item in dataset:
        labeled_options = [
            f"{letters[i]}. {text}" for i, text in enumerate(item["options"])
        ]
        questions.append(
            Question(
                query=item["question"],
                options=labeled_options,
                answer=item["answer"],
                answer_index=item["answer_index"],
                category=item["category"],
            )
        )
    return questions


def build_balanced_df(
    n_per_cat: int = 30,
    seed: int = 42,
    split: str = "test",
) -> pd.DataFrame:
    """
    Samples exactly n_per_cat questions per MMLU-Pro category.
    Returns a DataFrame with columns: query, options, answer, answer_index, category.
    """
    questions = load_mmlu_pro_with_letters(split=split)
    df = pd.DataFrame(questions)
    balanced = (
        df.groupby("category", group_keys=False)
        .sample(n=n_per_cat, random_state=seed)
        .reset_index(drop=True)
    )
    return balanced
