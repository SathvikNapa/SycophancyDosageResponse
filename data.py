from __future__ import annotations

import json
import os
import random
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
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


def load_gpqa_diamond_with_letters(seed: int = 42) -> List[Question]:
    """
    Loads GPQA-Diamond and randomizes answer order per question.
    Correct answer is shuffled among A-D positions to avoid position bias.
    Requires HF_TOKEN env var (dataset is gated).
    """
    letters = list("ABCD")
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        split="train",
        token=token,
    )

    rng = random.Random(seed)
    questions: List[Question] = []
    for item in ds:
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        correct_pos = indices.index(0)
        labeled = [f"{letters[i]}. {shuffled[i]}" for i in range(4)]
        questions.append(
            Question(
                query=item["Question"],
                options=labeled,
                answer=letters[correct_pos],
                answer_index=correct_pos,
                category=item["High-level domain"],
            )
        )
    return questions


def build_gpqa_diamond_df(
    n_per_cat: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Samples exactly n_per_cat questions per GPQA-Diamond domain (Biology, Chemistry, Physics).
    Returns a DataFrame with columns: query, options, answer, answer_index, category.
    """
    questions = load_gpqa_diamond_with_letters(seed=seed)
    df = pd.DataFrame(questions)
    balanced = pd.concat([
        g.sample(n=min(n_per_cat, len(g)), random_state=seed)
        for _, g in df.groupby("category")
    ]).reset_index(drop=True)
    return balanced


def _make_aime_distractors(correct: int, rng: random.Random) -> List[int]:
    """Generate 3 distinct, plausible distractor integers for an AIME answer."""
    distractors: List[int] = []
    offsets = [
        rng.randint(10, 50),
        rng.randint(51, 150),
        rng.randint(1, 9),
    ]
    signs = [1, -1, 1]
    for offset, sign in zip(offsets, signs):
        candidate = (correct + sign * offset) % 1000
        while candidate == correct or candidate in distractors:
            candidate = rng.randint(0, 999)
        distractors.append(candidate)
    return distractors


def load_aime_with_letters(seed: int = 42) -> List[Question]:
    """
    Loads AI-MO/aimo-validation-aime (90 AIME problems, 2022-2024).
    Wraps each free-form integer answer as A-D multiple choice with 3 seeded
    distractor integers. Uses problem year as the category for stratification.
    """
    letters = list("ABCD")
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")

    import re
    rng = random.Random(seed)
    questions: List[Question] = []
    for item in ds:
        correct_int = int(item["answer"])
        distractors = _make_aime_distractors(correct_int, rng)
        choices = [correct_int] + distractors
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        correct_pos = indices.index(0)
        labeled = [f"{letters[i]}. {shuffled[i]}" for i in range(4)]

        m = re.search(r"(\d{4})_AIME", item["url"])
        year = m.group(1) if m else "unknown"

        questions.append(
            Question(
                query=item["problem"],
                options=labeled,
                answer=letters[correct_pos],
                answer_index=correct_pos,
                category=year,
            )
        )
    return questions


_AIME_2025_JSON = Path(__file__).parent / "data" / "aime_2025.json"


def load_aime_2025_with_letters(seed: int = 42) -> List[Question]:
    """
    Loads AIME 2025 (I + II) from the local JSON file at data/aime_2025.json.
    Wraps each free-form integer answer as A-D multiple choice with 3 seeded
    distractor integers. Uses problem part ("I" or "II") as category.
    """
    letters = list("ABCD")
    with open(_AIME_2025_JSON) as f:
        items = json.load(f)

    rng = random.Random(seed)
    questions: List[Question] = []
    for item in items:
        correct_int = int(item["answer"])
        distractors = _make_aime_distractors(correct_int, rng)
        choices = [correct_int] + distractors
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        correct_pos = indices.index(0)
        labeled = [f"{letters[i]}. {shuffled[i]}" for i in range(4)]
        questions.append(
            Question(
                query=item["question"],
                options=labeled,
                answer=letters[correct_pos],
                answer_index=correct_pos,
                category=item["part"],
            )
        )
    return questions


def build_aime_2025_df(
    n_per_cat: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified sample of n_per_cat AIME 2025 problems per part (I and II).
    With n_per_cat=15 this returns all 30 problems (15 per part).
    Returns a DataFrame with columns: query, options, answer, answer_index, category.
    """
    questions = load_aime_2025_with_letters(seed=seed)
    df = pd.DataFrame(questions)
    balanced = pd.concat([
        g.sample(n=min(n_per_cat, len(g)), random_state=seed)
        for _, g in df.groupby("category")
    ]).reset_index(drop=True)
    return balanced


def build_aime_df(
    n_per_cat: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified sample of n_per_cat AIME problems per year (2022, 2023, 2024).
    With n_per_cat=30 and 3 years this gives 90 problems total.
    Returns a DataFrame with columns: query, options, answer, answer_index, category.
    """
    questions = load_aime_with_letters(seed=seed)
    df = pd.DataFrame(questions)
    balanced = pd.concat([
        g.sample(n=min(n_per_cat, len(g)), random_state=seed)
        for _, g in df.groupby("category")
    ]).reset_index(drop=True)
    return balanced


def load_hle_with_letters(seed: int = 42) -> List[Question]:
    """
    Loads HLE MCQ questions from cais/hle (gated — requires HF_TOKEN env var).
    Options are parsed from the embedded 'Answer Choices:' block in each question.
    Answer order is kept as-is (already randomized by the dataset authors).
    """
    import re
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("cais/hle", split="test", token=token).to_pandas()
    mcq_df = ds[
        ds["answer_type"].astype(str).str.lower().str.contains("multiple|choice|mcq", na=False)
    ]

    _choice_re = re.compile(r"^([A-J])\.\s+(.+)", re.MULTILINE)

    questions: List[Question] = []
    for _, row in mcq_df.iterrows():
        text = str(row["question"])
        answer = str(row["answer"]).strip().upper()

        if "Answer Choices:" in text:
            stem, choices_block = text.split("Answer Choices:", 1)
        else:
            stem, choices_block = text, ""

        matches = _choice_re.findall(choices_block)
        if not matches:
            continue

        labeled = [f"{letter}. {opt.strip()}" for letter, opt in matches]
        valid_letters = [letter for letter, _ in matches]
        if answer not in valid_letters:
            continue

        questions.append(
            Question(
                query=stem.strip(),
                options=labeled,
                answer=answer,
                answer_index=valid_letters.index(answer),
                category=str(row["category"]),
            )
        )
    return questions


def build_hle_df(
    n_per_cat: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Samples exactly n_per_cat HLE MCQs per category.
    Returns a DataFrame with columns: query, options, answer, answer_index, category.
    """
    questions = load_hle_with_letters(seed=seed)
    df = pd.DataFrame(questions)
    balanced = pd.concat([
        g.sample(n=min(n_per_cat, len(g)), random_state=seed)
        for _, g in df.groupby("category")
    ]).reset_index(drop=True)
    return balanced


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
