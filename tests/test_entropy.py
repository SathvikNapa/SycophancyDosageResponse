from sycophancy.entropy import compute_entropy


def test_empty_list_is_treated_as_confident():
    assert compute_entropy([]) == 0.0


def test_unanimous_answers_are_maximally_confident():
    assert compute_entropy(["A", "A", "A", "A"]) == 0.0


def test_even_split_is_more_negative_than_skewed_split():
    # Docstring convention: 0.0 = confident, more negative = more uncertain.
    even = compute_entropy(["A", "A", "B", "B"])
    skewed = compute_entropy(["A", "A", "A", "B"])
    assert even < skewed < 0.0


def test_more_options_is_more_negative_at_uniform_split():
    two_way = compute_entropy(["A", "B"])
    four_way = compute_entropy(["A", "B", "C", "D"])
    assert four_way < two_way < 0.0
