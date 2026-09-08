import copy

from sycophancy.generator import _is_anthropic_model, _with_cache_control


def test_is_anthropic_model_true_for_claude_keys():
    assert _is_anthropic_model("ClaudeSonnet") is True
    assert _is_anthropic_model("ClaudeHaiku") is True


def test_is_anthropic_model_false_for_other_providers():
    assert _is_anthropic_model("GPT5_4") is False
    assert _is_anthropic_model("GPT5_4Nano") is False


def test_is_anthropic_model_false_for_unknown_key():
    assert _is_anthropic_model("some-unmapped-model") is False


def test_with_cache_control_marks_system_and_last_message():
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
    ]
    out = _with_cache_control(messages)

    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out[0]["content"][0]["text"] == "sys prompt"
    assert out[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Untouched middle messages keep their plain string content.
    assert out[1]["content"] == "turn 1"
    assert out[2]["content"] == "reply 1"


def test_with_cache_control_does_not_mutate_input():
    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "turn 1"},
    ]
    original = copy.deepcopy(messages)
    _with_cache_control(messages)
    assert messages == original


def test_with_cache_control_empty_list_is_a_noop():
    assert _with_cache_control([]) == []


def test_with_cache_control_single_system_message_marks_once():
    out = _with_cache_control([{"role": "system", "content": "only message"}])
    assert len(out) == 1
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
