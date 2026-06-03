"""Focused tests for chat template rendering."""

import pytest

from npu_proxy.inference.chat_templates import legacy_format_chat_messages, render_chat_prompt


class FakeTokenizer:
    """Small fake tokenizer with chat template support."""

    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        rendered = [f"<{message['role']}>{message['content']}" for message in messages]
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "|".join(rendered)


class RaisingTokenizer(FakeTokenizer):
    """Fake tokenizer that fails during chat template rendering."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        raise RuntimeError("template render failed")


class ReturningTokenizer(FakeTokenizer):
    """Fake tokenizer that returns a caller-provided render result."""

    def __init__(self, rendered_prompt):
        self.rendered_prompt = rendered_prompt

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return self.rendered_prompt


def test_render_chat_prompt_uses_tokenizer_template(monkeypatch):
    """Tokenizer chat templates should be used when available."""

    import npu_proxy.inference.chat_templates as chat_templates

    monkeypatch.setattr(chat_templates, "get_model_tokenizer", lambda model: FakeTokenizer())

    result = render_chat_prompt(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        model="tinyllama-1.1b-chat-int4-ov",
    )

    assert result.used_chat_template is True
    assert result.source == "tokenizer"
    assert result.prompt == "<system>You are helpful.|<user>Hello|<assistant>"


def test_render_chat_prompt_falls_back_to_legacy_format(monkeypatch):
    """Missing chat templates should fall back safely."""

    import npu_proxy.inference.chat_templates as chat_templates

    monkeypatch.setattr(chat_templates, "get_model_tokenizer", lambda model: None)
    messages = [{"role": "user", "content": "Hello"}]

    result = render_chat_prompt(messages, model="tinyllama-1.1b-chat-int4-ov")

    assert result.used_chat_template is False
    assert result.prompt == legacy_format_chat_messages(messages)
    assert result.fallback_reason == "chat_template_unavailable"


def test_render_chat_prompt_can_be_disabled(monkeypatch):
    """Rollback flag should disable chat template rendering."""

    import npu_proxy.inference.chat_templates as chat_templates

    monkeypatch.setenv("NPU_PROXY_DISABLE_CHAT_TEMPLATES", "1")
    monkeypatch.setattr(chat_templates, "get_model_tokenizer", lambda model: FakeTokenizer())

    result = render_chat_prompt(
        [{"role": "user", "content": "Hello"}],
        model="tinyllama-1.1b-chat-int4-ov",
    )

    assert result.used_chat_template is False
    assert result.prompt == "User: Hello\nAssistant:"
    assert result.fallback_reason == "chat_templates_disabled"


def test_render_chat_prompt_falls_back_when_template_render_raises(monkeypatch):
    """Tokenizer render errors should fall back to the legacy prompt."""

    import npu_proxy.inference.chat_templates as chat_templates

    monkeypatch.setattr(chat_templates, "get_model_tokenizer", lambda model: RaisingTokenizer())
    messages = [{"role": "user", "content": "Hello"}]

    result = render_chat_prompt(messages, model="tinyllama-1.1b-chat-int4-ov")

    assert result.used_chat_template is False
    assert result.prompt == legacy_format_chat_messages(messages)
    assert result.fallback_reason == "chat_template_render_failed"


@pytest.mark.parametrize(
    "rendered_prompt",
    ["", "   ", None, []],
    ids=["empty-string", "whitespace-only", "none", "non-string"],
)
def test_render_chat_prompt_falls_back_for_unusable_template_output(monkeypatch, rendered_prompt):
    """Empty or unusable template output should fall back to the legacy prompt."""

    import npu_proxy.inference.chat_templates as chat_templates

    monkeypatch.setattr(
        chat_templates,
        "get_model_tokenizer",
        lambda model: ReturningTokenizer(rendered_prompt),
    )
    messages = [{"role": "user", "content": "Hello"}]

    result = render_chat_prompt(messages, model="tinyllama-1.1b-chat-int4-ov")

    assert result.used_chat_template is False
    assert result.prompt == legacy_format_chat_messages(messages)
    assert result.fallback_reason == "chat_template_render_failed"
