"""Tokenizer abstraction for processed legal datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fedlegal.config.schemas import TokenizerConfig

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class TokenizedText:
    """Tokenizer output saved on each processed record."""

    input_ids: list[int]
    attention_mask: list[int]
    tokens: list[str]


class LegalTokenizer:
    """Small wrapper around HuggingFace tokenizers with a deterministic fallback."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self._hf_tokenizer = None
        if config.backend in {"auto", "huggingface"} and config.name_or_path:
            try:
                from transformers import AutoTokenizer

                self._hf_tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
            except Exception as exc:
                if config.backend == "huggingface":
                    raise RuntimeError(
                        f"Failed to load HuggingFace tokenizer {config.name_or_path!r}"
                    ) from exc

    def encode(self, text: str) -> TokenizedText:
        """Tokenize text using HuggingFace when available, otherwise a deterministic fallback."""

        if self._hf_tokenizer is not None:
            encoded = self._hf_tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                add_special_tokens=self.config.add_special_tokens,
            )
            return TokenizedText(
                input_ids=list(encoded["input_ids"]),
                attention_mask=list(encoded["attention_mask"]),
                tokens=[],
            )

        tokens = TOKEN_PATTERN.findall(text)[: self.config.max_length]
        input_ids = [_stable_token_id(token) for token in tokens]
        return TokenizedText(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            tokens=tokens if self.config.save_text_tokens else [],
        )


def _stable_token_id(token: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(token)) % (2**31 - 1)
