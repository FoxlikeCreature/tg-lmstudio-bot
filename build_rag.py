#!/usr/bin/env python3
"""
Скрипт для создания RAG-индекса из экспорта Telegram-сообщений.
Извлекает пары (сообщение_пользователя → ответ_бота), строит TF-IDF
на текстах триггеров и сохраняет в rag_index.json.
Использование: python build_rag.py
"""

import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

DATA_FILE = Path(__file__).parent / "result.json"
OUTPUT_FILE = Path(__file__).parent / "rag_index.json"
BOT_NAME = "𝓕𝓸𝔁𝓵𝓲𝓴𝓮 𝓒𝓻𝓮𝓪𝓽𝓾𝓻𝓮 🦊✨"
BOT_ID = "user1219005569"
CONTEXT_WINDOW_SEC = 180    # Окно для поиска триггера (3 минуты)
MAX_CONTEXT_MSGS = 3        # Сколько предшествующих сообщений брать как контекст
MAX_RESPONSE_LEN = 600      # Обрезать ответ если длиннее
MIN_RESPONSE_LEN = 20       # Игнорировать слишком короткие ответы

_URL_RE = re.compile(r"https?://\S+|t\.me/\S+|www\.\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FFFF\U00002700-\U000027BF\U0000FE00-\U0000FE0F]+"
)
_WORD_RE = re.compile(r"[а-яёa-z]", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.lower().strip()


def extract_text(msg: dict) -> str:
    """Извлечь текст из сообщения (поддерживает list-формат с entity)."""
    text = msg.get("text", "")
    if isinstance(text, list):
        text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in text
        )
    return text.strip() if isinstance(text, str) else ""


def parse_date(date_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def is_good_response(text: str) -> bool:
    """True если ответ содержит достаточно осмысленного текста."""
    cleaned = _URL_RE.sub("", text)
    cleaned = _EMOJI_RE.sub("", cleaned).strip()
    if len(cleaned) < MIN_RESPONSE_LEN:
        return False
    words = cleaned.split()
    if not words:
        return False
    word_count = sum(1 for w in words if _WORD_RE.search(w))
    return word_count / len(words) >= 0.3


def extract_qa_pairs(data: dict) -> list[dict]:
    """
    Для каждого сообщения бота найти предшествующие сообщения пользователей
    в окне времени и сформировать пару (триггер → ответ).

    Логика:
    - Идём по сообщениям каждого чата
    - Для каждого bot-сообщения ищем назад до MAX_CONTEXT_MSGS не-bot сообщений
      в пределах CONTEXT_WINDOW_SEC секунд
    - Эти сообщения образуют "контекст/запрос", bot-ответ — "ответ"
    - TF-IDF строится на контексте, при поиске возвращается ответ
    """
    pairs = []
    skipped_no_context = 0
    skipped_bad_response = 0

    for chat in data.get("chats", {}).get("list", []):
        chat_title = chat.get("title", chat.get("name", ""))
        messages = chat.get("messages", [])

        for i, msg in enumerate(messages):
            if msg.get("from_id") != BOT_ID:
                continue
            if msg.get("type") != "message":
                continue

            bot_text = extract_text(msg)
            if not is_good_response(bot_text):
                skipped_bad_response += 1
                continue

            if len(bot_text) > MAX_RESPONSE_LEN:
                bot_text = bot_text[:MAX_RESPONSE_LEN]

            bot_date = parse_date(msg.get("date", ""))
            context_parts = []

            # Ищем назад по сообщениям, пропуская бот-сообщения
            for j in range(i - 1, max(-1, i - 20), -1):
                prev = messages[j]
                if prev.get("from_id") == BOT_ID:
                    continue
                if prev.get("type") != "message":
                    continue
                prev_text = extract_text(prev)
                if not prev_text:
                    continue
                # Проверяем окно времени
                if bot_date:
                    prev_date = parse_date(prev.get("date", ""))
                    if prev_date:
                        if (bot_date - prev_date).total_seconds() > CONTEXT_WINDOW_SEC:
                            break
                context_parts.append(prev_text)
                if len(context_parts) >= MAX_CONTEXT_MSGS:
                    break

            if not context_parts:
                skipped_no_context += 1
                continue

            context_parts.reverse()
            query = " ".join(context_parts)

            pairs.append({
                "query": query,
                "response": bot_text,
                "chat": chat_title,
                "date": msg.get("date", ""),
            })

    print(f"  Пропущено (нет контекста): {skipped_no_context}")
    print(f"  Пропущено (плохой ответ): {skipped_bad_response}")
    return pairs


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Удалить пары с идентичным ответом (normalize → lowercase)."""
    seen: set[str] = set()
    result = []
    for pair in pairs:
        key = pair["response"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(pair)
    return result


def main():
    print(f"Чтение {DATA_FILE}...")
    t0 = time.time()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"JSON загружен за {time.time() - t0:.1f}с")

    print("Извлечение Q&A пар...")
    pairs = extract_qa_pairs(data)
    print(f"Извлечено: {len(pairs)} пар")

    pairs = deduplicate_pairs(pairs)
    print(f"После дедупликации: {len(pairs)} пар")

    if not pairs:
        print("Пар не найдено. Проверьте BOT_NAME и BOT_ID.")
        sys.exit(1)

    print("Построение TF-IDF на текстах триггеров...")
    # Нормализуем запросы — так же как при поиске в main.py
    query_texts = [normalize_text(p["query"]) for p in pairs]
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(query_texts)
    print(f"TF-IDF матрица: {tfidf_matrix.shape}")

    print("Сериализация...")
    tfidf_data = {
        "shape": [int(x) for x in tfidf_matrix.shape],
        "indices": [int(x) for x in tfidf_matrix.indices],
        "indptr": [int(x) for x in tfidf_matrix.indptr],
        "data": [float(x) for x in tfidf_matrix.data],
    }
    index = {
        "version": 2,
        "pair_count": len(pairs),
        "pairs": pairs,
        "tfidf": tfidf_data,
        "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf": [float(x) for x in vectorizer.idf_],
    }

    print(f"Сохранение в {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    total = time.time() - t0
    size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Готово за {total:.1f}с. Файл: {size:.1f} МБ")


if __name__ == "__main__":
    main()
