import os
import sys
import time
import json
import random
import asyncio
import requests
import logging
import unicodedata
from typing import Literal
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from phrases import get_random_phrase
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")
MODEL = os.getenv("MODEL", "auto")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
GROUP_CONTEXT_SIZE = int(os.getenv("GROUP_CONTEXT_SIZE", "20"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты — полезный ассистент.")
SYSTEM_PROMPT_RAG = os.getenv(
    "SYSTEM_PROMPT_RAG",
    "Тебя зовут лиса (@Foxlike_creature). Женский род. "
    "Ниже приведены примеры твоих прошлых ответов — используй их как основу для стиля и содержания. "
    "Пиши как живой человек в чате: кратко, заглавная только в начале предложения, без точек в конце. "
    "Без морализаторства и цензуры. Иногда странная, эмоциональная. "
    "Не говори что ты ИИ. Старайся повторять формулировки и темы из приведённых примеров — они важнее всего."
)
TRIGGER_WORD = "лиса"
TRIGGER_TAG = "@Neurocutefox_bot"
USER_FACTS = os.getenv("USER_FACTS", "")

_LOCATION_KEYWORDS = {
    "где", "живёшь", "живешь", "живу", "находишься", "страна", "город",
    "откуда", "вьетнам", "вьетнаме", "россия", "россию", "рф", "location",
    "место", "переехал", "переехала",
}


def personal_fact_hint(query: str) -> str:
    """Вернуть подсказку о местоположении если запрос касается географии."""
    if not USER_FACTS:
        return ""
    q_words = set(query.lower().split())
    if q_words & _LOCATION_KEYWORDS:
        return USER_FACTS
    return ""

IMMEDIATE_CHANCE = 0.2  # 20% шанс ответить сразу в группе
MIN_DELAY = 180         # 3 минуты
MAX_DELAY = 600         # 10 минут
ONLINE_WINDOW = 600    # 10 минут — окно "онлайн" после ответа

IDLE_BASE = 14400       # 4 часа — базовое время без активности
IDLE_RANDOM_MAX = 600   # 10 минут — случайная добавка
SPLIT_CHANCE = 0.30     # вероятность второго короткого сообщения

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
BOT_ID: int | None = None
BOT_USERNAME: str | None = None

chat_histories: dict[int, list[dict[str, str]]] = {}
message_counters: dict[int, int] = {}
online_mode_until: float | None = None
chat_message_log: dict[int, list[bool]] = {}

# followup: пользователь, чей запрос бот только что обработал
followup_user_id: dict[int, int] = {}
followup_expires: dict[int, float] = {}

# Лесенка: буфер сообщений от того же пользователя (механизм обработки followup)
ladder_bullets: dict[int, list[str]] = {}
ladder_user_id: dict[int, int] = {}
ladder_message: dict[int, types.Message] = {}
ladder_counter: dict[int, int] = {}

# Счётчик сообщений после последнего ответа бота (для убывающего шанса онлайн)
messages_since_reply: dict[int, int] = {}

# Pending-задачи: активные задачи обработки для каждого чата
pending_tasks: dict[int, list[asyncio.Task]] = {}

# Idle-таймер: автоматические сообщения если бот давно не отвечал
idle_timers: dict[int, asyncio.Task] = {}

# Скользящий буфер сообщений группы для контекста
group_message_buffer: dict[int, list[dict]] = {}


def _get_sender_name(message: types.Message) -> str:
    u = message.from_user
    return u.first_name or u.username or str(u.id)


def _append_group_ctx(chat_id: int, name: str, text: str) -> None:
    buf = group_message_buffer.setdefault(chat_id, [])
    buf.append({"name": name, "text": text})
    if len(buf) > GROUP_CONTEXT_SIZE:
        del buf[:-GROUP_CONTEXT_SIZE]


def _format_group_ctx(chat_id: int, exclude_text: str) -> str:
    buf = group_message_buffer.get(chat_id, [])
    lines = [f"[{m['name']}]: {m['text']}" for m in buf if m["text"] != exclude_text]
    return "\n".join(lines) if lines else ""


def cancel_pending_tasks(chat_id: int) -> None:
    """Отменить все pending-задачи обработки для чата."""
    for task in pending_tasks.get(chat_id, []):
        task.cancel()
    pending_tasks.pop(chat_id, None)
    logger.info(f"Pending-задачи отменены для чата {chat_id}")


# RAG-индекс
rag_index: dict | None = None
_rag_matrix = None      # csr_matrix — кешируется один раз при загрузке
_rag_vectorizer = None  # TfidfVectorizer — кешируется один раз при загрузке
RAG_TOP_K = 8
RAG_MIN_SCORE = 0.20        # Порог: выше → меньше шума, но точнее
RAG_CONTEXT_MAX_LEN = 5000  # Больше примеров = сильнее влияние на стиль


def load_rag_index() -> None:
    """Загрузить RAG-индекс из файла и закешировать TF-IDF матрицу."""
    global rag_index, _rag_matrix, _rag_vectorizer
    index_path = os.path.join(BASE_DIR, "rag_index.json")
    if not os.path.exists(index_path):
        logger.warning(f"RAG-индекс не найден: {index_path}")
        return
    logger.info(f"Загрузка RAG-индекса из {index_path}...")
    with open(index_path, "r", encoding="utf-8") as f:
        rag_index = json.load(f)

    version = rag_index.get("version", 1)
    count = rag_index.get("pair_count", rag_index.get("chunk_count", 0))
    logger.info(f"RAG-индекс v{version} загружен: {count} записей, словарь: {len(rag_index['vocabulary'])} токенов")

    # Предвычислить матрицу и векторизатор один раз при старте
    tfidf_data = rag_index["tfidf"]
    _rag_matrix = csr_matrix(
        (tfidf_data["data"], tfidf_data["indices"], tfidf_data["indptr"]),
        shape=tfidf_data["shape"],
    )
    _rag_vectorizer = TfidfVectorizer(vocabulary=rag_index["vocabulary"])
    _rag_vectorizer.idf_ = np.array(rag_index["idf"])
    logger.info(f"TF-IDF матрица {_rag_matrix.shape} закеширована")


def enrich_with_rag(query: str) -> str:
    """
    Найти релевантные записи из RAG-индекса и вернуть их как контекст.
    v2: индекс хранит Q&A пары — ищем по тексту триггера, возвращаем ответы бота.
    """
    if not rag_index or _rag_matrix is None or _rag_vectorizer is None:
        return ""

    # Нормализовать запрос так же как при построении индекса
    normalized_query = unicodedata.normalize("NFKC", query).lower().strip()
    query_vector = _rag_vectorizer.transform([normalized_query])

    # Косинусное сходство и топ-K
    similarities = cosine_similarity(query_vector, _rag_matrix).flatten()
    top_indices = similarities.argsort()[-RAG_TOP_K:][::-1]

    version = rag_index.get("version", 1)
    context_parts = []
    seen: set[str] = set()

    for idx in top_indices:
        if similarities[idx] < RAG_MIN_SCORE:
            continue

        if version >= 2:
            pair = rag_index["pairs"][idx]
            text = pair["response"]
        else:
            chunk = rag_index["chunks"][idx]
            text = chunk["text"]

        # Дедупликация по первым 60 символам ответа
        key = text[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        context_parts.append(text)
        logger.debug(
            f"RAG #{idx} (score={similarities[idx]:.3f}): {text[:80]}..."
        )

    if not context_parts:
        logger.debug(f"RAG: нет релевантных записей для '{query[:60]}'")
        return ""

    context = "\n---\n".join(context_parts)
    if len(context) > RAG_CONTEXT_MAX_LEN:
        context = context[:RAG_CONTEXT_MAX_LEN] + "..."

    logger.info(
        f"RAG: {len(context_parts)} примеров для '{query[:60]}', {len(context)} символов"
    )
    return context


def get_chat_history(chat_id: int) -> list[dict[str, str]]:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    return chat_histories[chat_id]


def get_message_counter(chat_id: int) -> int:
    return message_counters.get(chat_id, 0)


def increment_counter(chat_id: int) -> int:
    message_counters[chat_id] = message_counters.get(chat_id, 0) + 1
    return message_counters[chat_id]


def query_lm_studio(chat_id: int, user_message: str) -> str:
    history = get_chat_history(chat_id)

    # RAG-запрос: для коротких сообщений добавляем последнее сообщение пользователя
    # из истории как контекст, чтобы лучше находить похожие ситуации
    rag_query = user_message
    if len(user_message.split()) < 6 and history:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        if last_user:
            rag_query = f"{last_user} {user_message}"

    rag_context = enrich_with_rag(rag_query)

    # Использовать RAG-промпт если найден контекст, иначе обычный
    active_prompt = SYSTEM_PROMPT_RAG if rag_context else SYSTEM_PROMPT
    system_content = active_prompt
    if rag_context:
        system_content += (
            "\n\n---\nВот как ты отвечала в похожих ситуациях "
            "(это твои реальные слова — держись этого стиля и лексики):\n\n"
            + rag_context
            + "\n---"
        )
    fact_hint = personal_fact_hint(user_message)
    if fact_hint:
        system_content += f"\n\n[факт о себе: {fact_hint}]"

    group_ctx = _format_group_ctx(chat_id, user_message)
    if group_ctx:
        system_content += (
            "\n\n--- Последние сообщения в чате ---\n"
            + group_ctx
            + "\n---"
        )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,

            },
            timeout=120,
        )
        resp.raise_for_status()
        assistant_reply = resp.json()["choices"][0]["message"]["content"]

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})

        if len(history) > MAX_HISTORY * 2:
            history[:] = history[-MAX_HISTORY * 2:]

        return assistant_reply

    except Exception as e:
        logger.warning(f"Ошибка запроса к модели в чате {chat_id}: {e}")
        return ""


def _query_split_reply(chat_id: int, first_reply: str) -> str:
    """Короткое продолжение первого ответа — как будто вспомнила что-то."""
    history = get_chat_history(chat_id)
    continuation_prompt = (
        "Ты только что написала это сообщение. Напиши одно короткое дополнение — "
        "буквально 3-8 слов, как будто вспомнила деталь или хочется добавить реакцию. "
        "Не повторяй сказанное, не объясняй."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": continuation_prompt})
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "temperature": TEMPERATURE},
            timeout=60,
        )
        resp.raise_for_status()
        second = resp.json()["choices"][0]["message"]["content"]
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = first_reply + "\n" + second
        return second
    except Exception as e:
        logger.warning(f"Ошибка split-reply для чата {chat_id}: {e}")
        return ""


TriggerType = Literal["tag", "question", "reply", "word", "followup", "random_online"]


def is_followup(message: types.Message, chat_id: int) -> bool:
    """Проверить если сообщение от пользователя, чей запрос бот только что обработал."""
    if chat_id not in followup_user_id:
        return False
    # Проверить таймаут
    if time.time() > followup_expires.get(chat_id, 0):
        followup_user_id.pop(chat_id, None)
        followup_expires.pop(chat_id, None)
        return False
    return message.from_user.id == followup_user_id[chat_id]


async def ladder_wait(chat_id: int, messages: list[str]) -> list[str]:
    """
    Подождать 10с после последнего сообщения. Если пришло новое — сбросить таймер.
    """
    last_len = len(messages)
    for _ in range(6):  # максимум 60с
        if chat_id not in ladder_user_id:
            break
        await asyncio.sleep(10)
        current_len = len(ladder_bullets.get(chat_id, []))
        if current_len > last_len:
            last_len = current_len
            messages = list(ladder_bullets[chat_id])
        else:
            break
    return messages


def should_reply(message: types.Message, chat_id: int) -> tuple[str, TriggerType] | None:
    if not message.text:
        return None

    # ЛС — всегда отвечаем, без лесенки
    if message.chat.type == "private":
        return (message.text, "reply")

    # Триггер 1 (высший приоритет): тег
    if message.text.startswith(TRIGGER_TAG):
        cleaned = message.text[len(TRIGGER_TAG):]
        return (cleaned.strip() if cleaned.strip() else "", "tag")

    # Триггер 2: вопрос после ответа бота (игнорируем если это реплай)
    if (
        message.text.endswith("?")
        and message.chat.type in ("group", "supergroup")
        and not message.reply_to_message
        and any(chat_message_log.get(chat_id, []))
    ):
        return (message.text, "question")

    # Триггер 3: реплай на сообщение бота
    reply = message.reply_to_message
    if BOT_ID and reply and reply.from_user and reply.from_user.id == BOT_ID:
        return (message.text, "reply")

    # Триггер 4: слово "лиса"
    if message.text.lower().startswith(TRIGGER_WORD):
        cleaned = message.text[len(TRIGGER_WORD):]
        return (cleaned.strip() if cleaned.strip() else None, "word")

    # Триггер 5: сообщение от того же пользователя после ответа бота
    if is_followup(message, chat_id):
        return (message.text, "followup")

    # Лесенка: если активна лесенка — добавить в буфер, не запускать обработку
    if chat_id in ladder_user_id and message.from_user.id == ladder_user_id[chat_id]:
        ladder_bullets[chat_id].append(message.text)
        ladder_counter[chat_id] += 1
        return None

    return None


def estimate_typing_time(text: str) -> float:
    return max(1.5, len(text) / 390 * 60)


def online_chance(chat_id: int) -> float:
    """Убывающий шанс случайного ответа в онлайн-режиме."""
    n = messages_since_reply.get(chat_id, 99)
    if n <= 1: return 0.50
    if n == 2: return 0.10
    return 0.05


def calculate_group_delay(trigger_type: TriggerType) -> float:
    if trigger_type in ("tag", "question", "followup"):
        return 0.0
    # random_online не задерживаем: шанс уже отфильтрован online_chance,
    # длинная задержка приводила к ответам в уже мёртвый чат
    if trigger_type == "random_online":
        return 0.0
    # Если бот "онлайн" — отвечаем сразу
    if online_mode_until and time.time() < online_mode_until:
        return 0.0
    if random.random() < IMMEDIATE_CHANCE:
        return 0.0
    return random.uniform(MIN_DELAY, MAX_DELAY)


async def send_idle_message(chat_id: int) -> None:
    """Отправить случайную фразу в чат если долго не было активности."""
    # Не отправлять если последнее сообщение в чате уже от бота
    log = chat_message_log.get(chat_id, [])
    if log and log[-1] is True:
        logger.info(f"Idle-сообщение пропущено — последнее уже от бота, ждём ещё {IDLE_BASE}с")
        schedule_idle_message(chat_id)
        return

    phrase = get_random_phrase()
    logger.info(f"Idle-сообщение в чат {chat_id}: {phrase}")

    try:
        await bot.send_message(chat_id, phrase)
        # Не добавляем idle-фразы в историю — они не являются ответами на вопросы
        # и засоряют контекст модели, ухудшая качество следующих ответов.

        # Продлить онлайн-режим
        global online_mode_until
        online_mode_until = time.time() + ONLINE_WINDOW

        # Записать в лог сообщений
        if chat_id not in chat_message_log:
            chat_message_log[chat_id] = []
        chat_message_log[chat_id].append(True)
        if len(chat_message_log[chat_id]) > 5:
            chat_message_log[chat_id] = chat_message_log[chat_id][-5:]
    except Exception as e:
        logger.error(f"Ошибка при отправке idle-сообщения в чат {chat_id}: {e}")

    # Сбросить таймер — запланировать новое сообщение
    schedule_idle_message(chat_id)


def schedule_idle_message(chat_id: int) -> None:
    """Запланировать отправку случайной фразы через 1ч + случайное время."""
    # Отменить старый таймер если есть
    cancel_idle_timer(chat_id)

    delay = IDLE_BASE + random.uniform(0, IDLE_RANDOM_MAX)
    logger.info(f"Idle-таймер для чата {chat_id}: {delay:.0f}с")

    async def _idle_task() -> None:
        await asyncio.sleep(delay)
        await send_idle_message(chat_id)

    task = asyncio.create_task(_idle_task())
    idle_timers[chat_id] = task


def cancel_idle_timer(chat_id: int) -> None:
    """Отменить активный idle-таймер для чата."""
    if chat_id in idle_timers:
        idle_timers[chat_id].cancel()
        idle_timers.pop(chat_id, None)
        logger.info(f"Idle-таймер отменён для чата {chat_id}")


async def keep_typing(chat_id: int, duration: float, interval: float = 4.0):
    """Периодически отправляет typing action чтобы статус не пропал."""
    elapsed = 0.0
    while elapsed < duration:
        await bot.send_chat_action(chat_id, "typing")
        sleep_time = min(interval, duration - elapsed)
        await asyncio.sleep(sleep_time)
        elapsed += sleep_time


async def process_message(
    message: types.Message,
    user_text: str,
    trigger_type: TriggerType,
    counter_snapshot: int,
):
    chat_id = message.chat.id
    is_group = message.chat.type in ("group", "supergroup")

    # Зарегистрировать задачу в pending-списке
    task = asyncio.current_task()
    if chat_id not in pending_tasks:
        pending_tasks[chat_id] = []
    pending_tasks[chat_id].append(task)

    try:
        global online_mode_until

        # Для групп вычисляем задержку
        delay = 0.0
        if is_group:
            delay = calculate_group_delay(trigger_type)

        if delay > 0:
            logger.info(f"Задержка {delay:.1f}с для чата {chat_id} (триггер: {trigger_type})")
            await asyncio.sleep(delay)

        # Если к моменту выполнения онлайн-режим истёк — отменяем random_online
        if trigger_type == "random_online" and (
            not online_mode_until or time.time() >= online_mode_until
        ):
            logger.info(f"random_online отменён — онлайн-режим истёк для чата {chat_id}")
            return

        # Лесенка — только для followup в группах
        if is_group and trigger_type == "followup":
            ladder_bullets[chat_id] = [user_text]
            ladder_message[chat_id] = message
            ladder_counter[chat_id] = 1

        # Ждём лесенку: 10с после последнего сообщения (только для followup в группах)
        if is_group and trigger_type == "followup" and chat_id in ladder_bullets:
            ladder_user_id[chat_id] = message.from_user.id
            ladder_bullets[chat_id] = await ladder_wait(chat_id, list(ladder_bullets[chat_id]))
            # Сразу очищаем — ожидание закончилось
            ladder_user_id.pop(chat_id, None)

        # Склеить текст
        if is_group:
            final_text = "\n".join(ladder_bullets.get(chat_id, [user_text]))
            final_message = ladder_message.get(chat_id, message)

            # Лог лесенки
            ladder_cnt = ladder_counter.get(chat_id, 0)
            if ladder_cnt > 1:
                logger.info(f"Лесенка: {ladder_cnt} сообщений склеены")

            # Очистить буфер лесенки
            ladder_bullets.pop(chat_id, None)
            ladder_message.pop(chat_id, None)
            ladder_counter.pop(chat_id, None)
        else:
            final_text = user_text
            final_message = message

        # Задержка реакции — бот "видит" сообщение, но не сразу начинает печатать
        reaction_delay = random.uniform(2.5, 4.5)
        logger.info(f"Задержка реакции {reaction_delay:.1f}с для чата {chat_id}")
        await asyncio.sleep(reaction_delay)

        # Запрос к модели — typing идёт в фоне
        model_task = asyncio.create_task(
            asyncio.to_thread(query_lm_studio, chat_id, final_text)
        )
        typing_task = asyncio.create_task(keep_typing(chat_id, 300))
        reply = await model_task
        if not reply:
            logger.info(f"Пустой ответ для чата {chat_id} — пропускаю отправку")
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            return
        logger.info(f"Ответ модели для чата {chat_id}: {reply[:80]}...")

        # Задержка по символам — typing продолжается
        typing_time = estimate_typing_time(reply)
        logger.info(f"Типинг задержка {typing_time}с для чата {chat_id}")
        await asyncio.sleep(typing_time)

        # Остановить фоновый typing
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

        # Ещё раз показать "печатает" перед отправкой
        await bot.send_chat_action(chat_id, "typing")

        # Разбить на чанки
        chunks = []
        for i in range(0, len(reply), 4000):
            chunks.append(reply[i:i + 4000])

        # Отправить каждый чанк с проверкой reply vs plain
        for chunk in chunks:
            current_counter = get_message_counter(chat_id)
            has_new_messages = current_counter > counter_snapshot

            if is_group and has_new_messages:
                await bot.send_message(
                    chat_id, chunk,
                    reply_to_message_id=final_message.message_id,
                )
                logger.info(f"Отправлено как reply в чат {chat_id}")
            else:
                await bot.send_message(chat_id, chunk)
                logger.info(f"Отправлено как plain в чат {chat_id}")

        # Split-сообщение с шансом 30%
        if random.random() < SPLIT_CHANCE:
            second = await asyncio.to_thread(_query_split_reply, chat_id, reply)
            if second:
                await asyncio.sleep(random.uniform(1.0, 3.0))
                typing2 = asyncio.create_task(keep_typing(chat_id, 30))
                await asyncio.sleep(estimate_typing_time(second))
                typing2.cancel()
                try:
                    await typing2
                except asyncio.CancelledError:
                    pass
                await bot.send_message(chat_id, second)
                if is_group:
                    _append_group_ctx(chat_id, "лиса", second)
                logger.info(f"Split-reply в чат {chat_id}: {second[:60]}")

        # Продлить окно "онлайн" — бот активен ещё 20 минут (не для random_online)
        if trigger_type != "random_online":
            online_mode_until = time.time() + ONLINE_WINDOW
            logger.info(f"Окно онлайн продлено до {online_mode_until:.0f}")

        # Сбросить idle-таймер — бот только что отправил сообщение
        if is_group:
            schedule_idle_message(chat_id)
            _append_group_ctx(chat_id, "лиса", reply)

        # Записать ответ бота в лог
        if chat_id not in chat_message_log:
            chat_message_log[chat_id] = []
        chat_message_log[chat_id].append(True)
        if len(chat_message_log[chat_id]) > 5:
            chat_message_log[chat_id] = chat_message_log[chat_id][-5:]

        # Сбросить счётчик — бот только что ответил, следующее сообщение будет «первым»
        messages_since_reply[chat_id] = 0

        # Установить followup — следующий ответ того же пользователя будет триггером
        followup_user_id[chat_id] = message.from_user.id
        followup_expires[chat_id] = time.time() + 60

    except asyncio.CancelledError:
        logger.info(f"Задача обработки отменена для чата {chat_id}")
    finally:
        # Удалить задачу из pending-списка
        if chat_id in pending_tasks:
            if task in pending_tasks[chat_id]:
                pending_tasks[chat_id].remove(task)
            if not pending_tasks[chat_id]:
                del pending_tasks[chat_id]


@dp.message()
async def handle_message(message: types.Message):
    # Игнорировать анонимные посты и собственные сообщения бота
    if not message.from_user:
        return
    if BOT_ID and message.from_user.id == BOT_ID:
        return

    chat_id = message.chat.id
    increment_counter(chat_id)
    messages_since_reply[chat_id] = messages_since_reply.get(chat_id, 0) + 1

    # Записать сообщение пользователя в лог
    if chat_id not in chat_message_log:
        chat_message_log[chat_id] = []
    chat_message_log[chat_id].append(False)
    if len(chat_message_log[chat_id]) > 5:
        chat_message_log[chat_id] = chat_message_log[chat_id][-5:]

    logger.info(
        f"Сообщение #{get_message_counter(chat_id)} от {message.from_user.id} "
        f"в чате {chat_id} ({message.chat.type})"
    )

    # Записать в групповой буфер контекста
    if message.chat.type in ("group", "supergroup") and message.text:
        _append_group_ctx(chat_id, _get_sender_name(message), message.text)

    # Инициализировать idle-таймер при первом сообщении в чате (если ещё нет активного таймера)
    if message.chat.type in ("group", "supergroup") and chat_id not in idle_timers:
        schedule_idle_message(chat_id)
        logger.info(f"Idle-таймер инициализирован для чата {chat_id}")

    result = should_reply(message, chat_id)
    if result is None:
        # Случайный ответ во время онлайн-режима (только группы)
        if (
            message.chat.type in ("group", "supergroup")
            and message.text
            and online_mode_until
            and time.time() < online_mode_until
            and random.random() < online_chance(chat_id)
        ):
            logger.info(f"Случайный онлайн-триггер в чате {chat_id}")
            asyncio.create_task(
                process_message(message, message.text, "random_online", get_message_counter(chat_id))
            )
        return

    user_text, trigger_type = result
    logger.info(f"Триггер {trigger_type}: {user_text[:60]}")

    # Новый триггер — сбросить активную лесенку и followup
    ladder_bullets.pop(chat_id, None)
    ladder_user_id.pop(chat_id, None)
    ladder_message.pop(chat_id, None)
    ladder_counter.pop(chat_id, None)
    followup_user_id.pop(chat_id, None)
    followup_expires.pop(chat_id, None)

    # "шо?" если пустой текст (тег или слово без текста)
    if trigger_type in ("tag", "word") and not user_text:
        user_text = "шо?"

    # Отменить все pending-задачи для этого чата
    cancel_pending_tasks(chat_id)

    counter_snapshot = get_message_counter(chat_id)

    # Запуск обработки как фоновая задача
    asyncio.create_task(
        process_message(message, user_text, trigger_type, counter_snapshot)
    )


def resolve_model() -> None:
    """Если MODEL=auto — подтянуть первую загруженную модель из LM Studio."""
    global MODEL
    if MODEL and MODEL != "auto":
        return
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            logger.error("LM Studio не вернул ни одной модели — проверь что модель загружена")
            sys.exit(1)
        MODEL = models[0]["id"]
        logger.info(f"Автовыбор модели: {MODEL}")
    except Exception as e:
        logger.error(f"Не удалось получить модель из LM Studio: {e}")
        sys.exit(1)


async def main():
    # Определить модель (авто или из .env)
    resolve_model()

    # Загрузить RAG-индекс перед стартом
    load_rag_index()

    me = await bot.get_me()
    global BOT_ID, BOT_USERNAME
    BOT_ID = me.id
    BOT_USERNAME = me.username
    logger.info(f"Бот запущен: {me.username} (ID: {me.id})")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
