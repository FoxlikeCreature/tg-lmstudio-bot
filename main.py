import os
import sys
import time
import random
import asyncio
import requests
import logging
from typing import Literal
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from phrases import get_random_phrase

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")
MODEL = os.getenv("MODEL")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты — полезный ассистент.")
TRIGGER_WORD = "лиса"
TRIGGER_TAG = "@Neurocutefox_bot"

IMMEDIATE_CHANCE = 0.2  # 20% шанс ответить сразу в группе
MIN_DELAY = 180         # 3 минуты
MAX_DELAY = 600         # 10 минут
ONLINE_WINDOW = 1200    # 20 минут — окно "онлайн" после ответа

IDLE_BASE = 3600        # 1 час — базовое время без активности
IDLE_RANDOM_MAX = 600   # 10 минут — случайная добавка

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

# Idle-таймер: автоматические сообщения если никто не триггерил
idle_timers: dict[int, asyncio.Task] = {}
last_trigger_time: dict[int, float] = {}


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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        return f"Ошибка при запросе к модели: {e}"


TriggerType = Literal["tag", "question", "reply", "word", "followup"]


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


def calculate_group_delay(trigger_type: TriggerType) -> float:
    if trigger_type in ("tag", "question", "followup"):
        return 0.0
    # Если бот "онлайн" — отвечаем сразу
    if online_mode_until and time.time() < online_mode_until:
        return 0.0
    if random.random() < IMMEDIATE_CHANCE:
        return 0.0
    return random.uniform(MIN_DELAY, MAX_DELAY)


async def send_idle_message(chat_id: int) -> None:
    """Отправить случайную фразу в чат если долго не было активности."""
    phrase = get_random_phrase()
    logger.info(f"Idle-сообщение в чат {chat_id}: {phrase}")

    try:
        await bot.send_message(chat_id, phrase)
        # Записать в историю чтобы модель понимала контекст
        history = get_chat_history(chat_id)
        history.append({"role": "assistant", "content": phrase})
        if len(history) > MAX_HISTORY * 2:
            history[:] = history[-MAX_HISTORY * 2:]

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

    # Для групп вычисляем задержку
    delay = 0.0
    if is_group:
        delay = calculate_group_delay(trigger_type)

    if delay > 0:
        logger.info(f"Задержка {delay:.1f}с для чата {chat_id} (триггер: {trigger_type})")
        await asyncio.sleep(delay)

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

    # Продлить окно "онлайн" — бот активен ещё 20 минут
    global online_mode_until
    online_mode_until = time.time() + ONLINE_WINDOW
    logger.info(f"Окно онлайн продлено до {online_mode_until:.0f}")

    # Записать ответ бота в лог
    if chat_id not in chat_message_log:
        chat_message_log[chat_id] = []
    chat_message_log[chat_id].append(True)
    if len(chat_message_log[chat_id]) > 5:
        chat_message_log[chat_id] = chat_message_log[chat_id][-5:]

    # Установить followup — следующий ответ того же пользователя будет триггером
    followup_user_id[chat_id] = message.from_user.id
    followup_expires[chat_id] = time.time() + 60


@dp.message()
async def handle_message(message: types.Message):
    chat_id = message.chat.id
    increment_counter(chat_id)

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

    # Для групповых чатов — обновлять idle-таймер при любом сообщении
    if message.chat.type in ("group", "supergroup"):
        last_trigger_time[chat_id] = time.time()
        schedule_idle_message(chat_id)

    result = should_reply(message, chat_id)
    if result is None:
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

    # "шо?" если пустой текст (тег без текста)
    if trigger_type == "tag" and not user_text:
        user_text = "шо?"

    counter_snapshot = get_message_counter(chat_id)

    # Запуск обработки как фоновая задача
    asyncio.create_task(
        process_message(message, user_text, trigger_type, counter_snapshot)
    )


async def main():
    me = await bot.get_me()
    global BOT_ID, BOT_USERNAME
    BOT_ID = me.id
    BOT_USERNAME = me.username
    logger.info(f"Бот запущен: {me.username} (ID: {me.id})")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
