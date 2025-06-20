import asyncio
import telegram
import os
from dotenv import load_dotenv
import nest_asyncio
import asyncio

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def send_telegram_message(task: str, duration: str, result: str):
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token or chat id not set.")
        return
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    message = f"Task: {task}\nDuration: {duration}\nResult: {result}"
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        print(f"Failed to send telegram message: {e}")

def send_message_sync(task: str, duration: str, result: str):
    asyncio.run(send_telegram_message(task, duration, result))
