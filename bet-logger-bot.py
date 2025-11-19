import logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Update, ParseMode
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

# === CONFIG ===
TELEGRAM_BOT_TOKEN = '7928890551:AAGQP6krbyp4_jAedVZTIDXa_QLI2_ynvs4'
GOOGLE_SHEET_NAME = 'ConfirmedBets'
JSON_CREDENTIALS_FILE = 'google-credentials.json'


# === SETUP LOGGING ===
logging.basicConfig(
    filename="task_scheduler_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("✅ Script started successfully!")


# === GOOGLE SHEETS SETUP ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(JSON_CREDENTIALS_FILE, scope)
client = gspread.authorize(credentials)
sheet = client.open(GOOGLE_SHEET_NAME).sheet1

# === TELEGRAM SETUP ===
logging.basicConfig(level=logging.INFO)

def confirm_handler(update: Update, context: CallbackContext):
    try:
        # Check if the message is a reply to a bot alert
        if update.message.reply_to_message and update.message.text.strip() == "✅":
            original = update.message.reply_to_message.text
            lines = original.split('\n')

            sport = None
            home_team = None
            away_team = None
            market = None
            direction = None
            movement = None
            predicted = None
            game_time = None

            for line in lines:
                if "vs" in line:
                    away_team, home_team = line.split("vs")
                    away_team, home_team = away_team.strip(), home_team.strip()
                if "Market" in line or "Peak" in line:
                    market = line
                if "Movement" in line:
                    movement = line.split("Movement: ")[-1].strip()
                if "Predicted" in line:
                    predicted = line.split("Predicted:")[-1].strip()
                if "Game Time" in line:
                    game_time = line.split("Game Time:")[-1].strip()

            sport = "unknown"
            timestamp = update.message.date.strftime("%Y-%m-%d %H:%M:%S")

            row = [timestamp, sport, home_team, away_team, market, direction, movement, predicted, game_time]
            sheet.append_row(row)
            update.message.reply_text(f"📌 Bet on *{predicted}* confirmed and logged!", parse_mode=ParseMode.MARKDOWN)
        else:
            update.message.reply_text("❌ Please reply to an alert message using ✅ to confirm a bet.")
    except Exception as e:
        logging.error(f"❌ Error handling confirmation: {e}")
        update.message.reply_text("⚠️ Error logging your confirmation. Please try again.")

# === MAIN BOT RUNNER ===
def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.reply & Filters.text & Filters.regex(r'✅'), confirm_handler))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
