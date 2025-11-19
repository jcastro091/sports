from telegram.ext import Updater, MessageHandler, Filters

BOT_TOKEN = '7928890551:AAGQP6krbyp4_jAedVZTIDXa_QLI2_ynvs4'

def welcome(update, context):
    for user in update.message.new_chat_members:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"👋 Welcome {user.first_name}! You're now part of the SharpSignal beta.\n\nUse /join to get full access to picks, performance & more."
        )

updater = Updater(BOT_TOKEN, use_context=True)
dp = updater.dispatcher
dp.add_handler(MessageHandler(Filters.status_update.new_chat_members, welcome))
updater.start_polling()
