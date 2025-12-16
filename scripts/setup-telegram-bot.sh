#!/bin/bash
# Setup Telegram Bot for Alerting

echo "=== Telegram Bot Setup ==="
echo ""
echo "1. Open Telegram and search for @BotFather"
echo "2. Send: /newbot"
echo "3. Follow instructions to create bot"
echo "4. Copy the BOT_TOKEN"
echo ""
echo "5. Start chat with your new bot"
echo "6. Send any message to the bot"
echo "7. Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates"
echo "8. Find your CHAT_ID in the response"
echo ""
echo "9. Add to .env file:"
echo "   TELEGRAM_BOT_TOKEN=your_bot_token_here"
echo "   TELEGRAM_CHAT_ID=your_chat_id_here"
echo ""
read -p "Enter BOT_TOKEN: " BOT_TOKEN
read -p "Enter CHAT_ID: " CHAT_ID

echo ""
echo "Testing bot..."
curl -s "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage?chat_id=${CHAT_ID}&text=✅%20Quantum%20Trader%20alerting%20configured!"

echo ""
echo "Add these to your .env file:"
echo "TELEGRAM_BOT_TOKEN=${BOT_TOKEN}"
echo "TELEGRAM_CHAT_ID=${CHAT_ID}"
