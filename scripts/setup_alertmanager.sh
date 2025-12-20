#!/bin/bash
# Quick Alertmanager Setup Script

echo "=== Alertmanager Setup ==="
echo ""
echo "Choose notification method:"
echo "1) Slack"
echo "2) Email (Gmail)"
echo "3) Skip for now"
read -p "Choice [1-3]: " choice

cd /home/qt/quantum_trader

case $choice in
  1)
    read -p "Slack Webhook URL: " webhook
    echo "SLACK_WEBHOOK_URL=$webhook" >> .env
    cp monitoring/alertmanager-slack.yml monitoring/alertmanager.yml
    docker compose -f docker-compose.alerting.yml restart alertmanager
    echo "✅ Slack notifications configured!"
    ;;
  2)
    read -p "Gmail address: " email
    read -p "Gmail app password: " password
    read -p "Alert email recipient: " alert_to
    cat >> .env << EOF
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=$email
SMTP_PASSWORD=$password
ALERT_EMAIL=$alert_to
EOF
    cp monitoring/alertmanager-email.yml monitoring/alertmanager.yml
    docker compose -f docker-compose.alerting.yml restart alertmanager
    echo "✅ Email notifications configured!"
    ;;
  3)
    echo "⏭️ Skipped"
    ;;
esac
