#!/bin/bash
# Quantum Trader - Interactive Alertmanager Configuration Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$HOME/.env"

echo "================================================"
echo "   Quantum Trader Alertmanager Configuration"
echo "================================================"
echo ""

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Function to add or update .env variable
update_env() {
    local key=$1
    local value=$2
    
    if grep -q "^${key}=" "$ENV_FILE"; then
        # Update existing
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        echo "✅ Updated $key"
    else
        # Add new
        echo "${key}=${value}" >> "$ENV_FILE"
        echo "✅ Added $key"
    fi
}

echo "Choose notification method:"
echo "1) Slack (Recommended)"
echo "2) Email (Gmail/SMTP)"
echo "3) Both"
echo "4) Skip configuration"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1|3)
        echo ""
        echo "=== Slack Configuration ==="
        echo ""
        echo "📝 Steps to get Slack Webhook URL:"
        echo "   1. Go to https://api.slack.com/apps"
        echo "   2. Create new app → 'Incoming Webhooks'"
        echo "   3. Activate webhooks → Add New Webhook"
        echo "   4. Select channel (e.g., #quantum-trader-alerts)"
        echo "   5. Copy webhook URL"
        echo ""
        read -p "Enter Slack Webhook URL: " slack_webhook
        
        if [ -n "$slack_webhook" ]; then
            update_env "SLACK_WEBHOOK_URL" "$slack_webhook"
            
            # Copy Slack config
            cp "${PROJECT_ROOT}/monitoring/alertmanager-slack.yml" \
               "${PROJECT_ROOT}/monitoring/alertmanager.yml"
            echo "✅ Slack configuration copied"
        else
            echo "⚠️  Slack webhook empty, skipping..."
        fi
        ;;
esac

case $choice in
    2|3)
        echo ""
        echo "=== Email Configuration ==="
        echo ""
        echo "Common SMTP settings:"
        echo "  Gmail:     smtp.gmail.com:587"
        echo "  Outlook:   smtp-mail.outlook.com:587"
        echo "  SendGrid:  smtp.sendgrid.net:587"
        echo ""
        
        read -p "SMTP Host (e.g., smtp.gmail.com): " smtp_host
        read -p "SMTP Port (default 587): " smtp_port
        smtp_port=${smtp_port:-587}
        
        read -p "SMTP Username (your email): " smtp_user
        read -sp "SMTP Password (app password): " smtp_pass
        echo ""
        
        read -p "Alert recipient email: " alert_email
        
        if [ -n "$smtp_host" ] && [ -n "$smtp_user" ] && [ -n "$smtp_pass" ] && [ -n "$alert_email" ]; then
            update_env "SMTP_HOST" "$smtp_host"
            update_env "SMTP_PORT" "$smtp_port"
            update_env "SMTP_USERNAME" "$smtp_user"
            update_env "SMTP_PASSWORD" "$smtp_pass"
            update_env "ALERT_EMAIL_TO" "$alert_email"
            
            # If Slack not configured, use email config
            if [ "$choice" == "2" ]; then
                cp "${PROJECT_ROOT}/monitoring/alertmanager-email.yml" \
                   "${PROJECT_ROOT}/monitoring/alertmanager.yml"
                echo "✅ Email configuration copied"
            else
                echo "✅ Email settings added (will use Slack config)"
                echo "   To use both, merge configs manually"
            fi
        else
            echo "⚠️  Incomplete email settings, skipping..."
        fi
        ;;
esac

if [ "$choice" == "4" ]; then
    echo "Configuration skipped"
    exit 0
fi

# Restart Alertmanager
echo ""
echo "=== Restarting Alertmanager ==="
cd "$PROJECT_ROOT"

if [ -f "docker-compose.alerting.yml" ]; then
    docker compose -f docker-compose.alerting.yml restart alertmanager
    echo "✅ Alertmanager restarted"
else
    echo "⚠️  docker-compose.alerting.yml not found, skipping restart"
fi

echo ""
echo "================================================"
echo "   ✅ Configuration Complete!"
echo "================================================"
echo ""
echo "Test your alerts:"
echo "  docker stop quantum_ai_engine"
echo "  sleep 35"
echo "  docker start quantum_ai_engine"
echo ""
echo "View alerts:"
echo "  curl http://localhost:9093/api/v2/alerts"
echo ""
