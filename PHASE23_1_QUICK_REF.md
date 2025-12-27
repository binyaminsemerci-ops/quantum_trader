# 🚨 Phase 23.1 Quick Reference - CI/CD Alerts

## 🔧 Setup Commands

### Slack Webhook Configuration
```bash
# Add to GitHub Secrets
Name: SLACK_WEBHOOK_URL
Value: <your-slack-webhook-url-from-slack-app-settings>
```

### Telegram Bot Configuration (Optional)
```bash
# Add to GitHub Secrets
Name: TELEGRAM_BOT_TOKEN
Value: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

Name: TELEGRAM_CHAT_ID
Value: -100XXXXXXXXX
```

---

## 🧪 Local Testing

### Test Slack Notification
```powershell
cd C:\quantum_trader
$env:SLACK_WEBHOOK_URL = "YOUR_WEBHOOK_URL"
node scripts/notify_slack.js success "Test message" "Manual Test"
```

### Test Telegram Notification
```powershell
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
node scripts/notify_telegram.js failure "Test alert" "Manual Test"
```

---

## 📋 Notification Script Usage

### notify_slack.js
```bash
node scripts/notify_slack.js <status> <message> <job_name>

# Status: success | failure | cancelled | unknown
# Message: Text description
# Job Name: Name of CI job

# Examples:
node scripts/notify_slack.js success "All tests passed" "Unit Tests"
node scripts/notify_slack.js failure "3 tests failed" "Integration Tests"
```

### notify_telegram.js
```bash
node scripts/notify_telegram.js <status> <message> <job_name>

# Same parameters as Slack
```

---

## 🎨 Notification Colors

| Status | Color | Emoji | Use Case |
|--------|-------|-------|----------|
| success | Green (#22c55e) | ✅ | Tests passed |
| failure | Red (#ef4444) | 🚨 | Tests failed |
| cancelled | Orange (#f59e0b) | ⚠️ | Pipeline stopped |
| unknown | Gray (#6b7280) | ❓ | Unknown state |

---

## 📊 Workflow Integration Points

### 1. Unit Tests (Vitest)
- **Trigger:** Test failure
- **Alert:** "Vitest unit tests failed - check logs for details"
- **Job Name:** Unit Tests

### 2. Integration Tests (Cypress)
- **Trigger:** E2E test failure
- **Alert:** "Cypress E2E tests failed - check screenshots and videos"
- **Job Name:** Integration Tests

### 3. Lint & Format Checks
- **Trigger:** ESLint failure
- **Alert:** "ESLint checks failed - code quality issues detected"
- **Job Name:** Lint & Format Check

### 4. Pipeline Summary
- **Trigger:** All jobs complete
- **Success Alert:** "All tests passed! Unit tests ✅ | Cypress E2E ✅ | Lint ✅"
- **Failure Alert:** "Pipeline failed! Unit: [status] | Integration: [status] | Lint: [status]"
- **Job Name:** Pipeline Complete / Pipeline Failed

---

## 🔗 Important Links

- **Slack Webhook Setup:** https://api.slack.com/messaging/webhooks
- **Telegram BotFather:** https://t.me/BotFather
- **GitHub Actions Logs:** https://github.com/binyaminsemerci-ops/quantum_trader/actions
- **Full Documentation:** PHASE23_1_SLACK_TELEGRAM_ALERTS.md

---

## 🐛 Quick Troubleshooting

### Slack Not Working
```bash
# Check if secret is set
gh secret list | grep SLACK_WEBHOOK_URL

# Test webhook manually
curl -X POST YOUR_WEBHOOK_URL -H 'Content-Type: application/json' \
  -d '{"text":"Test message"}'
```

### Telegram Not Working
```bash
# Verify bot token
curl "https://api.telegram.org/botYOUR_BOT_TOKEN/getMe"

# Verify chat ID
curl "https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates"
```

### Notifications Not Sent
- Ensure `node-fetch` is installed: `npm install node-fetch`
- Check GitHub Actions logs for error messages
- Verify secrets are set in repository settings

---

## 📦 Dependencies

```json
{
  "node-fetch": "^2.6.1"
}
```

Add to `package.json` if missing:
```bash
npm install --save node-fetch@2
```

---

## 🚀 Quick Deploy

```powershell
# Stage changes
git add scripts/notify_slack.js
git add scripts/notify_telegram.js
git add .github/workflows/test.yml
git add PHASE23_1_*.md

# Commit
git commit -m "feat: Phase 23.1 - Slack/Telegram CI/CD alerts

- Add Slack notification script
- Add Telegram notification script
- Integrate alerts into GitHub Actions workflow
- Real-time failure and success notifications"

# Push
git push origin main
```

---

## ✅ Verification Checklist

- [ ] Slack webhook created
- [ ] `SLACK_WEBHOOK_URL` secret added
- [ ] Local Slack test successful
- [ ] (Optional) Telegram bot created
- [ ] (Optional) Telegram secrets added
- [ ] (Optional) Local Telegram test successful
- [ ] Changes committed and pushed
- [ ] First GitHub Actions run shows alerts

---

## 🎯 Success Criteria

- ✅ Slack alerts arrive within 5 seconds
- ✅ Alerts are color-coded (green/red)
- ✅ Include commit hash and direct log links
- ✅ No false positives
- ✅ Mobile notifications work (via Slack/Telegram app)

---

**Created:** December 27, 2024  
**Phase:** 23.1 - CI/CD Alerting System  
**Status:** Production Ready ✅
