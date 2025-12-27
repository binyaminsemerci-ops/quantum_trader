# 🔐 GitHub Secrets Setup - Phase 23.1

This file contains step-by-step instructions for setting up the required GitHub Secrets for Slack and Telegram notifications.

---

## ⚠️ Important Notice

**DO NOT** add your actual webhook URLs or tokens to this file or commit them to Git!  
This guide shows you how to add them as GitHub Secrets instead.

---

## 1️⃣ Slack Webhook Setup (Required)

### Step 1: Create Slack Incoming Webhook

1. Go to: https://api.slack.com/apps
2. Click **"Create New App"** → **"From scratch"**
3. **App Name:** `QuantumFond CI Reporter`
4. **Workspace:** Select your Slack workspace
5. Click **"Create App"**

### Step 2: Enable Incoming Webhooks

1. In the left sidebar, click **"Incoming Webhooks"**
2. Toggle **"Activate Incoming Webhooks"** to **ON**
3. Scroll down and click **"Add New Webhook to Workspace"**
4. **Select a channel:** 
   - Choose `#quantumfond-ci` (create channel first if it doesn't exist)
   - Or select any other channel you want alerts sent to
5. Click **"Allow"**

### Step 3: Copy Webhook URL

You'll see a webhook URL that looks like:
```
https://hooks.slack.com/services/YOUR_WORKSPACE_ID/YOUR_CHANNEL_ID/YOUR_SECRET_TOKEN
```

**⚠️ Copy this URL - you'll need it in the next step!**

### Step 4: Add to GitHub Secrets

1. Go to your GitHub repository: https://github.com/binyaminsemerci-ops/quantum_trader
2. Click **Settings** (top navigation)
3. In the left sidebar, expand **Secrets and variables** → Click **Actions**
4. Click **"New repository secret"** (green button)
5. Fill in:
   - **Name:** `SLACK_WEBHOOK_URL`
   - **Secret:** Paste the webhook URL from Step 3
6. Click **"Add secret"**

✅ **Slack setup complete!**

---

## 2️⃣ Telegram Bot Setup (Optional)

### Step 1: Create Telegram Bot

1. Open Telegram on your phone/desktop
2. Search for: `@BotFather`
3. Start a chat and send: `/newbot`
4. Follow the prompts:
   - **Bot name:** `QuantumFond CI Reporter`
   - **Bot username:** `quantumfond_ci_bot` (must end with `_bot`)
5. You'll receive a message with your **Bot Token**:
   ```
   Use this token to access the HTTP API:
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

**⚠️ Copy this token - you'll need it!**

### Step 2: Get Chat ID

#### Option A: Create a Channel
1. In Telegram, create a new channel: `QuantumFond CI Alerts`
2. Add your bot as an administrator to the channel
3. Send a test message to the channel
4. Visit this URL in your browser (replace YOUR_BOT_TOKEN):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
5. Look for `"chat":{"id":-100XXXXXXXXX}`
6. Copy the Chat ID (include the minus sign!)

#### Option B: Use a Group
1. Create a Telegram group
2. Add your bot to the group
3. Send a test message
4. Visit the getUpdates URL (same as above)
5. Copy the Chat ID

### Step 3: Add to GitHub Secrets

1. Go to: https://github.com/binyaminsemerci-ops/quantum_trader/settings/secrets/actions
2. Click **"New repository secret"**
3. Add **first secret:**
   - **Name:** `TELEGRAM_BOT_TOKEN`
   - **Secret:** Paste the bot token from Step 1
4. Click **"Add secret"**
5. Click **"New repository secret"** again
6. Add **second secret:**
   - **Name:** `TELEGRAM_CHAT_ID`
   - **Secret:** Paste the chat ID from Step 2
7. Click **"Add secret"**

✅ **Telegram setup complete!**

---

## 🧪 Testing the Setup

### Test Locally (Before Push)

#### Test Slack:
```powershell
cd C:\quantum_trader

# Set environment variable (replace with your actual webhook)
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Test success notification
node scripts/notify_slack.js success "Local test successful" "Manual Test"

# You should see: ✅ Slack notification sent: ✅ Manual Test - success
# Check your Slack channel for the message!
```

#### Test Telegram:
```powershell
# Set environment variables (replace with your actual values)
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# Test success notification
node scripts/notify_telegram.js success "Local test successful" "Manual Test"

# You should see: ✅ Telegram notification sent: ✅ Manual Test - success
# Check your Telegram channel for the message!
```

### Test in GitHub Actions (After Push)

1. Push your changes to main branch:
   ```bash
   git push origin main
   ```

2. Go to: https://github.com/binyaminsemerci-ops/quantum_trader/actions

3. Click on the latest workflow run

4. When tests complete, you should see notifications in:
   - **Slack:** `#quantumfond-ci` channel
   - **Telegram:** Your configured channel/group

---

## 🔍 Verify Secrets Are Set

```bash
# Using GitHub CLI
gh secret list

# Expected output:
# SLACK_WEBHOOK_URL       Updated 2024-12-27
# TELEGRAM_BOT_TOKEN      Updated 2024-12-27  (if configured)
# TELEGRAM_CHAT_ID        Updated 2024-12-27  (if configured)
```

Or check in browser:
https://github.com/binyaminsemerci-ops/quantum_trader/settings/secrets/actions

You should see your secrets listed (values are hidden for security).

---

## 🐛 Troubleshooting

### Slack Webhook Not Working

**Error:** `404 Not Found`  
**Solution:** Webhook URL is incorrect. Go back to Slack app settings and regenerate webhook.

**Error:** `403 Forbidden`  
**Solution:** Webhook was revoked. Create a new webhook in Slack.

**Error:** `500 Internal Server Error`  
**Solution:** Slack API issue. Wait a few minutes and try again.

### Telegram Not Working

**Error:** `401 Unauthorized`  
**Solution:** Bot token is incorrect. Create new bot with @BotFather.

**Error:** `400 Bad Request: chat not found`  
**Solution:** Chat ID is wrong or bot is not added to channel/group.

**Error:** `400 Bad Request: bot was blocked by the user`  
**Solution:** Unblock the bot in Telegram settings.

### GitHub Actions Not Running Notifications

**Issue:** Secrets not available in workflow  
**Solution:** 
1. Verify secrets are added at repository level (not environment level)
2. Check secret names match exactly: `SLACK_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
3. Secrets are case-sensitive!

**Issue:** node-fetch not found  
**Solution:**
```bash
cd C:\quantum_trader
npm install
git add package.json package-lock.json
git commit -m "Add node-fetch dependency"
git push
```

---

## 🔐 Security Best Practices

1. ✅ **Never commit secrets to Git**
2. ✅ **Use GitHub Secrets for sensitive data**
3. ✅ **Rotate webhook URLs periodically**
4. ✅ **Limit Slack webhook to specific channel**
5. ✅ **Use read-only channel permissions**
6. ✅ **Review bot permissions regularly**

---

## ✅ Setup Complete Checklist

- [ ] Slack app created
- [ ] Slack incoming webhook enabled
- [ ] Slack channel created (`#quantumfond-ci`)
- [ ] `SLACK_WEBHOOK_URL` secret added to GitHub
- [ ] Local Slack test successful (green message in channel)
- [ ] (Optional) Telegram bot created with @BotFather
- [ ] (Optional) Telegram channel/group created
- [ ] (Optional) Bot added to Telegram channel
- [ ] (Optional) `TELEGRAM_BOT_TOKEN` secret added
- [ ] (Optional) `TELEGRAM_CHAT_ID` secret added
- [ ] (Optional) Local Telegram test successful
- [ ] `node-fetch` dependency installed
- [ ] Changes committed and pushed to GitHub
- [ ] GitHub Actions workflow ran successfully
- [ ] Notifications received in Slack (and Telegram if configured)

---

**Next Steps:**

Once all checkmarks are complete:
1. Push a test commit to trigger the workflow
2. Watch for notifications in your Slack/Telegram
3. Verify green (✅) alerts on success, red (🚨) on failure

---

**Support:** If you encounter issues, refer to PHASE23_1_SLACK_TELEGRAM_ALERTS.md for detailed troubleshooting.

**Created:** December 27, 2024  
**Phase:** 23.1 - CI/CD Alerting System
