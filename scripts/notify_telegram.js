#!/usr/bin/env node

/**
 * QuantumFond CI/CD Telegram Notifier
 * Sends formatted alerts to Telegram channel when tests pass/fail
 */

import fetch from 'node-fetch';

const status = process.argv[2] || 'unknown';
const summary = process.argv[3] || 'No summary provided';
const jobName = process.argv[4] || 'CI Job';

// Status emoji mapping
const statusEmoji = {
  success: '✅',
  failure: '🚨',
  cancelled: '⚠️',
  unknown: '❓'
};

const emoji = statusEmoji[status] || statusEmoji.unknown;

// Validate environment variables
const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const CHAT_ID = process.env.TELEGRAM_CHAT_ID;

if (!BOT_TOKEN || !CHAT_ID) {
  console.error('❌ Missing Telegram configuration:');
  if (!BOT_TOKEN) console.error('  - TELEGRAM_BOT_TOKEN not set');
  if (!CHAT_ID) console.error('  - TELEGRAM_CHAT_ID not set');
  process.exit(1);
}

// Build formatted Telegram message
const message = `${emoji} *QuantumFond CI - ${status.toUpperCase()}*

📋 *Job:* ${jobName}
📦 *Repository:* ${process.env.GITHUB_REPOSITORY || 'quantum_trader'}
🌿 *Branch:* ${process.env.GITHUB_REF_NAME || 'main'}
📝 *Commit:* \`${(process.env.GITHUB_SHA || 'unknown').substring(0, 7)}\`
⚙️ *Workflow:* ${process.env.GITHUB_WORKFLOW || 'CI Pipeline'}

💬 *Summary:* ${summary}

🔗 [View Logs](https://github.com/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID})`;

// Send notification to Telegram
const apiUrl = `https://api.telegram.org/bot${BOT_TOKEN}/sendMessage`;

fetch(apiUrl, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    chat_id: CHAT_ID,
    text: message,
    parse_mode: 'Markdown',
    disable_web_page_preview: true
  })
})
  .then(response => response.json())
  .then(data => {
    if (!data.ok) {
      throw new Error(`Telegram API error: ${data.description || 'Unknown error'}`);
    }
    console.log(`✅ Telegram notification sent: ${emoji} ${jobName} - ${status}`);
  })
  .catch(err => {
    console.error('❌ Telegram notification failed:', err.message);
    process.exit(1);
  });
