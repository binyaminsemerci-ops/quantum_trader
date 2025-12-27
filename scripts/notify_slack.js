#!/usr/bin/env node

/**
 * QuantumFond CI/CD Slack Notifier
 * Sends formatted alerts to Slack channel when tests pass/fail
 */

import fetch from 'node-fetch';

const status = process.argv[2] || 'unknown';
const summary = process.argv[3] || 'No summary provided';
const jobName = process.argv[4] || 'CI Job';

// Color and emoji mapping
const statusConfig = {
  success: {
    color: '#22c55e',
    emoji: '✅',
    title: 'Build Successful'
  },
  failure: {
    color: '#ef4444',
    emoji: '🚨',
    title: 'Build Failed'
  },
  cancelled: {
    color: '#f59e0b',
    emoji: '⚠️',
    title: 'Build Cancelled'
  },
  unknown: {
    color: '#6b7280',
    emoji: '❓',
    title: 'Build Status Unknown'
  }
};

const config = statusConfig[status] || statusConfig.unknown;

// Build rich Slack message payload
const payload = {
  attachments: [
    {
      color: config.color,
      title: `${config.emoji} QuantumFond CI - ${config.title}`,
      text: summary,
      fields: [
        {
          title: 'Job',
          value: jobName,
          short: true
        },
        {
          title: 'Status',
          value: status.toUpperCase(),
          short: true
        },
        {
          title: 'Repository',
          value: process.env.GITHUB_REPOSITORY || 'quantum_trader',
          short: true
        },
        {
          title: 'Branch',
          value: process.env.GITHUB_REF_NAME || 'main',
          short: true
        },
        {
          title: 'Commit',
          value: `\`${(process.env.GITHUB_SHA || 'unknown').substring(0, 7)}\``,
          short: true
        },
        {
          title: 'Workflow',
          value: process.env.GITHUB_WORKFLOW || 'CI Pipeline',
          short: true
        }
      ],
      footer: 'QuantumFond CI Reporter',
      footer_icon: 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png',
      ts: Math.floor(Date.now() / 1000),
      actions: [
        {
          type: 'button',
          text: 'View Logs',
          url: `https://github.com/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`
        }
      ]
    }
  ]
};

// Validate webhook URL
const webhookUrl = process.env.SLACK_WEBHOOK_URL;
if (!webhookUrl) {
  console.error('❌ SLACK_WEBHOOK_URL environment variable not set');
  process.exit(1);
}

// Send notification to Slack
fetch(webhookUrl, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
  .then(response => {
    if (!response.ok) {
      throw new Error(`Slack API error: ${response.status} ${response.statusText}`);
    }
    console.log(`✅ Slack notification sent: ${config.emoji} ${jobName} - ${status}`);
  })
  .catch(err => {
    console.error('❌ Slack webhook failed:', err.message);
    process.exit(1);
  });
