#!/usr/bin/env node
/**
 * Dashboard Auto-Repair CLI Tool
 * Kommando-linje verktøy for å teste og administrere dashboard auto-repair
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DASHBOARD_PATH = path.join(__dirname, '../pages/Dashboard.tsx');

// Simulated layout issues for testing
const layoutIssues = {
  'candles-in-header': {
    description: 'Market Candles component in header (should be in grid)',
    corruptedCode: '<h3>🕯️ Market Candles</h3><CandlesChart symbol="BTCUSDT" limit={50} />',
    fixedCode: `<CollapsiblePanel title="Market Candles" icon="🕯️" variant="default" defaultExpanded={true}>
              <CandlesChart symbol="BTCUSDT" limit={50} />
            </CollapsiblePanel>`
  },
  'narrow-trade-history': {
    description: 'Trade History not full width',
    corruptedCode: '<div className="lg:col-span-1 xl:col-span-2">',
    fixedCode: '<div className="sm:col-span-2 lg:col-span-4 xl:col-span-4">'
  },
  'missing-grid': {
    description: 'Missing grid container classes',
    corruptedCode: '<div className="space-y-6">',
    fixedCode: '<div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-6">'
  },
  'corrupted-imports': {
    description: 'Corrupted import statements',
    corruptedCode: 'import { useEffect, useState }          <div className="min-h-full">',
    fixedCode: 'import { useEffect, useState } from \'react\';'
  }
};

function showHelp() {
  console.log(`
🤖 Dashboard Auto-Repair CLI Tool

Usage: node auto-repair-cli.js [command] [options]

Commands:
  check           Check dashboard health
  repair          Run automatic repair
  corrupt [type]  Simulate corruption for testing (type: ${Object.keys(layoutIssues).join(', ')})
  reset          Reset to optimal layout
  status         Show current auto-repair status

Options:
  --help, -h     Show this help message
  --verbose, -v  Show detailed output

Examples:
  node auto-repair-cli.js check
  node auto-repair-cli.js corrupt candles-in-header
  node auto-repair-cli.js repair
  node auto-repair-cli.js reset

Dette verktøyet lar deg teste auto-repair funksjonaliteten uten å måtte gjøre det manuelt!
`);
}

function checkHealth() {
  console.log('🔍 Checking dashboard health...');
  
  if (!fs.existsSync(DASHBOARD_PATH)) {
    console.log('❌ Dashboard.tsx file not found!');
    return false;
  }

  const content = fs.readFileSync(DASHBOARD_PATH, 'utf8');
  const issues = [];

  // Check for common issues
  Object.entries(layoutIssues).forEach(([issueType, issue]) => {
    if (content.includes(issue.corruptedCode)) {
      issues.push(`❌ ${issue.description}`);
    }
  });

  // Check for proper grid structure
  if (!content.includes('grid-cols-1 sm:grid-cols-2')) {
    issues.push('❌ Missing responsive grid classes');
  }

  // Check for CollapsiblePanel usage
  const panelCount = (content.match(/CollapsiblePanel/g) || []).length;
  if (panelCount < 3) {
    issues.push('❌ Insufficient CollapsiblePanel usage');
  }

  // Check for full-width Trade History
  if (!content.includes('xl:col-span-4') || !content.includes('Trade History')) {
    issues.push('❌ Trade History not full width');
  }

  if (issues.length === 0) {
    console.log('✅ Dashboard health: HEALTHY');
    console.log('📊 All layout components properly positioned');
    console.log('🎯 Responsive grid structure intact');
    console.log('📱 CollapsiblePanels functioning correctly');
    return true;
  } else {
    console.log('🚨 Dashboard health: ISSUES DETECTED');
    issues.forEach(issue => console.log(issue));
    console.log(`\n💡 Found ${issues.length} issues that need attention`);
    return false;
  }
}

function simulateCorruption(corruptionType) {
  console.log(`🧪 Simulating corruption: ${corruptionType}...`);
  
  const issue = layoutIssues[corruptionType];
  if (!issue) {
    console.log(`❌ Unknown corruption type: ${corruptionType}`);
    console.log(`Available types: ${Object.keys(layoutIssues).join(', ')}`);
    return false;
  }

  if (!fs.existsSync(DASHBOARD_PATH)) {
    console.log('❌ Dashboard.tsx file not found!');
    return false;
  }

  let content = fs.readFileSync(DASHBOARD_PATH, 'utf8');
  
  // Apply corruption
  if (content.includes(issue.fixedCode)) {
    content = content.replace(issue.fixedCode, issue.corruptedCode);
    fs.writeFileSync(DASHBOARD_PATH, content);
    console.log(`✅ Corruption applied: ${issue.description}`);
    console.log('🔧 You can now test the auto-repair functionality!');
    return true;
  } else {
    console.log(`⚠️  Could not apply corruption - pattern not found`);
    return false;
  }
}

function runAutoRepair() {
  console.log('🚀 Running automatic repair...');
  
  if (!fs.existsSync(DASHBOARD_PATH)) {
    console.log('❌ Dashboard.tsx file not found!');
    return false;
  }

  let content = fs.readFileSync(DASHBOARD_PATH, 'utf8');
  let repairsApplied = 0;

  // Apply all known fixes
  Object.entries(layoutIssues).forEach(([issueType, issue]) => {
    if (content.includes(issue.corruptedCode)) {
      content = content.replace(issue.corruptedCode, issue.fixedCode);
      repairsApplied++;
      console.log(`✅ Fixed: ${issue.description}`);
    }
  });

  if (repairsApplied > 0) {
    fs.writeFileSync(DASHBOARD_PATH, content);
    console.log(`🎉 Auto-repair completed! Applied ${repairsApplied} fixes.`);
    console.log('🔄 Please refresh your browser to see the changes.');
  } else {
    console.log('✅ No issues found that could be automatically repaired.');
  }

  return repairsApplied > 0;
}

function resetToOptimal() {
  console.log('🎯 Resetting dashboard to optimal layout...');
  
  // This would trigger a complete layout reset
  // For now, just run repair to fix known issues
  const repaired = runAutoRepair();
  
  if (repaired) {
    console.log('🎉 Dashboard reset to optimal layout completed!');
  } else {
    console.log('✅ Dashboard already in optimal state.');
  }
}

function showStatus() {
  console.log('📊 Auto-Repair System Status');
  console.log('============================');
  
  const isHealthy = checkHealth();
  
  console.log('\n🔧 Available Repairs:');
  Object.entries(layoutIssues).forEach(([type, issue]) => {
    console.log(`  • ${type}: ${issue.description}`);
  });
  
  console.log('\n💡 Quick Commands:');
  console.log('  • Check health:    node auto-repair-cli.js check');
  console.log('  • Run repair:      node auto-repair-cli.js repair'); 
  console.log('  • Reset layout:    node auto-repair-cli.js reset');
  console.log('  • Test corruption: node auto-repair-cli.js corrupt candles-in-header');
}

// Main CLI logic
const args = process.argv.slice(2);
const command = args[0];
const options = args.slice(1);

console.log('🤖 Dashboard Auto-Repair CLI Tool');
console.log('==================================');

switch (command) {
  case 'check':
    checkHealth();
    break;
    
  case 'repair':
    runAutoRepair();
    break;
    
  case 'corrupt':
    const corruptionType = options[0];
    if (!corruptionType) {
      console.log('❌ Please specify corruption type');
      console.log(`Available: ${Object.keys(layoutIssues).join(', ')}`);
    } else {
      simulateCorruption(corruptionType);
    }
    break;
    
  case 'reset':
    resetToOptimal();
    break;
    
  case 'status':
    showStatus();
    break;
    
  case 'help':
  case '--help':
  case '-h':
  default:
    showHelp();
    break;
}