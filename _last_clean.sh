#!/bin/bash
echo "Remaining pos:"
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null
echo "Remaining cooldown:"
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null
# Delete them
redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | while read k; do redis-cli DEL "$k" 2>/dev/null; done
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do redis-cli DEL "$k" 2>/dev/null; done
echo "---"
echo "pos=$(redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | wc -l)"
echo "cooldowns=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | wc -l)"
