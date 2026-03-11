import json, collections
data = [json.loads(l) for l in open('/tmp/dpo_pairs_patch10a.jsonl')]
rl = collections.Counter(r['regret_label'] for r in data)
sym = collections.Counter(r['symbol'] for r in data)
rewards = [r['reward'] for r in data]
print('Total pairs  :', len(data))
print('Regret labels:', dict(rl))
print('Symbols      :', dict(sym))
print('Reward range : min', min(rewards), 'max', max(rewards))
print()
print('Sample pairs:')
for r in data[:3]:
    print(' ', r['symbol'], r['live_action'], '->', r['preferred_action'],
          'reward=', r['reward'], 'regret=', r['regret_label'])
