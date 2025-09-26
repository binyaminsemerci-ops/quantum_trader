"""Simple GitHub Actions monitor for a PR branch.

Usage:
  set GITHUB_TOKEN=ghp_...   # or use MY_PAT
  python scripts/monitor_pr_ci.py --repo binyaminsemerci-ops/quantum_trader --branch chore/mypy-fixes

This script requires `requests` and a token with `repo` and `actions:read` scopes.
"""
import os
import time
import argparse
import requests


def list_runs(repo: str, branch: str, token: str):
    url = f'https://api.github.com/repos/{repo}/actions/runs'
    params = {'branch': branch, 'per_page': 10}
    headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--interval', type=int, default=15)
    args = p.parse_args()

    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('MY_PAT')
    if not token:
        print('Set GITHUB_TOKEN or MY_PAT in environment')
        return

    print(f'Polling CI for {args.repo} {args.branch} (every {args.interval}s)')
    while True:
        try:
            data = list_runs(args.repo, args.branch, token)
            runs = data.get('workflow_runs', [])
            if not runs:
                print('No workflow runs found')
            for r in runs[:5]:
                print(f"- {r['name']} #{r['run_number']} status={r['status']} conclusion={r['conclusion']} url={r['html_url']}")
        except Exception as e:
            print('Error fetching runs:', e)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
