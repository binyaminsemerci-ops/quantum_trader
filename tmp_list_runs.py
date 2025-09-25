import os,requests,json
token=os.environ.get('GITHUB_TOKEN') or os.environ.get('MY_PAT')
if not token:
    raise SystemExit('no token')
url='https://api.github.com/repos/binyaminsemerci-ops/quantum_trader/actions/runs'
r=requests.get(url, params={'branch':'chore/mypy-fixes','per_page':10}, headers={'Authorization':f'token {token}','Accept':'application/vnd.github.v3+json'}, timeout=30)
r.raise_for_status()
data=r.json()
runs=data.get('workflow_runs',[])
for i,run in enumerate(runs):
    print(f"{i+1}. name={run.get('name')} id={run.get('id')} run_number={run.get('run_number')} status={run.get('status')} conclusion={run.get('conclusion')} url={run.get('html_url')}")
