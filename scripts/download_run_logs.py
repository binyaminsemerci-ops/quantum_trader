import re
import subprocess

runs = [
    "https://github.com/binyaminsemerci-ops/quantum_trader/actions/runs/18052869000",
    "https://github.com/binyaminsemerci-ops/quantum_trader/actions/runs/18052868998",
    "https://github.com/binyaminsemerci-ops/quantum_trader/actions/runs/18052868999",
    "https://github.com/binyaminsemerci-ops/quantum_trader/actions/runs/18052869006",
    "https://github.com/binyaminsemerci-ops/quantum_trader/actions/runs/18052869009",
    "https://github.com/binyaminsemerci-ops/quantum_trader/actions/runs/18052868980",
    "https://github.com/binyaminsemerci-ops/quantum_trader/actions/runs/18052868849",
]

for url in runs:
    m = re.search(r"/actions/runs/(\d+)", url)
    if not m:
        print(f"Could not find run id in url: {url}")
        continue
    rid = m.group(1)
    outname = f"run-{rid}-log.txt"
    print(f"Fetching logs (text) for run {rid} ...")
    try:
        with open(outname, "w", encoding="utf-8") as fh:
            subprocess.run(
                [
                    "gh",
                    "run",
                    "view",
                    rid,
                    "--repo",
                    "binyaminsemerci-ops/quantum_trader",
                    "--log",
                ],
                check=True,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        print(f"Saved {outname}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch logs for run {rid}: {e}")
    except Exception as e:
        print(f"Unexpected error for run {rid}: {e}")
