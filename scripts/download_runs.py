import re
import subprocess
import os
import shutil

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
    dirname = f"run_{rid}_artifacts"
    outname = f"run-{rid}-logs.zip"
    print(f"Downloading logs for run {rid} ...")
    try:
        # Last ned til en egen katalog for å holde styr på filen
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                rid,
                "--repo",
                "binyaminsemerci-ops/quantum_trader",
                "--dir",
                dirname,
            ],
            check=True,
        )
        # Finn første .zip-fil i katalogen
        zips = [f for f in os.listdir(dirname) if f.endswith(".zip")]
        if not zips:
            print(f"Ingen zip-filer funnet for run {rid}")
            continue
        # Flytt/gi nytt navn til ønsket filnavn
        shutil.move(os.path.join(dirname, zips[0]), outname)
        print(f"Saved {outname}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download run {rid}: {e}")
    except Exception as e:
        print(f"Unexpected error for run {rid}: {e}")
    finally:
        # Rydd opp katalogen
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
