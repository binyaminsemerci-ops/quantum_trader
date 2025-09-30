import os
import sys
import runpy

# Use an in-memory sqlite DB by overriding the env var so the scripts create
# their engine against an ephemeral DB for the test run. Running both scripts
# via runpy in the same process means they can share the same module state.
os.environ['QUANTUM_TRADER_DATABASE_URL'] = 'sqlite:///:memory:'

# runpy.run_path will execute the script which uses argparse and reads
# sys.argv — pytest injects its own CLI args which will confuse those
# parsers. Set sys.argv explicitly for each script run to avoid that.
old_argv = sys.argv.copy()
try:
    sys.argv[:] = ['scheduler.py', '--once', '--limit', '1']
    runpy.run_path('scripts/scheduler.py', run_name='__main__')

    sys.argv[:] = ['autotrader.py', '--once', '--max-symbols', '3', '--dry-run']
    runpy.run_path('scripts/autotrader.py', run_name='__main__')
finally:
    sys.argv[:] = old_argv


def test_sanity():
    # If the scripts raised an exception during import/run the test would fail
    assert True
