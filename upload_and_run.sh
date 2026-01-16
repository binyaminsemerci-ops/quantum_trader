#!/bin/bash
scp -i ~/.ssh/hetzner_fresh /mnt/c/quantum_trader/deploy_rl_shadow.sh root@46.224.116.254:/tmp/
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'bash /tmp/deploy_rl_shadow.sh'
