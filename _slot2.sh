#!/bin/bash
echo "=slots desired=" 
redis-cli get quantum:slots:desired
echo "=slots available="
redis-cli get quantum:slots:available
echo "=slot keys="
redis-cli keys "quantum:slots:*"
echo "=position state keys="
redis-cli keys "quantum:state:position*" | head -10
echo "=portfolio state="
redis-cli hgetall quantum:state:portfolio | head -20
echo "=active ledger positions count="
redis-cli keys "quantum:ledger:position:*" | wc -l
echo "=ledger position keys sample="
redis-cli keys "quantum:ledger:position:*" | head -5
echo "=authoritative source check="
redis-cli get "quantum:state:slot_count"
redis-cli get "quantum:state:position_count"
redis-cli hget quantum:state:portfolio open_positions
redis-cli hget quantum:state:portfolio active_slots
