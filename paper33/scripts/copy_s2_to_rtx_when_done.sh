#!/usr/bin/env bash
# Wait until all 10 small-pair seed-2 evaluations are done on the 4090, then copy their held-out responses to the RTX
# (results/import_4090/<label>/refusal/responses.jsonl, md5-verified) so judge__batch2 picks them up.
P=/data/heretic_2/33_2611_does_refusal_removal_get_harder; L=$P/runs/local-4090/logs; LOG=$P/runs/s2_copy.log
labs=$(grep -v -E '^\s*(#|$)' $P/runs/local-4090/bench_queue.txt | cut -d'|' -f1 | tr -d ' ' | grep -- '-s2$')
until [ -z "$(for l in $labs; do grep -q "^.* $l rc=0$" $L/bench_runs_index.log || echo $l; done)" ]; do sleep 120; done
for l in $labs; do f=$P/results/bench-4090/$l/refusal/responses.jsonl
  ssh mc-rtx "mkdir -p /workspace/paper33/results/import_4090/$l/refusal" && scp -q $f mc-rtx:/workspace/paper33/results/import_4090/$l/refusal/responses.jsonl \
  && [ "$(md5sum < $f)" = "$(ssh mc-rtx "md5sum < /workspace/paper33/results/import_4090/$l/refusal/responses.jsonl")" ] \
  && echo "$(date -Is) copied $l" >> $LOG || echo "$(date -Is) COPY FAILED $l" >> $LOG; done
