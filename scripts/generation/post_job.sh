#!/usr/bin/env bash
# The queue runner holds the lane while this runs (300 s limit), so only the sub-second steps run here; the slow pass is detached.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs/post

if [ "$1" = "--slow-pass" ]; then
  # One slow pass at a time; a pass that starts while another runs waits, then picks up whatever is new.
  exec 9> logs/post/slow.lock
  flock 9
  {
    date '+=== %m-%d %H:%M:%S slow pass'
    nice -n 10 ./.venv/bin/python scripts/evaluation/extra_metrics.py
    nice -n 10 ./.venv/bin/python scripts/evaluation/paired_fd_update.py --set dev --baseline cfg7
    ./.venv/bin/python scripts/analysis/knowledge.py
    date '+=== %m-%d %H:%M:%S slow pass done'
  } >> logs/post/post_job.log 2>&1
  exit 0
fi

{
  date '+=== %m-%d %H:%M:%S fast pass'
  ./.venv/bin/python scripts/analysis/findings_table.py > results/analysis/findings_table.md
  ./.venv/bin/python scripts/analysis/knowledge.py
} >> logs/post/post_job.log 2>&1
setsid nohup bash scripts/generation/post_job.sh --slow-pass > /dev/null 2>&1 < /dev/null &
exit 0
