#!/usr/bin/env bash
# Health check for HW2 background work: GPU temperature, clocks and throttling, running jobs, their latest
# progress line, stalled logs, free disk. Prints a report and one summary line.
# Exit code: 0 = no background work, 10 = work running and healthy, 20 = work running with a warning.
#
# bash scripts/health_check.sh            (from hw2/; HOT_C and STALL_MIN override the thresholds; 84 C = this card's max operating temp from nvidia-smi -q -d TEMPERATURE, NVIDIA spec max 90 C)

HW2="$(cd "$(dirname "$0")/.." && pwd)"
HOT_C="${HOT_C:-84}"
STALL_MIN="${STALL_MIN:-20}"
warn=()

echo "== health check $(date '+%F %T')"

if command -v nvidia-smi >/dev/null; then
    IFS=', ' read -r temp util sm smmax power fan memu memt reasons < <(nvidia-smi \
        --query-gpu=temperature.gpu,utilization.gpu,clocks.sm,clocks.max.sm,power.draw,fan.speed,memory.used,memory.total,clocks_throttle_reasons.active \
        --format=csv,noheader,nounits | head -1)
    echo "GPU: ${temp} C, util ${util}%, SM ${sm}/${smmax} MHz, ${power} W, fan ${fan}%, mem ${memu}/${memt} MiB, throttle ${reasons}"
    [ "${temp%.*}" -ge "$HOT_C" ] && warn+=("GPU at ${temp} C (>= ${HOT_C})")
    case "$reasons" in
        0x0000000000000000|0x0000000000000001|0x0000000000000004|0x0000000000000400) ;;
        *) [ "${util%.*}" -gt 0 ] && warn+=("GPU clocks limited, reasons ${reasons} (see nvidia-smi -q -d PERFORMANCE)") ;;
    esac
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/GPU process: pid /'
else
    warn+=("nvidia-smi not found")
fi

jobs=$(ps -eo pid=,etime=,args= | grep -E "python[^ ]* .*scripts/(train|generate|bonus_ablation|tokenize|eval)[a-z_]*\.py" | grep -v -E "grep|health_check")
if [ -z "$jobs" ]; then
    echo "jobs: none"
else
    echo "jobs:"
    while read -r pid etime args; do
        echo "  pid $pid, running $etime: $(echo "$args" | cut -c1-160)"
        run=$(echo "$args" | grep -oE -- "--run_name [^ ]+" | cut -d' ' -f2)
        log=""
        [ -n "$run" ] && [ -f "$HW2/runs/$run.log" ] && log="$HW2/runs/$run.log"
        [ -z "$log" ] && [ -n "$run" ] && [ -f "$HW2/runs/$run/log.csv" ] && log="$HW2/runs/$run/log.csv"
        if [ -n "$log" ]; then
            age=$(( ($(date +%s) - $(stat -c %Y "$log")) / 60 ))
            last=$(grep -E "^step|^[0-9]+," "$log" | tail -1)
            echo "    log ${log#$HW2/}, updated ${age} min ago: ${last:-no step line yet}"
            [ "$age" -ge "$STALL_MIN" ] && warn+=("$run log not updated for ${age} min")
        fi
    done <<< "$jobs"
fi

for log in $(find "$HW2/runs" "${CLAUDE_JOB_DIR:-/nonexistent}/tmp" -maxdepth 1 -name "*.log" -mmin -60 2>/dev/null); do
    if grep -qiE "Traceback|CUDA error|out of memory|Killed" "$log"; then
        warn+=("error text in ${log}: $(grep -iE "Traceback|CUDA error|out of memory|Killed" "$log" | tail -1 | cut -c1-120)")
    fi
done

free=$(df -h --output=avail "$HW2" | tail -1 | tr -d ' ')
echo "disk free at hw2: $free"

for w in "${warn[@]}"; do echo "WARN: $w"; done
if [ -z "$jobs" ]; then
    echo "SUMMARY: no background work"
    exit 0
elif [ ${#warn[@]} -gt 0 ]; then
    echo "SUMMARY: work running, ${#warn[@]} warning(s)"
    exit 20
else
    echo "SUMMARY: work running, healthy"
    exit 10
fi
