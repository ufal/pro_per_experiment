#!/bin/bash
# Submit cluster jobs for all experiment combinations that don't have an output file yet.
# Output files are expected at: outputs/v2/llm{L}_{mode}_pro{P}_eval{E}.tsv
#
# Usage:
#   bash submit_missing.sh           # submit missing jobs
#   bash submit_missing.sh --dry-run # print what would be submitted, don't actually submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/outputs/v2"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] No jobs will be submitted."
fi

# Count entries in each config file
n_llms=$(grep -c . "$SCRIPT_DIR/llms")
n_pro=$(grep -c . "$SCRIPT_DIR/pro_temp")
n_perc=$(grep -c . "$SCRIPT_DIR/perc_temp")

echo "LLMs: $n_llms  |  pro_temp: $n_pro  |  perc_temp: $n_perc"
echo "Output dir: $OUTPUT_DIR"
echo ""

submitted=0
skipped=0

submit_or_skip() {
    local l=$1 mode=$2 p=$3 e=$4
    local outfile="$OUTPUT_DIR/llm${l}_${mode}_pro${p}_eval${e}.tsv"
    if [[ -f "$outfile" ]]; then
        echo "  SKIP  llm=$l mode=$mode pro=$p eval=$e  (output exists)"
        ((skipped++)) || true
    else
        echo "  SUBMIT  llm=$l mode=$mode pro=$p eval=$e"
        if ! $DRY_RUN; then
            sbatch --job-name="exp_llm${l}_${mode}_p${p}_e${e}" \
                "$SCRIPT_DIR/run_cluster_job.sh" \
                --llm "$l" --mode "$mode" --pro-template "$p" --eval-template "$e" \
                --output "$outfile"
        fi
        ((submitted++)) || true
    fi
}

# pro-perc: all llm × pro_temp × perc_temp combinations
echo "=== pro-perc ==="
for ((l=0; l<n_llms; l++)); do
    for ((p=0; p<n_pro; p++)); do
        for ((e=0; e<n_perc; e++)); do
            submit_or_skip "$l" "pro-perc" "$p" "$e"
        done
    done
done

# pro-pro: all llm × pro_temp pairs where gen != eval
echo ""
echo "=== pro-pro ==="
for ((l=0; l<n_llms; l++)); do
    for ((p=0; p<n_pro; p++)); do
        for ((e=0; e<n_pro; e++)); do
            [[ "$p" == "$e" ]] && continue
            submit_or_skip "$l" "pro-pro" "$p" "$e"
        done
    done
done

echo ""
if $DRY_RUN; then
    echo "Dry run complete. Would submit: $submitted  |  Would skip: $skipped"
else
    echo "Done. Submitted: $submitted  |  Skipped: $skipped"
fi
