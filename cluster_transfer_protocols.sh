#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${SCRIPT_DIR}/.env"
set +a

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [target]

Commands:
  build            Build the Singularity image from the local Docker daemon
  push [sif|ddm_data]  Transfer file(s) to the cluster (default: both sif and ddm_data)
  pull             Download processed DDM results from the cluster
  help             Show this message

Examples:
  $(basename "$0") build
  $(basename "$0") push
  $(basename "$0") push sif
  $(basename "$0") push ddm_data
  $(basename "$0") pull
EOF
}

cmd_build() {
    singularity build nhp-prior.sif docker-daemon://test:latest
}

cmd_push_sif() {
    scp nhp-prior.sif "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/"
}

cmd_push_ddm_data() {
    scp "${CONTAINER_DATA_PATH}/processed/ddm/behavior_data.csv" \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/ddm/"
}

cmd_pull() {
    scp -r "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/ddm/" \
        "${CONTAINER_DATA_PATH}/processed/"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

case "$1" in
    build)
        cmd_build
        ;;
    push)
        target="${2:-all}"
        case "$target" in
            sif)      cmd_push_sif ;;
            ddm_data) cmd_push_ddm_data ;;
            all)      cmd_push_sif; cmd_push_ddm_data ;;
            *)
                echo "Unknown push target: $target (expected: sif, ddm_data, or omit for both)" >&2
                exit 1
                ;;
        esac
        ;;
    pull)
        cmd_pull
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1" >&2
        usage
        exit 1
        ;;
esac
