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
  build [docker_image|sif_image]  Build docker image or apptainer SIF (default: both)
  push  [sif_image|ddm_data]     Transfer file(s) to the cluster (default: both)
  pull                           Download processed DDM results from the cluster
  help                           Show this message

Examples:
  $(basename "$0") build
  $(basename "$0") build docker_image
  $(basename "$0") build sif_image [name.sif]
  $(basename "$0") push
  $(basename "$0") push sif_image
  $(basename "$0") push ddm_data
  $(basename "$0") pull
EOF
}

cmd_build_docker() {
    docker build -t test:latest .
}

cmd_build_sif() {
    local sif_name="${1:-pd-prior.sif}"
    apptainer build "${sif_name}" docker-daemon://test:latest
}

cmd_push_sif() {
    scp pd-prior.sif "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/"
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
        target="${2:-all}"
        case "$target" in
            docker_image) cmd_build_docker ;;
            sif_image)    cmd_build_sif "${3:-}" ;;
            all)          cmd_build_docker; cmd_build_sif ;;
            *)
                echo "Unknown build target: $target (expected: docker_image, sif_image, or omit for both)" >&2
                exit 1
                ;;
        esac
        ;;
    push)
        target="${2:-all}"
        case "$target" in
            sif_image)      cmd_push_sif ;;
            ddm_data) cmd_push_ddm_data ;;
            all)      cmd_push_sif; cmd_push_ddm_data ;;
            *)
                echo "Unknown push target: $target (expected: sif_image, ddm_data, or omit for both)" >&2
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
