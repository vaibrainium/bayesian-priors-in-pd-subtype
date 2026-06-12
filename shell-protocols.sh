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
  build [docker_image|sif-image]            Build docker image or apptainer SIF (default: both)
  push  [sif-image|ddm-data|glm-hmm-data]   Transfer file(s) to the cluster (default: sif + ddm-data)
  pull  [ddm-models|glm-hmm-models]         Download processed results from the cluster
  help                                      Show this message

Examples:
  $(basename "$0") build
  $(basename "$0") build sif-image [name.sif]
  $(basename "$0") push
  $(basename "$0") push sif-image
  $(basename "$0") push ddm-data
  $(basename "$0") push glm-hmm-data
  $(basename "$0") pull ddm-models
  $(basename "$0") pull glm-hmm-models
  $(basename "$0") pull shared-glm-hmm-models
EOF
}

cmd_build_sif() {
    docker build -t test:latest .
    local sif_name="${1:-pd-prior.sif}"
    apptainer build "${sif_name}" docker-daemon://test:latest
}

cmd_push_sif() {
    scp pd-prior.sif "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/"
    rm test:latest
    rm "${sif_name}"
}

cmd_push_ddm_data() {
    scp "${CONTAINER_DATA_PATH}/processed/ddm/behavior_data.csv" \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/ddm/"
}

cmd_pull_ddm_models() {
    scp -r "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/ddm/" \
        "${CONTAINER_DATA_PATH}/processed/"
}

cmd_push_glm_hmm_data() {
    scp "${CONTAINER_DATA_PATH}/processed/processed_all_data_accu_60_all.csv" \
        "${CONTAINER_DATA_PATH}/processed/processed_metadata_all_data_accu_60.csv" \
        "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/"
}

cmd_pull_glm_hmm_models() {
    scp -r "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/glm_hmm/" \
        "${CONTAINER_DATA_PATH}/processed/"
}

cmd_pull_shared_glm_hmm_models() {
    scp -r "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_DATA_PATH}/processed/shared_glm_hmm/" \
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
            sif-image)    cmd_build_sif "${3:-}" ;;
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
            sif-image)    cmd_push_sif ;;
            ddm-data)     cmd_push_ddm_data ;;
            glm-hmm-data) cmd_push_glm_hmm_data ;;
            all)          cmd_push_sif; cmd_push_ddm_data ;;
            *)
                echo "Unknown push target: $target (expected: sif-image, ddm-data, glm-hmm-data, or omit for both)" >&2
                exit 1
                ;;
        esac
        ;;
    pull)
        target="${2:-all}"
        case "$target" in
            ddm-models)     cmd_pull_ddm_models ;;
            glm-hmm-models) cmd_pull_glm_hmm_models ;;
            shared-glm-hmm-models) cmd_pull_shared_glm_hmm_models ;;
            *)
                echo "Unknown pull target: $target (expected: ddm-models, glm-hmm-models, shared-glm-hmm-models)" >&2
                exit 1
                ;;
        esac
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
