#!/bin/bash

set -e

TIDB_VERSION=v8.4.0

# Download tiup
function prepare_tidb () {
  curl --proto '=https' --tlsv1.2 -sSf https://tiup-mirrors.pingcap.com/install.sh | sh
  PATH=${PATH}:/home/runner/.tiup/bin
  tiup install playground tidb:${TIDB_VERSION} pd:${TIDB_VERSION} tikv:${TIDB_VERSION} tiflash:${TIDB_VERSION}
}

# Pull docker images
function prepare_docker_images() {
  docker compose pull frontend background backend redis static-web-server
}

prepare_tidb &
PID1=$!

prepare_docker_images &
PID2=$!

wait $PID1
wait $PID2
