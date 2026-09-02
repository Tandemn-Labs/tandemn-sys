#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SYSTEM_ROOT="$ROOT/tandemn-system"
STORE_ROOT="$ROOT/tandemn-store"
STATE_FILE="${STATE_FILE:-/tmp/tandemn-gke-batch-e2e.env}"

STORE_URL="${TANDEMN_POSTGRES_URL:-postgresql+psycopg://tandemn:tandemn@127.0.0.1:55432/tandemn}"
GKE_CONTEXT="${GKE_CONTEXT:-gke_tandemn_us-central1_tandemn-us-central1}"
CHUNK_MANAGER_TARGET="${TANDEMN_CHUNK_MANAGER_TARGET:-ab2e44b704c384775ad2b3c3842bd8ae-79009fb0ce2fd129.elb.us-east-1.amazonaws.com:9090}"
S3_BUCKET="${S3_BUCKET:-batched-chunks}"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-00000000000000000000000001}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
NODE_POOL="${NODE_POOL:-g2-standard-8}"
AWS_REGION="${AWS_REGION:-us-east-2}"
WORKER_SECRET="${WORKER_SECRET:-tandemn-worker-secrets}"

PROM_SESSION=tandemn-batch-e2e-prometheus
COLLECTOR_SESSION=tandemn-batch-e2e-collector
ORCA_SESSION=tandemn-batch-e2e-orca

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null || die "$1 is required"
}

require_state() {
  [[ -f "$STATE_FILE" ]] || die "state file not found: $STATE_FILE"
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

stop_session() {
  tmux has-session -t "$1" 2>/dev/null && tmux kill-session -t "$1"
  return 0
}

start() {
  [[ ! -f "$STATE_FILE" ]] || die "existing run found; use cleanup first: $STATE_FILE"
  for command in aws curl docker jq kubectl tmux uv; do
    require_command "$command"
  done
  kubectl config use-context "$GKE_CONTEXT" >/dev/null
  if pgrep -f 'tandemn-gpu-metrics-collector' >/dev/null; then
    die "another GPU metrics collector is running; stop it to avoid duplicate Store rows"
  fi

  docker compose --project-directory "$STORE_ROOT" up -d postgres
  TANDEMN_POSTGRES_URL="$STORE_URL" uv run --project "$STORE_ROOT" alembic upgrade head

  TANDEMN_POSTGRES_URL="$STORE_URL" \
  CHUNK_MANAGER_TARGET="$CHUNK_MANAGER_TARGET" \
  S3_BUCKET="$S3_BUCKET" \
  SOURCE_JOB_ID="$SOURCE_JOB_ID" \
  MODEL_ID="$MODEL_ID" \
  AWS_REGION="$AWS_REGION" \
  uv run --project "$SYSTEM_ROOT" python - <<'PY' > "$STATE_FILE"
import os
import shlex
from pathlib import PurePosixPath

import boto3
import grpc

from tandemn.chunkmanager.v1 import chunk_manager_pb2, chunk_manager_pb2_grpc
from tandemn_orca.scripts.init_chunk_job import initialize_job
from tandemn_orca.scripts.submit_job import normalize_job_spec
from tandemn_system_data.clients import JobStore, PostgresClient, UserStore
from tandemn_system_data.models import Job, JobKind, User

client = PostgresClient()
user = User(name="gke-batch-e2e")
UserStore(client).ensure(user)
model_id = os.environ["MODEL_ID"]
spec = normalize_job_spec(
    {"model_id": model_id, "deadline_hrs": 1, "total_token_budget": 25600},
    JobKind.BATCH,
)
job = Job(user_id=user.user_id, kind=JobKind.BATCH, spec_json=spec)
JobStore(client).submit(job)
bare_job_id = job.job_id.removeprefix("job_")

s3 = boto3.client("s3", region_name=os.environ["AWS_REGION"])
bucket = os.environ["S3_BUCKET"]
source_prefix = f"{os.environ['SOURCE_JOB_ID']}/input/"
keys = sorted(
    item["Key"]
    for item in s3.list_objects_v2(Bucket=bucket, Prefix=source_prefix).get("Contents", [])
    if item["Key"].endswith(".jsonl")
)
if not keys:
    raise SystemExit(f"no source chunks found at s3://{bucket}/{source_prefix}")

chunks = []
for key in keys:
    chunk_id = int(PurePosixPath(key).stem)
    target_key = f"{bare_job_id}/input/{chunk_id}.jsonl"
    s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=target_key)
    chunks.append(
        chunk_manager_pb2.ChunkRegistration(
            chunk_id=chunk_id,
            input_ref=f"s3://{bucket}/{target_key}",
        )
    )

with grpc.insecure_channel(os.environ["CHUNK_MANAGER_TARGET"]) as channel:
    initialize_job(
        chunk_manager_pb2_grpc.PlannerServiceStub(channel),
        job.job_id,
        chunks,
        max_retries=2,
        retry_backoff_seconds=1,
        lease_duration_seconds=120,
    )

values = {
    "TANDEMN_USER_ID": user.user_id,
    "JOB_ID": job.job_id,
    "BARE_JOB_ID": bare_job_id,
    "CHUNK_COUNT": str(len(chunks)),
}
for key, value in values.items():
    print(f"export {key}={shlex.quote(value)}")
PY

  require_state
  cat >> "$STATE_FILE" <<EOF
export TANDEMN_POSTGRES_URL=$(printf '%q' "$STORE_URL")
export TANDEMN_CHUNK_MANAGER_TARGET=$(printf '%q' "$CHUNK_MANAGER_TARGET")
export S3_BUCKET=$(printf '%q' "$S3_BUCKET")
export MODEL_ID=$(printf '%q' "$MODEL_ID")
EOF
  require_state

  if ! curl -fsS http://127.0.0.1:9090/-/ready >/dev/null 2>&1; then
    tmux new-session -d -s "$PROM_SESSION" -c "$ROOT" \
      "kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090"
  fi

  tmux new-session -d -s "$COLLECTOR_SESSION" -c "$SYSTEM_ROOT" \
    "env TANDEMN_POSTGRES_URL='$STORE_URL' TANDEMN_USER_ID='$TANDEMN_USER_ID' uv run tandemn-gpu-metrics-collector --prometheus-url http://127.0.0.1:9090 --kube-context '$GKE_CONTEXT' --namespace dynamo-system --batch-namespace tandemn-system --user-id '$TANDEMN_USER_ID'"

  tmux new-session -d -s "$ORCA_SESSION" -c "$SYSTEM_ROOT" \
    "env TANDEMN_POSTGRES_URL='$STORE_URL' TANDEMN_USER_ID='$TANDEMN_USER_ID' uv run tandemn-orca --user-id '$TANDEMN_USER_ID' --namespace dynamo-system --batch-namespace tandemn-system --chunk-manager-target '$CHUNK_MANAGER_TARGET' --batch-worker-secret '$WORKER_SECRET' --batch-aws-region '$AWS_REGION' --skip-capacity-refresh --interval-seconds 5"

  TANDEMN_POSTGRES_URL="$STORE_URL" \
  TANDEMN_USER_ID="$TANDEMN_USER_ID" \
  JOB_ID="$JOB_ID" \
  MODEL_ID="$MODEL_ID" \
  NODE_POOL="$NODE_POOL" \
  uv run --project "$SYSTEM_ROOT" python - <<'PY' >> "$STATE_FILE"
import os
import shlex

from tandemn_system_data.clients import PlanStore, PostgresClient
from tandemn_system_data.ids import new_rank_id
from tandemn_system_data.models import ActionType, Plan, PlanAction

rank_id = new_rank_id()
plan = Plan(
    user_id=os.environ["TANDEMN_USER_ID"],
    tick_rationale="one-GPU GKE batch E2E",
    actions=[
        PlanAction(
            job_id=os.environ["JOB_ID"],
            type=ActionType.PLACE,
            ladder=[
                {
                    "role": "aggregate",
                    "rank_id": rank_id,
                    "env": ["on_demand", "gcp", "us-central1", "us-central1-c", "L4"],
                    "config": {
                        "model_id": os.environ["MODEL_ID"],
                        "engine_name": "vllm",
                        "instance_type": os.environ["NODE_POOL"],
                        "gpu_type": "L4",
                        "gpu_count": 1,
                        "node_count": 1,
                        "tp": 1,
                        "pp": 1,
                    },
                    "n_replicas": 1,
                }
            ],
        )
    ],
)
PlanStore(PostgresClient()).create(plan)
print(f"export RANK_ID={shlex.quote(rank_id)}")
print(f"export PLAN_ID={shlex.quote(plan.plan_id)}")
PY

  require_state
  printf 'Started batch E2E\n'
  printf '  user:   %s\n' "$TANDEMN_USER_ID"
  printf '  job:    %s\n' "$JOB_ID"
  printf '  rank:   %s\n' "$RANK_ID"
  printf '  plan:   %s\n' "$PLAN_ID"
  printf '  chunks: %s\n' "$CHUNK_COUNT"
  printf 'Run: %s status\n' "$0"
}

status() {
  require_state
  printf '%s\n' '--- Kubernetes ---'
  kubectl get jobs,pods -n tandemn-system -l "tandemn.com/job-id=$JOB_ID" -o wide
  printf '%s\n' '--- Processes ---'
  tmux list-sessions 2>/dev/null | grep 'tandemn-batch-e2e' || true
  printf '%s\n' '--- Store ---'
  TANDEMN_POSTGRES_URL="$STORE_URL" JOB_ID="$JOB_ID" \
    uv run --project "$STORE_ROOT" python - <<'PY'
import os
from tandemn_system_data.clients import GpuMetricStore, JobStore, PostgresClient

client = PostgresClient()
job_id = os.environ["JOB_ID"]
print(JobStore(client).get(job_id))
rows = GpuMetricStore(client).recent(job_id, limit=3)
for row in rows:
    print({
        "ts": row.ts.isoformat(),
        "rank_id": row.rank_id,
        "chain_index": row.chain_index,
        "inflight": row.batched_reqs_inflight,
        "processed": row.batched_reqs_processed_total,
        "pulled": row.batched_chunks_input_pulled_total,
        "written": row.batched_chunks_output_written_total,
        "throughput": row.throughput_token_per_sec,
    })
PY
}

wait_for_completion() {
  require_state
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}" \
  TANDEMN_POSTGRES_URL="$STORE_URL" \
  JOB_ID="$JOB_ID" \
    uv run --project "$STORE_ROOT" python - <<'PY'
import os
import time

from tandemn_system_data.clients import JobStore, PostgresClient
from tandemn_system_data.models import JobStatus

deadline = time.monotonic() + float(os.environ["TIMEOUT_SECONDS"])
store = JobStore(PostgresClient())
while time.monotonic() < deadline:
    job = store.get(os.environ["JOB_ID"])
    if job is None:
        raise SystemExit("Store job disappeared")
    print(job.status, job.finish_reason, flush=True)
    if job.status is JobStatus.FINISHED:
        raise SystemExit(0 if job.finish_reason is None else 1)
    time.sleep(10)
raise SystemExit("timed out waiting for Store completion")
PY
  status
}

cleanup() {
  require_state
  uv run --project "$SYSTEM_ROOT" python - <<PY
import grpc
from tandemn_orca.chunk_manager import ChunkManagerClient

client = ChunkManagerClient("$CHUNK_MANAGER_TARGET")
try:
    client.cancel_job("$JOB_ID")
    print("chunk-manager job cancelled")
except grpc.RpcError as error:
    print(f"chunk-manager cancellation skipped: {error.code().name}: {error.details()}")
finally:
    client.close()
PY
  sleep 10
  stop_session "$ORCA_SESSION"
  stop_session "$COLLECTOR_SESSION"
  stop_session "$PROM_SESSION"
  kubectl delete job -n tandemn-system -l "tandemn.com/job-id=$JOB_ID" --ignore-not-found
  aws s3 rm "s3://$S3_BUCKET/$BARE_JOB_ID/" --recursive
  rm -f "$STATE_FILE"
  printf 'Batch E2E resources removed; local Store Postgres was preserved.\n'
}

case "${1:-}" in
  start) start ;;
  status) status ;;
  wait) wait_for_completion ;;
  cleanup) cleanup ;;
  *)
    printf 'usage: %s {start|status|wait|cleanup}\n' "$0" >&2
    exit 2
    ;;
esac
