from tandemn_system_data.models.enums import RankRole
from tandemn_system_data.models.rank import Rank

from tandemn_orca.batch_compiler import compile_batch_job

JOB_ID = "job_01JBM2YQYZ1KQ9C8GZP1XB6V5T"
RANK_ID = "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8"


def _rank(*, node_count: int = 1, replicas: int = 2) -> Rank:
    return Rank(
        rank_id=RANK_ID,
        job_id=JOB_ID,
        plan_id="plan_1",
        role=RankRole.AGGREGATE,
        n_replicas=replicas,
        shape_json={
            "model_id": "microsoft/phi-4",
            "instance_type": "g2-standard-48",
            "count": 4,
            "tp": 2,
            "pp": 2,
            "node_count": node_count,
        },
    )


def _env(container: dict) -> dict[str, str]:
    return {item["name"]: item["value"] for item in container["env"]}


def test_compile_single_node_job_per_chain():
    objects = compile_batch_job(JOB_ID, [_rank()], "tandemn-system", "chunk-manager:9090")

    assert [obj["kind"] for obj in objects] == ["Job", "Job"]
    assert (
        objects[0]["spec"]["template"]["metadata"]["labels"]["tandemn.com/pods-discovery"]
        == "batch-worker"
    )
    assert {
        obj["spec"]["template"]["metadata"]["labels"]["tandemn.com/chain-id"] for obj in objects
    } == {
        "0",
        "1",
    }
    pod = objects[0]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert pod["nodeSelector"] == {"cloud.google.com/gke-nodepool": "g2-standard-48"}
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert container["readinessProbe"]["httpGet"] == {"path": "/health", "port": "vllm"}
    assert _env(container) == {
        "TD_VLLM_MODEL": "microsoft/phi-4",
        "TD_VLLM_HOST": "0.0.0.0",
        "TD_VLLM_EXTRA_ARGS": "--pipeline-parallel-size=2 --tensor-parallel-size=2",
        "TD_CHUNK_MANAGER_ADDRESS": "chunk-manager:9090",
        "TD_JOB_ID": "01JBM2YQYZ1KQ9C8GZP1XB6V5T",
        "TD_RANK_ID": "01JBM30YQ7X3WQAR6HF8C2Q9T8",
        "TD_CHAIN_ID": "0",
    }


def test_compile_multinode_lws_per_chain():
    rank = _rank()
    rank.shape_json.pop("node_count")
    rank.shape_json["num_nodes_per_chain"] = 2
    objects = compile_batch_job(JOB_ID, [rank], "tandemn-system", "chunk-manager:9090")

    assert [obj["kind"] for obj in objects] == ["LeaderWorkerSet", "LeaderWorkerSet"]
    lws = objects[0]["spec"]
    assert lws["replicas"] == 1
    assert lws["leaderWorkerTemplate"]["size"] == 2
    leader = lws["leaderWorkerTemplate"]["leaderTemplate"]["spec"]["containers"][0]
    worker = lws["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    assert (
        lws["leaderWorkerTemplate"]["workerTemplate"]["metadata"]["labels"][
            "tandemn.com/pods-discovery"
        ]
        == "batch-worker"
    )
    assert leader["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert worker["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert "--nnodes $(LWS_GROUP_SIZE)" in _env(leader)["TD_VLLM_EXTRA_ARGS"]
    assert "--headless" in worker["args"][0]


def test_batch_workload_name_is_stable_across_plans():
    rank = _rank(replicas=1)
    first = compile_batch_job(JOB_ID, [rank], "ns", "chunk-manager:9090")[0]
    second = compile_batch_job(
        JOB_ID,
        [rank.model_copy(update={"plan_id": "plan_2"})],
        "ns",
        "chunk-manager:9090",
    )[0]

    assert first["metadata"]["name"] == second["metadata"]["name"]
    assert "tandemn.com/plan-id" not in first["metadata"]["labels"]
    assert "tandemn.com/plan-id" not in first["spec"]["template"]["metadata"]["labels"]


def test_batch_worker_accepts_secret_and_aws_region():
    job = compile_batch_job(
        JOB_ID,
        [_rank(replicas=1)],
        "ns",
        "chunk-manager:9090",
        worker_secret="tandemn-worker-secrets",
        aws_region="us-east-2",
    )[0]
    container = job["spec"]["template"]["spec"]["containers"][0]

    assert container["envFrom"] == [
        {"secretRef": {"name": "tandemn-worker-secrets", "optional": False}}
    ]
    assert _env(container)["AWS_DEFAULT_REGION"] == "us-east-2"
