"""Problem: Service sessionAffinity missing with per-pod in-memory session state.

Real-world failure class
------------------------
Teams scale a Deployment to multiple replicas and front it with a ClusterIP
Service while keeping session or user data **in process memory** (or on pod-local
disk). With the default ``sessionAffinity: None``, kube-proxy spreads requests
across endpoints. A write on pod A is invisible to pod B, so clients see
intermittent 404 / empty responses while every pod stays ``Running`` and probes
pass.

References
~~~~~~~~~~
* Kubernetes Service session affinity:
  https://kubernetes.io/docs/concepts/services-networking/service/#session-affinity
* Support Tools — session affinity and Kubernetes:
  https://support.tools/session-affinity-kubernetes/
* Stack Harbor — sticky sessions when affinity hides a bug:
  https://stackharbor.com/en/knowledge-base/sticky-session-when-affinity-hides-bug/

Simulation in SREGym
--------------------
A small ``session-store`` workload is deployed into the Hotel Reservation
namespace: three replicas, in-memory dict, Service on port 8080 with
``sessionAffinity: None``. Hotel Reservation microservices stay healthy.

Accepted mitigations
~~~~~~~~~~~~~~~~~~~~
* Patch Service to ``sessionAffinity: ClientIP`` (with timeoutSeconds).
* Scale the session-store Deployment to ``replicas: 1``.

Rejected mitigations
~~~~~~~~~~~~~~~~~~~~
* Delete the session-store workload only.
* Restart unrelated Hotel Reservation Deployments without fixing routing/state.
"""

from __future__ import annotations

import time

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from sregym.conductor.oracles.llm_as_a_judge.llm_as_a_judge_oracle import LLMAsAJudgeOracle
from sregym.conductor.oracles.service_session_affinity_missing_local_state_mitigation import (
    ServiceSessionAffinityMissingLocalStateMitigationOracle,
)
from sregym.conductor.problems.base import Problem
from sregym.service.apps.hotel_reservation import HotelReservation
from sregym.service.kubectl import KubeCtl
from sregym.utils.decorators import mark_fault_injected

DEPLOYMENT_NAME = "session-store"
SERVICE_NAME = "session-store"
SERVICE_PORT = 8080
REPLICAS_FAULT = 3
REPLICAS_RECOVERED = 1

PLATFORM_LABELS = {
    "app.kubernetes.io/part-of": "platform",
    "app.kubernetes.io/component": "session-store",
}

_SESSION_STORE_SCRIPT = r"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSIONS = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _session_key(self):
        if not self.path.startswith("/session/"):
            return None
        return self.path.split("/session/", 1)[1].split("?", 1)[0]

    def do_GET(self):
        key = self._session_key()
        if key is None:
            self.send_response(404)
            self.end_headers()
            return
        if key == "healthz":
            self.send_response(200)
            self.end_headers()
            return
        if key not in SESSIONS:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(SESSIONS[key]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        key = self._session_key()
        if key is None:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        SESSIONS[key] = json.loads(raw.decode() or "{}")
        self.send_response(200)
        self.end_headers()


ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
"""


class ServiceSessionAffinityMissingLocalStateHotelReservation(Problem):
    """Session stickiness missing while session-store keeps per-pod memory."""

    DEPLOYMENT_NAME = DEPLOYMENT_NAME
    SERVICE_NAME = SERVICE_NAME
    SERVICE_PORT = SERVICE_PORT
    REPLICAS_FAULT = REPLICAS_FAULT

    def __init__(self, faulty_service: str = SERVICE_NAME):
        self.faulty_service = faulty_service
        self.app = HotelReservation()
        super().__init__(app=self.app, namespace=self.app.namespace)

        self.namespace = self.app.namespace
        self.kubectl = KubeCtl()
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()

        self.root_cause = self.build_structured_root_cause(
            component=f"Service/{SERVICE_NAME}",
            namespace=self.namespace,
            description=(
                f"A platform session-store Deployment '{DEPLOYMENT_NAME}' in namespace "
                f"'{self.namespace}' runs {REPLICAS_FAULT} replicas with an in-memory "
                "session map per pod. The ClusterIP Service keeps the default "
                "`sessionAffinity: None`, so kube-proxy load-balances requests across "
                "pods. POST /session/{id} on one replica is not visible to GET on "
                "another, producing intermittent 404 responses while all pods stay "
                "Running and readiness probes pass. Hotel Reservation microservices "
                "remain healthy; this is Service routing plus local state, not Mongo "
                "auth_miss/revoke_auth, wrong Service selectors, duplicate PVC mounts, "
                "or internalTrafficPolicy: Local. Accepted mitigations: patch the Service "
                "to sessionAffinity: ClientIP (with clientIP timeoutSeconds), or scale "
                "session-store to replicas: 1. Deleting only the session-store workload "
                "or restarting unrelated HR Deployments without fixing affinity/replicas "
                "are rejected."
            ),
        )

        self.diagnosis_oracle = LLMAsAJudgeOracle(problem=self, expected=self.root_cause)
        self.app.create_workload()
        self.mitigation_oracle = ServiceSessionAffinityMissingLocalStateMitigationOracle(problem=self)

    @mark_fault_injected
    def inject_fault(self):
        print("== Fault Injection ==")
        self._delete_fault_objects_quiet()
        self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace,
            body=self._build_deployment(replicas=REPLICAS_FAULT),
        )
        self.core_v1.create_namespaced_service(
            namespace=self.namespace,
            body=self._build_service(session_affinity_client_ip=False),
        )
        self._wait_for_deployment_ready(DEPLOYMENT_NAME, expected_replicas=REPLICAS_FAULT)
        self._wait_for_split_brain_symptom()
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    @mark_fault_injected
    def recover_fault(self):
        print("== Fault Recovery ==")
        self._delete_fault_objects_quiet()
        self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace,
            body=self._build_deployment(replicas=REPLICAS_RECOVERED),
        )
        self.core_v1.create_namespaced_service(
            namespace=self.namespace,
            body=self._build_service(session_affinity_client_ip=True),
        )
        self._wait_for_deployment_ready(DEPLOYMENT_NAME, expected_replicas=REPLICAS_RECOVERED)
        self._wait_for_session_probe_success(min_success_rate=0.9)
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def _build_deployment(self, *, replicas: int) -> dict:
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": DEPLOYMENT_NAME,
                "namespace": self.namespace,
                "labels": {
                    **PLATFORM_LABELS,
                    "app.kubernetes.io/name": DEPLOYMENT_NAME,
                },
            },
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app.kubernetes.io/name": DEPLOYMENT_NAME}},
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": DEPLOYMENT_NAME}},
                    "spec": {
                        "containers": [
                            {
                                "name": "store",
                                "image": "python:3.12-alpine",
                                "command": ["python", "-c", _SESSION_STORE_SCRIPT],
                                "ports": [{"containerPort": SERVICE_PORT, "name": "http"}],
                                "readinessProbe": {
                                    "httpGet": {"path": "/session/healthz", "port": SERVICE_PORT},
                                    "periodSeconds": 5,
                                    "initialDelaySeconds": 3,
                                    "failureThreshold": 6,
                                },
                                "resources": {
                                    "requests": {"cpu": "25m", "memory": "64Mi"},
                                    "limits": {"cpu": "200m", "memory": "128Mi"},
                                },
                            }
                        ],
                    },
                },
            },
        }

    def _build_service(self, *, session_affinity_client_ip: bool) -> dict:
        spec: dict = {
            "selector": {"app.kubernetes.io/name": DEPLOYMENT_NAME},
            "ports": [{"name": "http", "port": SERVICE_PORT, "targetPort": SERVICE_PORT}],
            "type": "ClusterIP",
        }
        if session_affinity_client_ip:
            spec["sessionAffinity"] = "ClientIP"
            spec["sessionAffinityConfig"] = {
                "clientIP": {"timeoutSeconds": 10800},
            }
        else:
            spec["sessionAffinity"] = "None"
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": SERVICE_NAME,
                "namespace": self.namespace,
                "labels": {
                    **PLATFORM_LABELS,
                    "app.kubernetes.io/name": SERVICE_NAME,
                },
            },
            "spec": spec,
        }

    def _wait_for_split_brain_symptom(self) -> None:
        print(
            "Waiting for intermittent session GET failures via the Service "
            "(sessionAffinity=None, replicas>1)..."
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            ok, fail = self.run_session_probe(attempts=20)
            print(f"  Probe: {ok} successes, {fail} failures (via {SERVICE_NAME}:{SERVICE_PORT})")
            if ok >= 1 and fail >= 3:
                return
            time.sleep(10)
        raise RuntimeError(
            "Timed out waiting for split-brain symptom (need POST success and intermittent GET 404s)."
        )

    def _wait_for_session_probe_success(self, *, min_success_rate: float) -> None:
        print(f"Waiting for session probe success rate >= {min_success_rate:.0%}...")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            ok, fail = self.run_session_probe(attempts=15)
            total = ok + fail
            if total and (ok / total) >= min_success_rate:
                return
            time.sleep(5)
        print("⚠️ Timed out waiting for reliable session reads after recovery.")

    def run_session_probe(self, *, attempts: int) -> tuple[int, int]:
        """POST a session key then GET it repeatedly through the ClusterIP Service."""
        job_name = f"session-probe-{int(time.time()) % 1_000_000}"
        script = f"""set -eu
KEY=probe-key
BASE=http://{SERVICE_NAME}:{SERVICE_PORT}/session/$KEY
curl -sf -X POST "$BASE" -H 'Content-Type: application/json' -d '{{"ok":true}}'
ok=0
fail=0
i=0
while [ $i -lt {attempts} ]; do
  code=$(curl -s -o /dev/null -w '%{{http_code}}' "$BASE" || echo 000)
  if [ "$code" = "200" ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
  i=$((i+1))
done
echo OK=$ok FAIL=$fail
"""
        try:
            self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body={
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "metadata": {"name": job_name, "namespace": self.namespace},
                    "spec": {
                        "ttlSecondsAfterFinished": 120,
                        "backoffLimit": 0,
                        "template": {
                            "metadata": {"labels": {"job-name": job_name}},
                            "spec": {
                                "restartPolicy": "Never",
                                "automountServiceAccountToken": False,
                                "containers": [
                                    {
                                        "name": "probe",
                                        "image": "curlimages/curl:8.11.1",
                                        "command": ["sh", "-c", script],
                                    }
                                ],
                            },
                        },
                    },
                },
            )
            self._wait_for_job_complete(job_name, timeout_s=120)
            logs = self._read_job_pod_logs(job_name)
            return self._parse_probe_output(logs)
        finally:
            self._delete_job_quiet(job_name)

    def _parse_probe_output(self, logs: str) -> tuple[int, int]:
        ok = fail = 0
        for line in logs.splitlines():
            line = line.strip()
            if line.startswith("OK="):
                parts = line.replace("OK=", "").split(" FAIL=")
                if parts:
                    ok = int(parts[0])
                if len(parts) > 1:
                    fail = int(parts[1])
        return ok, fail

    def _wait_for_job_complete(self, job_name: str, *, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            job = self.batch_v1.read_namespaced_job_status(name=job_name, namespace=self.namespace)
            if job.status.succeeded:
                return
            if job.status.failed:
                raise RuntimeError(f"Probe Job '{job_name}' failed.")
            time.sleep(2)
        raise RuntimeError(f"Timed out waiting for probe Job '{job_name}'.")

    def _read_job_pod_logs(self, job_name: str) -> str:
        pods = self.core_v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=f"job-name={job_name}",
        )
        if not pods.items:
            return ""
        pod_name = pods.items[0].metadata.name
        return self.core_v1.read_namespaced_pod_log(name=pod_name, namespace=self.namespace)

    def _delete_job_quiet(self, job_name: str) -> None:
        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Foreground",
            )
        except ApiException as e:
            if e.status != 404:
                raise

    def _wait_for_deployment_ready(self, name: str, *, expected_replicas: int) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            dep = self.apps_v1.read_namespaced_deployment(name=name, namespace=self.namespace)
            ready = dep.status.ready_replicas or 0
            if ready >= expected_replicas:
                return
            time.sleep(5)
        raise RuntimeError(f"Timed out waiting for Deployment '{name}' to reach {expected_replicas} ready replicas.")

    def _delete_fault_objects_quiet(self) -> None:
        for kind, delete_fn, name in (
            ("deployment", self.apps_v1.delete_namespaced_deployment, DEPLOYMENT_NAME),
            ("service", self.core_v1.delete_namespaced_service, SERVICE_NAME),
        ):
            try:
                delete_fn(name=name, namespace=self.namespace)
            except ApiException as e:
                if e.status != 404:
                    raise
