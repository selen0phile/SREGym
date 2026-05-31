"""Problem: large ConfigMap churn causes apiserver watch/relist pressure.

Simulates cert-bundle rotation that fully replaces a ~960KB ConfigMap every
minute. Hotel Reservation microservices stay healthy; the failure mode is
control-plane sluggishness and ConfigMap resourceVersion churn.
"""

from __future__ import annotations

import subprocess
import time

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from sregym.conductor.oracles.configmap_watch_relist_mitigation import (
    ConfigMapWatchRelistStormMitigationOracle,
)
from sregym.conductor.oracles.llm_as_a_judge.llm_as_a_judge_oracle import LLMAsAJudgeOracle
from sregym.conductor.problems.base import Problem
from sregym.service.apps.hotel_reservation import HotelReservation
from sregym.service.kubectl import KubeCtl
from sregym.utils.decorators import mark_fault_injected

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CM_NAME = "ca-trust-bundle"
CM_SHARD_PREFIX = "ca-trust-bundle-"
CM_KEY = "ca-bundle.crt"
CRONJOB_NAME = "ca-bundle-rotator"
SUSTAIN_DEPLOYMENT = "ca-bundle-rotator-sustain"
CONSUMER_DEPLOYMENT = "trust-store-consumer"
ROTATION_STATUS_CM = "ca-trust-bundle-rotation-status"
ROTATOR_SA = "ca-bundle-rotator"
SUSTAIN_INTERVAL_S = 30
TRUST_MOUNT_PATH = "/etc/ssl/certs/ca-bundle.crt"

# Stay under the Kubernetes ConfigMap size limit (1 MiB including metadata).
BUNDLE_TARGET_BYTES = 983_040
SHARD_TARGET_BYTES = 95_000
MAX_CM_BYTES = 120_000

ROTATOR_SCHEDULE = "*/1 * * * *"
RECOVERED_SCHEDULE = "0 */6 * * *"
# 1.32 is often missing on kind nodes; latest pulls reliably and includes kubectl+openssl.
ROTATOR_IMAGE = "bitnami/kubectl:latest"

INJECT_TIMEOUT_S = 200
INJECT_POLL_INTERVAL_S = 5
INJECT_MIN_SUCCESSFUL_JOBS = 2
INJECT_MIN_RV_CHANGES = 2
# kind is smaller than prod; still expect human-noticeable lag when churn is active.
INJECT_KUBECTL_SLOW_THRESHOLD_S = 0.8
CONSUMER_REPLICAS = 15
INJECT_BURST_SECONDS = 90
INJECT_BURST_INTERVAL_S = 15

RECOVERY_TIMEOUT_S = 300
RECOVERY_POLL_INTERVAL_S = 5

COMPLIANCE_LABELS = {
    "app.kubernetes.io/part-of": "compliance",
    "app.kubernetes.io/component": "trust-store",
}


class ConfigMapWatchRelistStormHotelReservation(Problem):
    """Large ConfigMap rotation storm in the Hotel Reservation namespace."""

    CM_NAME = CM_NAME
    CM_SHARD_PREFIX = CM_SHARD_PREFIX
    CRONJOB_NAME = CRONJOB_NAME
    CONSUMER_DEPLOYMENT = CONSUMER_DEPLOYMENT
    MAX_CM_BYTES = MAX_CM_BYTES
    TRUST_MOUNT_PATH = TRUST_MOUNT_PATH

    def __init__(self, faulty_service: str = CRONJOB_NAME):
        self.faulty_service = faulty_service
        self.app = HotelReservation()
        super().__init__(app=self.app, namespace=self.app.namespace)

        self.namespace = self.app.namespace
        self.kubectl = KubeCtl()
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()
        self.rbac_v1 = client.RbacAuthorizationV1Api()

        self.root_cause = self.build_structured_root_cause(
            component=f"CronJob/{CRONJOB_NAME}",
            namespace=self.namespace,
            description=(
                f"A compliance CronJob '{CRONJOB_NAME}' in namespace '{self.namespace}' "
                f"fully replaces a large ConfigMap '{CM_NAME}' (roughly {BUNDLE_TARGET_BYTES // 1000}KB "
                f"of PEM-like trust material in key '{CM_KEY}') on schedule '{ROTATOR_SCHEDULE}'. "
                f"The bundle is mounted by Deployment '{CONSUMER_DEPLOYMENT}' for in-cluster TLS "
                "trust. Hotel Reservation microservices remain Running; the failure is "
                "control-plane stress from large-object watch relist pressure while "
                f"metadata.resourceVersion on '{CM_NAME}' churns (see ConfigMap "
                f"'{ROTATION_STATUS_CM}' and Warning events on the bundle). On small "
                "clusters kubectl get pods may stay fast; diagnose via rotation_count, "
                "resourceVersion changes, and a ~960KB bundle size (time get -o yaml). "
                "The mechanism is a "
                "large etcd object updated frequently (full replace), which advances "
                "revision/compaction and forces watch relists — not wrong application "
                "config keys (distinct from configmap_drift). "
                "Accepted mitigations: split the bundle into multiple smaller ConfigMaps "
                "under the size cap and remount with a projected volume; change the rotator "
                "to strategic/merge patches or skip writes when the hash is unchanged; "
                "or reduce rotation frequency to sub-hourly or slower. "
                "Deleting the CronJob or ConfigMap, suspending rotation only, or "
                "restarting unrelated Hotel Reservation Deployments without fixing size/"
                "update pattern are rejected."
            ),
        )

        self.diagnosis_oracle = LLMAsAJudgeOracle(problem=self, expected=self.root_cause)
        self.app.create_workload()
        self.mitigation_oracle = ConfigMapWatchRelistStormMitigationOracle(problem=self)

    # ------------------------------------------------------------------
    @mark_fault_injected
    def inject_fault(self):
        print("== Fault Injection ==")
        self._delete_fault_objects_quiet()

        self._ensure_rotator_rbac()
        self.core_v1.create_namespaced_config_map(
            namespace=self.namespace, body=self._build_bundle_configmap(revision="0")
        )
        self.core_v1.create_namespaced_config_map(
            namespace=self.namespace, body=self._build_rotation_status_configmap("0", "0")
        )
        self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace, body=self._build_consumer_deployment(mount_shards=False)
        )
        self.batch_v1.create_namespaced_cron_job(
            namespace=self.namespace, body=self._build_rotator_cronjob(full_replace=True)
        )
        self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace, body=self._build_sustain_deployment()
        )
        self._trigger_immediate_rotator_job()
        self._create_inject_burst_job()
        self._wait_for_inject_storm()
        self._print_diagnosis_hints()
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    @mark_fault_injected
    def recover_fault(self):
        print("== Fault Recovery ==")
        self._delete_fault_objects_quiet()
        self._wait_for_fault_objects_absent()
        self._ensure_rotator_rbac()

        payload = self._bundle_payload("recovered")
        half = len(payload) // 2
        self.core_v1.create_namespaced_config_map(
            namespace=self.namespace,
            body=self._build_shard_configmap(f"{CM_SHARD_PREFIX}1", payload[:half]),
        )
        self.core_v1.create_namespaced_config_map(
            namespace=self.namespace,
            body=self._build_shard_configmap(f"{CM_SHARD_PREFIX}2", payload[half:]),
        )
        self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace, body=self._build_consumer_deployment(mount_shards=True)
        )
        self.batch_v1.create_namespaced_cron_job(
            namespace=self.namespace, body=self._build_rotator_cronjob(full_replace=False)
        )

        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    @staticmethod
    def _bundle_payload(revision: str) -> str:
        block = (
            "-----BEGIN CERTIFICATE-----\n"
            + ("MIIFakeRootCA" + ("A" * 180) + "\n")
            + "-----END CERTIFICATE-----\n"
        )
        body = (block * 1500) + f"\n# revision={revision}\n"
        if len(body) < BUNDLE_TARGET_BYTES:
            body = body * (BUNDLE_TARGET_BYTES // len(body) + 1)
        return body[:BUNDLE_TARGET_BYTES]

    def _build_bundle_configmap(self, revision: str) -> dict:
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": CM_NAME,
                "namespace": self.namespace,
                "labels": {
                    **COMPLIANCE_LABELS,
                    "app.kubernetes.io/name": CM_NAME,
                },
                "annotations": {"rotator.kubernetes.io/last-rotation": revision},
            },
            "data": {CM_KEY: self._bundle_payload(revision)},
        }

    def _build_shard_configmap(self, name: str, data: str) -> dict:
        if len(data) > SHARD_TARGET_BYTES:
            data = data[:SHARD_TARGET_BYTES]
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": {
                    **COMPLIANCE_LABELS,
                    "app.kubernetes.io/name": name,
                },
            },
            "data": {CM_KEY: data},
        }

    def _build_consumer_deployment(self, *, mount_shards: bool) -> dict:
        if mount_shards:
            volume = {
                "name": "trust-bundle",
                "projected": {
                    "sources": [
                        {
                            "configMap": {
                                "name": f"{CM_SHARD_PREFIX}1",
                                "items": [{"key": CM_KEY, "path": "ca-bundle-1.crt"}],
                            }
                        },
                        {
                            "configMap": {
                                "name": f"{CM_SHARD_PREFIX}2",
                                "items": [{"key": CM_KEY, "path": "ca-bundle-2.crt"}],
                            }
                        },
                    ]
                },
            }
            mounts = [{"name": "trust-bundle", "mountPath": "/etc/ssl/certs", "readOnly": True}]
            readiness_cmd = ["test", "-f", "/etc/ssl/certs/ca-bundle-1.crt"]
        else:
            volume = {"name": "trust-bundle", "configMap": {"name": CM_NAME}}
            # Mount the whole ConfigMap so ca-bundle.crt is at TRUST_MOUNT_PATH (not a subPath file trap).
            mounts = [{"name": "trust-bundle", "mountPath": "/etc/ssl/certs", "readOnly": True}]
            readiness_cmd = ["test", "-f", TRUST_MOUNT_PATH]
            keepalive_path = TRUST_MOUNT_PATH

        if mount_shards:
            keepalive_path = "/etc/ssl/certs/ca-bundle-1.crt"

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": CONSUMER_DEPLOYMENT,
                "namespace": self.namespace,
                "labels": {
                    **COMPLIANCE_LABELS,
                    "app.kubernetes.io/name": CONSUMER_DEPLOYMENT,
                },
            },
            "spec": {
                "replicas": CONSUMER_REPLICAS if not mount_shards else 2,
                "selector": {"matchLabels": {"app.kubernetes.io/name": CONSUMER_DEPLOYMENT}},
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": CONSUMER_DEPLOYMENT}},
                    "spec": {
                        "containers": [
                            {
                                "name": "consumer",
                                "image": "busybox:1.36",
                                "command": [
                                    "sh",
                                    "-c",
                                    f"while test -f {keepalive_path}; do sleep 3600; done; exit 1",
                                ],
                                "volumeMounts": mounts,
                                "readinessProbe": {
                                    "exec": {"command": readiness_cmd},
                                    "periodSeconds": 5,
                                },
                                "resources": {
                                    "requests": {"cpu": "10m", "memory": "16Mi"},
                                    "limits": {"cpu": "50m", "memory": "32Mi"},
                                },
                            }
                        ],
                        "volumes": [volume],
                    },
                },
            },
        }

    @staticmethod
    def _rotation_telemetry_shell() -> str:
        status_cm = ROTATION_STATUS_CM
        return f"""
STATUS_CM={status_cm}
RV=$(kubectl get cm "$CM" -n "$NS" -o jsonpath='{{.metadata.resourceVersion}}')
COUNT=$(kubectl get cm "$STATUS_CM" -n "$NS" -o jsonpath='{{.data.rotation_count}}' 2>/dev/null || echo 0)
COUNT=$((COUNT + 1))
kubectl create configmap "$STATUS_CM" -n "$NS" \\
  --from-literal=rotation_count="$COUNT" \\
  --from-literal=last_resource_version="$RV" \\
  --from-literal=last_rotation="$REV" \\
  --from-literal=bundle_bytes="$TARGET" \\
  --from-literal=diagnosis_hint="ConfigMap watch relist storm: large ca-trust-bundle full-replaced frequently; HR pods stay healthy" \\
  --dry-run=client -o yaml | kubectl replace -f -
kubectl create event -n "$NS" \\
  --reporting-component=ca-bundle-rotator \\
  --reason=TrustBundleReplaced \\
  --message="Full replace of $CM (~$TARGET bytes); resourceVersion=$RV; rotation #$COUNT" \\
  --type=Warning \\
  --regarding-configmap="$CM" 2>/dev/null || true
"""

    @classmethod
    def _full_replace_rotator_script(
        cls,
        namespace: str,
        *,
        loop_seconds: int | None = None,
        repeat_forever: bool = False,
        sleep_seconds: int | None = None,
    ) -> str:
        loop_header = ""
        loop_footer = ""
        interval = sleep_seconds or INJECT_BURST_INTERVAL_S
        if repeat_forever:
            loop_header = "while true; do\n"
            loop_footer = f"  sleep {interval}\ndone\n"
        elif loop_seconds is not None:
            loop_header = f'END=$(( $(date +%s) + {loop_seconds} ))\nwhile [ "$(date +%s)" -lt "$END" ]; do\n'
            loop_footer = f"  sleep {interval}\ndone\n"

        return f"""set -e
{loop_header}NS={namespace}
CM={CM_NAME}
KEY={CM_KEY}
TARGET={BUNDLE_TARGET_BYTES}
: > /tmp/bundle.crt
while [ "$(wc -c < /tmp/bundle.crt | tr -d ' ')" -lt "$TARGET" ]; do
  echo "-----BEGIN CERTIFICATE-----" >> /tmp/bundle.crt
  openssl rand -base64 2048 >> /tmp/bundle.crt 2>/dev/null || head -c 2048 /dev/urandom | base64 >> /tmp/bundle.crt
  echo "-----END CERTIFICATE-----" >> /tmp/bundle.crt
done
head -c "$TARGET" /tmp/bundle.crt > /tmp/bundle.trim.crt
REV=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Use replace (not apply): apply embeds the large object in last-applied-configuration
# and exceeds the 256KiB annotation limit.
kubectl create configmap "$CM" --from-file="$KEY"=/tmp/bundle.trim.crt -n "$NS" \\
  --dry-run=client -o yaml | kubectl replace -f -
kubectl annotate configmap "$CM" -n "$NS" rotator.kubernetes.io/last-rotation="$REV" --overwrite
{cls._rotation_telemetry_shell()}
{loop_footer}"""

    def _build_rotation_status_configmap(self, rotation_count: str, resource_version: str) -> dict:
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": ROTATION_STATUS_CM,
                "namespace": self.namespace,
                "labels": {**COMPLIANCE_LABELS, "app.kubernetes.io/name": ROTATION_STATUS_CM},
            },
            "data": {
                "rotation_count": rotation_count,
                "last_resource_version": resource_version,
                "last_rotation": "",
                "bundle_bytes": str(BUNDLE_TARGET_BYTES),
                "diagnosis_hint": (
                    "Increments each time ca-bundle-rotator fully replaces ca-trust-bundle. "
                    "Compare resourceVersion over time; HR pods should stay Running."
                ),
            },
        }

    def _build_sustain_deployment(self) -> dict:
        script = self._full_replace_rotator_script(
            self.namespace, repeat_forever=True, sleep_seconds=SUSTAIN_INTERVAL_S
        )
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": SUSTAIN_DEPLOYMENT,
                "namespace": self.namespace,
                "labels": {
                    **COMPLIANCE_LABELS,
                    "app.kubernetes.io/name": SUSTAIN_DEPLOYMENT,
                },
            },
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app.kubernetes.io/name": SUSTAIN_DEPLOYMENT}},
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": SUSTAIN_DEPLOYMENT}},
                    "spec": {
                        "serviceAccountName": ROTATOR_SA,
                        "containers": [
                            {
                                "name": "rotator",
                                "image": ROTATOR_IMAGE,
                                "command": ["sh", "-c", script],
                                "resources": {
                                    "requests": {"cpu": "50m", "memory": "64Mi"},
                                    "limits": {"cpu": "300m", "memory": "384Mi"},
                                },
                            }
                        ],
                    },
                },
            },
        }

    def _build_rotator_cronjob(self, *, full_replace: bool) -> dict:
        ns = self.namespace
        if full_replace:
            script = self._full_replace_rotator_script(ns)
        else:
            script = f"""set -e
NS={ns}
REV=$(date -u +%Y-%m-%dT%H:%M:%SZ)
kubectl patch configmap {CM_SHARD_PREFIX}1 -n "$NS" --type merge \\
  -p "{{\\"metadata\\":{{\\"annotations\\":{{\\"rotator.kubernetes.io/last-rotation\\":\\"$REV\\"}}}}}}"
"""

        return {
            "apiVersion": "batch/v1",
            "kind": "CronJob",
            "metadata": {
                "name": CRONJOB_NAME,
                "namespace": self.namespace,
                "labels": {
                    **COMPLIANCE_LABELS,
                    "app.kubernetes.io/name": CRONJOB_NAME,
                },
            },
            "spec": {
                "schedule": ROTATOR_SCHEDULE if full_replace else RECOVERED_SCHEDULE,
                "concurrencyPolicy": "Forbid",
                "successfulJobsHistoryLimit": 1,
                "failedJobsHistoryLimit": 1,
                "jobTemplate": {
                    "spec": {
                        "template": {
                            "spec": {
                                "serviceAccountName": ROTATOR_SA,
                                "restartPolicy": "OnFailure",
                                "containers": [
                                    {
                                        "name": "rotator",
                                        "image": ROTATOR_IMAGE,
                                        "command": ["sh", "-c", script],
                                        "resources": {
                                            "requests": {"cpu": "50m", "memory": "64Mi"},
                                            "limits": {"cpu": "200m", "memory": "256Mi"},
                                        },
                                    }
                                ],
                            }
                        }
                    }
                },
            },
        }

    def _ensure_rotator_rbac(self):
        try:
            self.core_v1.read_namespaced_service_account(ROTATOR_SA, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise
            self.core_v1.create_namespaced_service_account(
                namespace=self.namespace,
                body=client.V1ServiceAccount(
                    metadata=client.V1ObjectMeta(name=ROTATOR_SA, namespace=self.namespace),
                ),
            )

        role_name = f"{ROTATOR_SA}-role"
        try:
            self.rbac_v1.read_namespaced_role(role_name, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise
            self.rbac_v1.create_namespaced_role(
                namespace=self.namespace,
                body=client.V1Role(
                    metadata=client.V1ObjectMeta(name=role_name, namespace=self.namespace),
                    rules=[
                        client.V1PolicyRule(
                            api_groups=[""],
                            resources=["configmaps"],
                            verbs=["get", "list", "watch", "create", "update", "patch", "delete"],
                        ),
                        client.V1PolicyRule(
                            api_groups=[""],
                            resources=["events"],
                            verbs=["create"],
                        ),
                    ],
                ),
            )

        binding_name = f"{ROTATOR_SA}-binding"
        try:
            self.rbac_v1.read_namespaced_role_binding(binding_name, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise
            self.rbac_v1.create_namespaced_role_binding(
                namespace=self.namespace,
                body=client.V1RoleBinding(
                    metadata=client.V1ObjectMeta(name=binding_name, namespace=self.namespace),
                    role_ref=client.V1RoleRef(
                        api_group="rbac.authorization.k8s.io",
                        kind="Role",
                        name=role_name,
                    ),
                    subjects=[
                        client.RbacV1Subject(
                            kind="ServiceAccount",
                            name=ROTATOR_SA,
                            namespace=self.namespace,
                        )
                    ],
                ),
            )

    # ------------------------------------------------------------------
    # Inject verification (storm must be observable)
    # ------------------------------------------------------------------
    def _create_inject_burst_job(self):
        """Extra replace loop during inject so RV churn is obvious before the agent runs."""
        script = self._full_replace_rotator_script(self.namespace, loop_seconds=INJECT_BURST_SECONDS)
        try:
            self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=client.V1Job(
                    api_version="batch/v1",
                    kind="Job",
                    metadata=client.V1ObjectMeta(
                        name=f"{CRONJOB_NAME}-inject-burst",
                        namespace=self.namespace,
                        labels={**COMPLIANCE_LABELS, "app.kubernetes.io/name": CRONJOB_NAME},
                    ),
                    spec=client.V1JobSpec(
                        template=client.V1PodTemplateSpec(
                            spec=client.V1PodSpec(
                                service_account_name=ROTATOR_SA,
                                restart_policy="Never",
                                containers=[
                                    client.V1Container(
                                        name="rotator",
                                        image=ROTATOR_IMAGE,
                                        command=["sh", "-c", script],
                                    )
                                ],
                            )
                        ),
                        backoff_limit=0,
                        ttl_seconds_after_finished=600,
                    ),
                ),
            )
            print(
                f"  Started inject burst Job ({INJECT_BURST_SECONDS}s, "
                f"every {INJECT_BURST_INTERVAL_S}s replace) to amplify apiserver churn."
            )
        except ApiException as e:
            if e.status != 409:
                print(f"  warning: could not create inject burst Job: {e.reason}")

    def _trigger_immediate_rotator_job(self):
        """Run one rotator Job now so humans/agents do not wait only on the CronJob tick."""
        try:
            self.batch_v1.create_namespaced_job(
                namespace=self.namespace,
                body=client.V1Job(
                    api_version="batch/v1",
                    kind="Job",
                    metadata=client.V1ObjectMeta(
                        generate_name=f"{CRONJOB_NAME}-bootstrap-",
                        namespace=self.namespace,
                        labels={**COMPLIANCE_LABELS, "app.kubernetes.io/name": CRONJOB_NAME},
                    ),
                    spec=self.batch_v1.read_namespaced_cron_job(
                        CRONJOB_NAME, self.namespace
                    ).spec.job_template.spec,
                ),
            )
            print("  Triggered bootstrap rotator Job (in addition to CronJob schedule).")
        except ApiException as e:
            print(f"  warning: could not create bootstrap rotator Job: {e.reason}")

    def _wait_for_inject_storm(self):
        print(
            f"Waiting for rotator Jobs to rewrite '{CM_NAME}' "
            f"(≥{INJECT_MIN_SUCCESSFUL_JOBS} successes, ≥{INJECT_MIN_RV_CHANGES} RV changes)..."
        )
        baseline_rv = self._read_cm_rv(CM_NAME)
        seen_rvs = {baseline_rv} if baseline_rv else set()
        deadline = time.monotonic() + INJECT_TIMEOUT_S
        slow_kubectl_seen = False

        while time.monotonic() < deadline:
            jobs_ok = self._count_successful_rotator_jobs() >= INJECT_MIN_SUCCESSFUL_JOBS
            rv = self._read_cm_rv(CM_NAME)
            if rv:
                seen_rvs.add(rv)
            rv_ok = len(seen_rvs) >= INJECT_MIN_RV_CHANGES + (1 if baseline_rv else 0)

            elapsed = self._kubectl_get_pods_seconds()
            if elapsed is not None and elapsed >= INJECT_KUBECTL_SLOW_THRESHOLD_S:
                slow_kubectl_seen = True

            if jobs_ok and rv_ok:
                print(
                    f"  Rotator active: {self._count_successful_rotator_jobs()} successful Job(s), "
                    f"{len(seen_rvs)} distinct resourceVersion(s) on '{CM_NAME}'."
                )
                if elapsed is not None:
                    print(f"  Latest kubectl get pods: {elapsed:.2f}s")
                if slow_kubectl_seen:
                    print("  Apiserver/kubectl sluggishness observed during inject.")
                else:
                    print(
                        "  kubectl get pods may stay fast on kind; use rotation status + large GET:\n"
                        f"    sgkc get cm {ROTATION_STATUS_CM} -n {self.namespace} -o yaml\n"
                        f"    sgkc get events -n {self.namespace} --field-selector involvedObject.name={CM_NAME}\n"
                        f"    time sgkc get cm {CM_NAME} -n {self.namespace} -o yaml"
                    )
                return

            time.sleep(INJECT_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Inject timed out after {INJECT_TIMEOUT_S}s: rotator did not churn '{CM_NAME}' "
            f"(jobs={self._count_successful_rotator_jobs()}, rvs={len(seen_rvs)}). "
            f"Check CronJob '{CRONJOB_NAME}' pods and image '{ROTATOR_IMAGE}'."
        )

    def _count_successful_rotator_jobs(self) -> int:
        jobs = self.batch_v1.list_namespaced_job(namespace=self.namespace)
        count = 0
        for job in jobs.items:
            name = job.metadata.name or ""
            if not name.startswith(CRONJOB_NAME):
                continue
            if (job.status.succeeded or 0) >= 1:
                count += 1
        return count

    def _print_diagnosis_hints(self):
        ns = self.namespace
        print("== Diagnosis hints (storm signals) ==")
        print(f"  sgkc get cm {ROTATION_STATUS_CM} -n {ns} -o yaml")
        print(f"  watch -n5 'sgkc get cm {ROTATION_STATUS_CM} -n {ns} -o json | jq -r .data.rotation_count'")
        print(f"  sgkc get events -n {ns} --field-selector involvedObject.name={CM_NAME}")
        print(f"  time sgkc get cm {CM_NAME} -n {ns} -o yaml   # transfers ~{BUNDLE_TARGET_BYTES} bytes")

    def _kubectl_get_pods_seconds(self) -> float | None:
        cmd = f"kubectl get pods -n {self.namespace} --request-timeout=30s"
        start = time.monotonic()
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=45)
            return time.monotonic() - start
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------
    def _wait_for_fault_objects_absent(self):
        deadline = time.monotonic() + RECOVERY_TIMEOUT_S
        targets = [
            ("CronJob", self._get_cronjob),
            ("Deployment", self._get_deployment),
            (f"Deployment/{SUSTAIN_DEPLOYMENT}", lambda: self._get_deployment_by_name(SUSTAIN_DEPLOYMENT)),
            (f"ConfigMap/{CM_NAME}", lambda: self._read_cm_rv(CM_NAME)),
            (f"ConfigMap/{ROTATION_STATUS_CM}", lambda: self._read_cm_rv(ROTATION_STATUS_CM)),
            (f"ConfigMap/{CM_SHARD_PREFIX}1", lambda: self._read_cm_rv(f"{CM_SHARD_PREFIX}1")),
            (f"ConfigMap/{CM_SHARD_PREFIX}2", lambda: self._read_cm_rv(f"{CM_SHARD_PREFIX}2")),
        ]
        while time.monotonic() < deadline:
            if all(check() is None for _, check in targets):
                return
            time.sleep(RECOVERY_POLL_INTERVAL_S)
        print("⚠️ Timed out waiting for fault objects to be deleted before recovery recreate.")

    def _get_cronjob(self):
        try:
            return self.batch_v1.read_namespaced_cron_job(CRONJOB_NAME, self.namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _get_deployment(self):
        return self._get_deployment_by_name(CONSUMER_DEPLOYMENT)

    def _get_deployment_by_name(self, name: str):
        try:
            return self.apps_v1.read_namespaced_deployment(name, self.namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _read_cm_rv(self, name: str) -> str | None:
        try:
            cm = self.core_v1.read_namespaced_config_map(name, namespace=self.namespace)
            return cm.metadata.resource_version
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _delete_fault_objects_quiet(self):
        for kind, delete_fn, name in [
            ("CronJob", self.batch_v1.delete_namespaced_cron_job, CRONJOB_NAME),
            ("Deployment", self.apps_v1.delete_namespaced_deployment, SUSTAIN_DEPLOYMENT),
            ("Deployment", self.apps_v1.delete_namespaced_deployment, CONSUMER_DEPLOYMENT),
            ("Job", self.batch_v1.delete_namespaced_job, f"{CRONJOB_NAME}-inject-burst"),
            ("ConfigMap", self.core_v1.delete_namespaced_config_map, ROTATION_STATUS_CM),
            ("ConfigMap", self.core_v1.delete_namespaced_config_map, CM_NAME),
            ("ConfigMap", self.core_v1.delete_namespaced_config_map, f"{CM_SHARD_PREFIX}1"),
            ("ConfigMap", self.core_v1.delete_namespaced_config_map, f"{CM_SHARD_PREFIX}2"),
        ]:
            try:
                delete_fn(
                    name=name,
                    namespace=self.namespace,
                    propagation_policy="Foreground",
                )
            except ApiException as e:
                if e.status != 404:
                    print(f"  warning: could not delete {kind} {name}: {e.reason}")
