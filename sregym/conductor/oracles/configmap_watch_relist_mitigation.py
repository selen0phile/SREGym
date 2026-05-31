"""Mitigation oracle for ``configmap_watch_relist_storm`` problems.

Validates that the agent stopped apiserver churn from a large, frequently
updated ConfigMap without deleting the cert-rotation workload (CronJob or trust
data). Accepts any one of four fix classes (split shards, delta patch, hash-gated
updates, or slow rotation schedule) once hard gates pass.
"""

from __future__ import annotations

import subprocess
import time

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from sregym.conductor.oracles.base import Oracle

_ROLLOUT_SETTLE_SECONDS = 60
_ROLLOUT_POLL_INTERVAL = 5
_KUBECTL_FAST_THRESHOLD_S = 4.0
_KUBECTL_PROBE_COUNT = 3
_RV_STABLE_WINDOW_S = 45
_RV_POLL_INTERVAL_S = 10

_FREQUENT_SCHEDULES = frozenset(
    {
        "* * * * *",
        "*/1 * * * *",
        "*/2 * * * *",
        "*/5 * * * *",
        "*/10 * * * *",
        "*/15 * * * *",
        "*/30 * * * *",
    }
)


class ConfigMapWatchRelistStormMitigationOracle(Oracle):
    importance = 1.0

    def __init__(self, problem):
        super().__init__(problem)
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()

    def evaluate(self) -> dict:
        print("== ConfigMap Watch Relist Storm Mitigation Evaluation ==")

        namespace = self.problem.namespace
        kubectl = self.problem.kubectl

        self._wait_for_rollouts(kubectl, namespace)

        cronjob_name = self.problem.CRONJOB_NAME
        consumer_name = self.problem.CONSUMER_DEPLOYMENT
        max_bytes = self.problem.MAX_CM_BYTES

        cj = self._get_cronjob(cronjob_name, namespace)
        workaround = self._workaround_reason(cj, namespace)
        if workaround:
            return self._fail(workaround)

        if self._storm_active(namespace, max_bytes, cj):
            return self._fail(
                "Apiserver churn is still active: a ConfigMap larger than "
                f"{max_bytes} bytes exists and the rotator is still updating it on "
                "a frequent schedule (or resourceVersion is still changing). "
                "Split the bundle into smaller ConfigMaps and remount the consumer, "
                "change the rotator to merge/hash-gated patches, or slow the schedule."
            )

        unhealthy = self._unhealthy_deployment(namespace, exclude={consumer_name})
        if unhealthy:
            return self._fail(
                f"Deployment '{unhealthy}' is under-replicated; mitigation hurt the application."
            )

        if not self._consumer_healthy(namespace, consumer_name):
            return self._fail(
                f"Deployment '{consumer_name}' is not ready or the trust bundle file "
                "is missing from its pods."
            )

        if not self._kubectl_responsive(namespace):
            return self._fail(
                f"kubectl get pods in '{namespace}' did not complete within "
                f"{_KUBECTL_FAST_THRESHOLD_S}s on {_KUBECTL_PROBE_COUNT} probes."
            )

        fix_class = self._detect_fix_class(cj, namespace, max_bytes, consumer_name)
        if fix_class is None:
            return self._fail(
                "Cluster is calm but no accepted mitigation was detected. Valid fixes: "
                "(A) split the large ConfigMap into shards under the size cap and remount "
                "trust-store-consumer with a projected volume; (B) rotator uses patch/merge "
                "instead of full replace; (C) rotator skips updates when the bundle hash "
                "is unchanged; (D) CronJob schedule is no longer sub-hourly. Deleting the "
                "CronJob or ConfigMap is not accepted."
            )

        print(f"✅ Mitigation accepted (class {fix_class}); apiserver responsive; trust workload preserved.")
        return {"success": True, "fix_class": fix_class}

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    def _workaround_reason(self, cj, namespace: str) -> str | None:
        if cj is None:
            return (
                f"CronJob '{self.problem.CRONJOB_NAME}' was deleted. Deletion removes "
                "cert rotation rather than fixing how the bundle is stored and updated."
            )
        if cj.spec.suspend:
            return (
                f"CronJob '{self.problem.CRONJOB_NAME}' is suspended. Suspend stops rotation "
                "without fixing object size or update pattern."
            )

        trust_cms = self._trust_configmaps(namespace)
        if not trust_cms:
            return (
                "No trust-bundle ConfigMap remains in the namespace. The bundle must "
                "still be delivered to workloads."
            )
        return None

    def _storm_active(self, namespace: str, max_bytes: int, cj) -> bool:
        large = self._large_configmaps(namespace, max_bytes)
        if not large:
            return False
        if cj is None or cj.spec.suspend:
            return False
        schedule = (cj.spec.schedule or "").strip()
        if schedule in _FREQUENT_SCHEDULES:
            return True
        for cm in large:
            if self._resource_version_changed(cm.metadata.name, namespace, _RV_STABLE_WINDOW_S):
                return True
        return False

    def _kubectl_responsive(self, namespace: str) -> bool:
        for _ in range(_KUBECTL_PROBE_COUNT):
            elapsed = self._kubectl_get_pods_seconds(namespace)
            if elapsed is None or elapsed > _KUBECTL_FAST_THRESHOLD_S:
                return False
        return True

    def _consumer_healthy(self, namespace: str, deployment_name: str) -> bool:
        try:
            dep = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
        except ApiException as e:
            if e.status == 404:
                return False
            raise
        desired = dep.spec.replicas or 1
        ready = dep.status.ready_replicas or 0
        if ready < desired:
            return False

        label_selector = ",".join(f"{k}={v}" for k, v in (dep.spec.selector.match_labels or {}).items())
        pods = self.core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        trust_paths = [self.problem.TRUST_MOUNT_PATH, "/etc/ssl/certs/ca-bundle-1.crt"]
        for pod in pods.items:
            if pod.status.phase != "Running":
                return False
            name = pod.metadata.name
            ok = False
            for path in trust_paths:
                cmd = f"kubectl exec -n {namespace} {name} -- test -f {path}"
                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=30)
                    ok = True
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    continue
            if not ok:
                return False
        return True

    # ------------------------------------------------------------------
    # Fix classes
    # ------------------------------------------------------------------
    def _detect_fix_class(self, cj, namespace: str, max_bytes: int, consumer_name: str) -> str | None:
        if self._class_a_split(namespace, max_bytes, consumer_name):
            return "A"
        rotator_cmd = self._rotator_command_text(cj)
        if rotator_cmd and self._class_b_delta(rotator_cmd):
            return "B"
        if rotator_cmd and self._class_c_idempotent(rotator_cmd):
            return "C"
        if cj and self._class_d_slow_schedule(cj.spec.schedule):
            return "D"
        return None

    def _class_a_split(self, namespace: str, max_bytes: int, consumer_name: str) -> bool:
        if self._large_configmaps(namespace, max_bytes):
            return False
        try:
            dep = self.apps_v1.read_namespaced_deployment(consumer_name, namespace)
        except ApiException:
            return False

        cm_sources = 0
        for vol in dep.spec.template.spec.volumes or []:
            if vol.config_map:
                cm_sources += 1
            projected = vol.projected
            if projected:
                for src in projected.sources or []:
                    if src.config_map:
                        cm_sources += 1
        return cm_sources >= 2

    @staticmethod
    def _class_b_delta(rotator_cmd: str) -> bool:
        lowered = rotator_cmd.lower()
        if "patch" not in lowered:
            return False
        return any(token in lowered for token in ("merge", "strategic", "json", "application/"))

    @staticmethod
    def _class_c_idempotent(rotator_cmd: str) -> bool:
        lowered = rotator_cmd.lower()
        return "md5" in lowered or "sha256" in lowered or "hash" in lowered and "skip" in lowered

    @staticmethod
    def _class_d_slow_schedule(schedule: str | None) -> bool:
        if not schedule:
            return False
        sched = schedule.strip()
        if sched in _FREQUENT_SCHEDULES:
            return False
        if sched.startswith("@hourly") or sched.startswith("@daily"):
            return True
        if sched.startswith("0 */") or sched.startswith("0 0 "):
            return True
        return sched not in _FREQUENT_SCHEDULES and "*" not in sched.split()[0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _wait_for_rollouts(self, kubectl, namespace: str) -> None:
        deadline = time.monotonic() + _ROLLOUT_SETTLE_SECONDS
        while time.monotonic() < deadline:
            deployments = kubectl.list_deployments(namespace)
            all_settled = True
            for dep in deployments.items:
                status = dep.status
                desired = dep.spec.replicas or 1
                if (
                    (status.updated_replicas or 0) < desired
                    or (status.ready_replicas or 0) < desired
                    or (status.unavailable_replicas or 0) > 0
                ):
                    all_settled = False
                    break
            if all_settled:
                return
            time.sleep(_ROLLOUT_POLL_INTERVAL)
        print("⚠️ Timed out waiting for deployments to settle; evaluating current state")

    def _get_cronjob(self, name: str, namespace: str):
        try:
            return self.batch_v1.read_namespaced_cron_job(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _trust_configmaps(self, namespace: str) -> list:
        prefix = self.problem.CM_SHARD_PREFIX
        legacy = self.problem.CM_NAME
        out = []
        for cm in self.core_v1.list_namespaced_config_map(namespace=namespace).items:
            name = cm.metadata.name
            if name == legacy or name.startswith(prefix):
                out.append(cm)
        return out

    def _large_configmaps(self, namespace: str, max_bytes: int) -> list:
        out = []
        for cm in self.core_v1.list_namespaced_config_map(namespace=namespace).items:
            if self._configmap_data_bytes(cm) > max_bytes:
                out.append(cm)
        return out

    @staticmethod
    def _configmap_data_bytes(cm) -> int:
        total = 0
        if cm.data:
            total += sum(len(v.encode("utf-8")) for v in cm.data.values())
        if cm.binary_data:
            total += sum(len(v.encode("utf-8")) for v in cm.binary_data.values())
        return total

    def _resource_version_changed(self, name: str, namespace: str, window_s: float) -> bool:
        rv1 = self._read_resource_version(name, namespace)
        if rv1 is None:
            return False
        deadline = time.monotonic() + window_s
        while time.monotonic() < deadline:
            time.sleep(_RV_POLL_INTERVAL_S)
            rv2 = self._read_resource_version(name, namespace)
            if rv2 is not None and rv2 != rv1:
                return True
        return False

    def _read_resource_version(self, name: str, namespace: str) -> str | None:
        try:
            cm = self.core_v1.read_namespaced_config_map(name=name, namespace=namespace)
            return cm.metadata.resource_version
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _kubectl_get_pods_seconds(self, namespace: str) -> float | None:
        cmd = f"kubectl get pods -n {namespace} --request-timeout=10s"
        start = time.monotonic()
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=15)
            return time.monotonic() - start
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

    def _rotator_command_text(self, cj) -> str:
        if cj is None:
            return ""
        containers = cj.spec.job_template.spec.template.spec.containers or []
        if not containers:
            return ""
        parts = []
        cmd = containers[0].command or []
        args = containers[0].args or []
        parts.extend(cmd)
        parts.extend(args)
        return " ".join(parts)

    def _unhealthy_deployment(self, namespace: str, exclude: set[str] | None = None):
        exclude = exclude or set()
        for dep in self.apps_v1.list_namespaced_deployment(namespace=namespace).items:
            if dep.metadata.name in exclude:
                continue
            desired = dep.spec.replicas or 1
            ready = dep.status.ready_replicas or 0
            if ready < desired:
                return dep.metadata.name
        return None

    @staticmethod
    def _fail(reason: str) -> dict:
        print(f"❌ {reason}")
        return {"success": False, "reason": reason}
