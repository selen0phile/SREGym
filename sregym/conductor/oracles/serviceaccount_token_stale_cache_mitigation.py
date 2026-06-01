"""Mitigation oracle for the ``serviceaccount_token_stale_cache`` problem.

The default ``MitigationOracle`` is insufficient here: the Hotel Reservation
microservices stay ``Running`` while a platform helper logs ``401`` responses
from the Kubernetes API. This oracle inspects the ``secret-resolver`` Deployment
spec and pod logs for sustained API success after the agent fixes token reload
behavior (or rollout-restarts the workload).
"""

from __future__ import annotations

import re
import time

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from sregym.conductor.oracles.base import Oracle

_ROLLOUT_SETTLE_SECONDS = 60
_ROLLOUT_POLL_INTERVAL = 5
_LOG_TAIL_LINES = 60
_RECENT_LINE_WINDOW = 12
_MIN_SUCCESS_LINES = 4
_MIN_FAILURE_LINES = 2


class ServiceAccountTokenStaleCacheMitigationOracle(Oracle):
    importance = 1.0

    def __init__(self, problem):
        super().__init__(problem)
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

    def evaluate(self) -> dict:
        print("== ServiceAccount Token Stale Cache Mitigation Evaluation ==")

        namespace = self.problem.namespace
        deployment_name = self.problem.DEPLOYMENT_NAME

        self._wait_for_rollouts(namespace)

        dep = self._get_deployment(deployment_name, namespace)
        workaround = self._workaround_reason(dep)
        if workaround:
            return self._fail(workaround)

        if not self._resolver_ready(namespace, deployment_name):
            return self._fail(
                f"Deployment '{deployment_name}' is missing or not ready. "
                "The secret-resolver workload must stay running while token handling is fixed."
            )

        unhealthy = self._unhealthy_deployment(namespace, exclude={deployment_name})
        if unhealthy:
            return self._fail(f"Deployment '{unhealthy}' is under-replicated; mitigation hurt the application.")

        cmd = self._resolver_command_text(dep)
        logs = self._resolver_logs(namespace, deployment_name, tail_lines=_LOG_TAIL_LINES)
        recent = [line for line in logs.splitlines() if "http=" in line][-_RECENT_LINE_WINDOW:]
        successes = sum(1 for line in recent if "http=200" in line)
        failures = sum(1 for line in recent if "http=401" in line)

        reloads = self._reloads_token_each_request(cmd)
        startup_cache = self._uses_startup_token_cache(cmd)

        if reloads and successes >= 2:
            print("✅ Mitigation accepted (class B): entrypoint re-reads projected token each loop.")
            return {"success": True, "fix_class": "B"}

        if successes >= _MIN_SUCCESS_LINES and failures == 0:
            fix_class = "A" if startup_cache else "B"
            print(f"✅ Mitigation accepted (class {fix_class}): Kubernetes API calls succeed from '{deployment_name}'.")
            return {"success": True, "fix_class": fix_class}

        if failures >= _MIN_FAILURE_LINES and successes <= 1:
            return self._fail(
                f"'{deployment_name}' is still getting Unauthorized responses from the API "
                f"(recent log lines: {failures} failures, {successes} successes). "
                "Rollout-restart the Deployment or patch the container to re-read "
                f"{self.problem.TOKEN_PATH} on each request instead of caching the JWT at startup."
            )

        if startup_cache and failures >= 1:
            return self._fail(
                "The resolver still caches the ServiceAccount token at startup. "
                "Kubelet rotates the projected token file, but the in-memory bearer is stale. "
                "Restart the Deployment or move TOKEN=$(cat ...) inside the request loop."
            )

        return self._fail(
            "Could not confirm API access from the resolver. Check pod logs for http=200 vs "
            "http=401 and fix token reload behavior."
        )

    # ------------------------------------------------------------------
    def _workaround_reason(self, dep) -> str | None:
        if dep is None:
            return (
                f"Deployment '{self.problem.DEPLOYMENT_NAME}' was deleted. Removing the "
                "platform helper is not an accepted mitigation."
            )
        return None

    def _resolver_ready(self, namespace: str, deployment_name: str) -> bool:
        try:
            dep = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
        except ApiException as e:
            if e.status == 404:
                return False
            raise
        desired = dep.spec.replicas or 1
        ready = dep.status.ready_replicas or 0
        return ready >= desired and desired > 0

    def _resolver_command_text(self, dep) -> str:
        containers = dep.spec.template.spec.containers or []
        if not containers:
            return ""
        parts: list[str] = []
        parts.extend(containers[0].command or [])
        parts.extend(containers[0].args or [])
        return " ".join(parts)

    def _resolver_logs(self, namespace: str, deployment_name: str, *, tail_lines: int) -> str:
        try:
            dep = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
        except ApiException:
            return ""
        selector = ",".join(f"{k}={v}" for k, v in (dep.spec.selector.match_labels or {}).items())
        pods = self.core_v1.list_namespaced_pod(namespace=namespace, label_selector=selector).items
        if not pods:
            return ""
        pod_name = pods[0].metadata.name
        try:
            return self.core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
            )
        except ApiException:
            return ""

    @staticmethod
    def _uses_startup_token_cache(cmd: str) -> bool:
        if not cmd:
            return False
        while_idx = cmd.find("while true")
        if while_idx < 0:
            return bool(re.search(r"TOKEN=\$\(cat", cmd))
        before_loop = cmd[:while_idx]
        return bool(re.search(r"TOKEN=\$\(cat", before_loop))

    @staticmethod
    def _reloads_token_each_request(cmd: str) -> bool:
        if not cmd:
            return False
        while_idx = cmd.find("while true")
        if while_idx < 0:
            return False
        loop_body = cmd[while_idx:]
        return bool(re.search(r"TOKEN=\$\(cat", loop_body))

    def _wait_for_rollouts(self, namespace: str) -> None:
        deadline = time.monotonic() + _ROLLOUT_SETTLE_SECONDS
        while time.monotonic() < deadline:
            deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace).items
            all_settled = True
            for dep in deployments:
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

    def _get_deployment(self, name: str, namespace: str):
        try:
            return self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

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
