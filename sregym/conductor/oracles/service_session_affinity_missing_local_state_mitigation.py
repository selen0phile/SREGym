"""Mitigation oracle for ``service_session_affinity_missing_local_state``.

Hotel Reservation microservices stay Running while a session-store Deployment
returns intermittent 404s because requests hit different replicas without
ClientIP session affinity (or a single replica).
"""

from __future__ import annotations

import time

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from sregym.conductor.oracles.base import Oracle

_ROLLOUT_SETTLE_SECONDS = 60
_ROLLOUT_POLL_INTERVAL = 5
_PROBE_ATTEMPTS = 20
_MIN_SUCCESS_RATE_FIXED = 0.85
_MIN_FAILURE_RATE_FAULT = 0.15


class ServiceSessionAffinityMissingLocalStateMitigationOracle(Oracle):
    importance = 1.0

    def __init__(self, problem):
        super().__init__(problem)
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

    def evaluate(self) -> dict:
        print("== Session Affinity Missing Local State Mitigation Evaluation ==")

        namespace = self.problem.namespace
        deployment_name = self.problem.DEPLOYMENT_NAME
        service_name = self.problem.SERVICE_NAME

        self._wait_for_rollouts(namespace)

        dep = self._get_deployment(deployment_name, namespace)
        svc = self._get_service(service_name, namespace)

        workaround = self._workaround_reason(dep, svc)
        if workaround:
            return self._fail(workaround)

        if dep is None or svc is None:
            return self._fail(
                f"Deployment '{deployment_name}' and Service '{service_name}' must remain in "
                f"namespace '{namespace}'. Deleting the session workload is not an accepted fix."
            )

        if not self._deployment_ready(dep):
            return self._fail(
                f"Deployment '{deployment_name}' is not fully ready. "
                "Restore the session-store workload while fixing routing or replica count."
            )

        unhealthy = self._unhealthy_deployment(namespace, exclude={deployment_name})
        if unhealthy:
            return self._fail(
                f"Deployment '{unhealthy}' is under-replicated; mitigation hurt the application."
            )

        replicas = dep.spec.replicas or 0
        client_ip = self._has_client_ip_affinity(svc)
        single_replica = replicas == 1

        ok, fail = self.problem.run_session_probe(attempts=_PROBE_ATTEMPTS)
        total = ok + fail
        if total == 0:
            return self._fail(
                "Session probe could not reach the session-store Service. "
                "Check that the Deployment and Service exist and port 8080 is reachable."
            )

        success_rate = ok / total

        if not client_ip and replicas > 1:
            if fail >= 2 and success_rate < (1.0 - _MIN_FAILURE_RATE_FAULT):
                return self._fail(
                    f"Service '{service_name}' still uses sessionAffinity=None with {replicas} replicas. "
                    f"Session reads fail intermittently ({ok}/{total} GET successes). "
                    "Patch the Service to sessionAffinity: ClientIP or scale session-store to one replica."
                )
            if ok == total:
                return self._fail(
                    f"Service '{service_name}' still has no ClientIP affinity and {replicas} replicas, "
                    "but the probe did not reproduce split-brain failures yet. "
                    "Confirm sessionAffinity and replica count, then re-test GET /session/{{id}}."
                )

        if (client_ip or single_replica) and success_rate >= _MIN_SUCCESS_RATE_FIXED:
            fix_class = "A" if client_ip else "B"
            label = (
                "ClientIP session affinity on the Service"
                if client_ip
                else "single-replica session-store Deployment"
            )
            print(f"✅ Mitigation accepted (class {fix_class}): {label} and session reads succeed.")
            return {"success": True, "fix_class": fix_class, "success_rate": success_rate}

        if client_ip or single_replica:
            return self._fail(
                f"Session routing looks fixed (ClientIP={client_ip}, replicas={replicas}) but "
                f"only {ok}/{total} probe GETs succeeded. Verify the session-store pods are healthy."
            )

        return self._fail(
            f"Session-store still misconfigured: replicas={replicas}, "
            f"sessionAffinity={svc.spec.session_affinity or 'None'}. "
            "Enable ClientIP affinity or scale to one replica."
        )

    def _workaround_reason(self, dep, svc) -> str | None:
        if dep is None and svc is None:
            return (
                "The session-store Deployment and Service were removed. "
                "That deletes the workload instead of fixing sticky routing or shared state."
            )
        if dep is None:
            return (
                f"Deployment '{self.problem.DEPLOYMENT_NAME}' was deleted. "
                "Keep the session-store Deployment and fix Service affinity or replica count."
            )
        if svc is None:
            return (
                f"Service '{self.problem.SERVICE_NAME}' was deleted. "
                "Patch sessionAffinity or scale replicas instead of removing the Service."
            )
        return None

    def _has_client_ip_affinity(self, svc) -> bool:
        return (svc.spec.session_affinity or "").lower() == "clientip"

    def _deployment_ready(self, dep) -> bool:
        desired = dep.spec.replicas or 0
        ready = dep.status.ready_replicas or 0
        return desired > 0 and ready >= desired

    def _get_deployment(self, name: str, namespace: str):
        try:
            return self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _get_service(self, name: str, namespace: str):
        try:
            return self.core_v1.read_namespaced_service(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _unhealthy_deployment(self, namespace: str, *, exclude: set[str]) -> str | None:
        for dep in self.apps_v1.list_namespaced_deployment(namespace=namespace).items:
            name = dep.metadata.name
            if name in exclude:
                continue
            desired = dep.spec.replicas or 0
            if desired == 0:
                continue
            ready = dep.status.ready_replicas or 0
            if ready < desired:
                return name
        return None

    def _wait_for_rollouts(self, namespace: str) -> None:
        kubectl = self.problem.kubectl
        deadline = time.monotonic() + _ROLLOUT_SETTLE_SECONDS
        while time.monotonic() < deadline:
            out = kubectl.exec_command(
                f"kubectl get deploy -n {namespace} -o jsonpath="
                "'{{range .items}}{{.metadata.name}} {{.status.readyReplicas}} "
                "{{.spec.replicas}}{{\"\\n\"}}{{end}}'"
            )
            pending = False
            for line in (out or "").splitlines():
                parts = line.split()
                if len(parts) != 3:
                    continue
                _, ready, desired = parts
                if ready != desired:
                    pending = True
                    break
            if not pending:
                return
            time.sleep(_ROLLOUT_POLL_INTERVAL)

    def _fail(self, reason: str) -> dict:
        print(f"❌ {reason}")
        return {"success": False, "reason": reason}
