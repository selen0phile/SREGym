"""Problem: in-memory ServiceAccount token cache after projected token expiry.

Real-world failure class
------------------------
Since Kubernetes 1.22/1.24, pods receive **bound projected** ServiceAccount
tokens (KEP-1205) with a bounded lifetime (often ~1 hour). The kubelet refreshes
the token file on disk before expiry, but processes that read the JWT **once at
startup** keep using the stale in-memory bearer. API calls then return
``401 Unauthorized`` / ``token expired`` while pods stay ``Running`` and probes
pass. This is common in legacy operators, shell entrypoints, and Vault/K8s-auth
integrations — not in apps using client-go's file-backed token source.

References
~~~~~~~~~~
* Podostack #019 — "The expiry that breaks your CI on weekends":
  https://podostack.com/p/bound-sa-tokens-silent-expiry
* KEP-1205 (Bound Service Account Tokens):
  https://github.com/kubernetes/enhancements/tree/master/keps/sig-auth/1205-bound-service-account-tokens
* Kubernetes — configure projected SA tokens:
  https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/#service-account-token-volume-projection
* client-go ``NewCachedFileTokenSource``:
  https://github.com/kubernetes/client-go/blob/master/transport/token_source.go

Simulation in SREGym
--------------------
A platform helper ``secret-resolver`` is deployed into the Hotel Reservation
namespace. It mounts a projected ``serviceAccountToken`` (minimum 600s TTL per
apiserver validation) and caches the bearer at startup, then loops on
``GET /api/v1/namespaces/hotel-reservation/secrets``. Hotel Reservation
microservices stay healthy; the failure is auth to the Kubernetes API.

Accepted mitigations
~~~~~~~~~~~~~~~~~~~~
* **Rollout restart** the Deployment (picks up a fresh token).
* **Patch the entrypoint** to re-read the token file each loop (or use
  client-go's cached file token source).

Rejected mitigations
~~~~~~~~~~~~~~~~~~~~
* Delete ``secret-resolver`` only (removes the workload).
* Patch Mongo / app-layer auth (wrong layer).
"""

from __future__ import annotations

import time

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from sregym.conductor.oracles.llm_as_a_judge.llm_as_a_judge_oracle import LLMAsAJudgeOracle
from sregym.conductor.oracles.serviceaccount_token_stale_cache_mitigation import (
    ServiceAccountTokenStaleCacheMitigationOracle,
)
from sregym.conductor.problems.base import Problem
from sregym.observer.jaeger.jaeger import Jaeger
from sregym.service.apps.hotel_reservation import HotelReservation
from sregym.service.kubectl import KubeCtl
from sregym.utils.decorators import mark_fault_injected

# SREGym conductor creates these after every HR deploy (observe-stack wiring).
# They are unrelated to this fault but confuse agents that query Jaeger/OTel.
_JAEGER_ALIAS_SERVICES = ("jaeger", "jaeger-agent", "jaeger-collector", "jaeger-query")

DEPLOYMENT_NAME = "secret-resolver"
SA_NAME = "secret-resolver"
ROLE_NAME = f"{SA_NAME}-role"
BINDING_NAME = f"{SA_NAME}-binding"

TOKEN_MOUNT_DIR = "/var/run/secrets/tokens"
TOKEN_PATH = f"{TOKEN_MOUNT_DIR}/token"
CA_PATH = f"{TOKEN_MOUNT_DIR}/ca.crt"

# API server rejects expirationSeconds < 600 (10 minutes).
EXPIRATION_SECONDS = 600
INJECT_POLL_INTERVAL_S = 15
INJECT_MAX_WAIT_S = 780
INJECT_MIN_FAILURE_LINES = 2
RECOVERY_TIMEOUT_S = 300
RECOVERY_POLL_INTERVAL_S = 5

PLATFORM_LABELS = {
    "app.kubernetes.io/part-of": "platform",
    "app.kubernetes.io/component": "secret-resolver",
}

_FAULT_SCRIPT = f"""set -u
TP={TOKEN_PATH}
CA={CA_PATH}
NS={{{{NAMESPACE}}}}
TOKEN=$(cat "$TP")
while true; do
  CODE=$(curl -sS -o /dev/null -w '%{{http_code}}' --cacert "$CA" \\
    -H "Authorization: Bearer ${{TOKEN}}" \\
    "https://kubernetes.default.svc/api/v1/namespaces/${{NS}}/pods?limit=1" 2>/dev/null || echo 000)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) http=${{CODE}}"
  sleep 15
done
"""

_RECOVERED_SCRIPT = f"""set -u
TP={TOKEN_PATH}
CA={CA_PATH}
NS={{{{NAMESPACE}}}}
while true; do
  TOKEN=$(cat "$TP")
  CODE=$(curl -sS -o /dev/null -w '%{{http_code}}' --cacert "$CA" \\
    -H "Authorization: Bearer ${{TOKEN}}" \\
    "https://kubernetes.default.svc/api/v1/namespaces/${{NS}}/pods?limit=1" 2>/dev/null || echo 000)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) http=${{CODE}}"
  sleep 15
done
"""


class ServiceAccountTokenStaleCacheHotelReservation(Problem):
    """Stale in-memory SA token cache in the Hotel Reservation namespace."""

    DEPLOYMENT_NAME = DEPLOYMENT_NAME
    SA_NAME = SA_NAME
    TOKEN_PATH = TOKEN_PATH
    EXPIRATION_SECONDS = EXPIRATION_SECONDS

    def __init__(self, faulty_service: str = DEPLOYMENT_NAME):
        self.faulty_service = faulty_service
        self.app = HotelReservation()
        super().__init__(app=self.app, namespace=self.app.namespace)

        self.namespace = self.app.namespace
        self.kubectl = KubeCtl()
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.rbac_v1 = client.RbacAuthorizationV1Api()

        self.root_cause = self.build_structured_root_cause(
            component=f"Deployment/{DEPLOYMENT_NAME}",
            namespace=self.namespace,
            description=(
                f"A platform helper Deployment '{DEPLOYMENT_NAME}' in namespace "
                f"'{self.namespace}' uses a projected serviceAccountToken volume "
                f"(expirationSeconds={EXPIRATION_SECONDS}) and reads the JWT once at "
                "pod startup into memory. The kubelet refreshes the token file on disk "
                "before expiry, but the process never reopens the mount, so Kubernetes "
                "API calls return 401 Unauthorized / token expired while pods stay "
                "Running and readiness probes pass. Hotel Reservation microservices "
                "remain healthy; this is a bound-token lifecycle bug (KEP-1205), not "
                "Mongo auth_miss/revoke_auth, not RBAC drift, and not a missing mount. "
                "Legacy ServiceAccount token Secrets without exp are a red herring. "
                "Jaeger/otel-collector ExternalName Services in this namespace are "
                "normal SREGym observability plumbing, not this fault. "
                "Accepted mitigations: rollout restart the resolver; patch the entrypoint "
                "to re-read the token file each request or loop (pods API probe). Deleting only the "
                "resolver without fixing token handling, or patching unrelated Hotel "
                "Reservation auth/Mongo settings, are rejected."
            ),
        )

        self.diagnosis_oracle = LLMAsAJudgeOracle(problem=self, expected=self.root_cause)
        self.app.create_workload()
        self.mitigation_oracle = ServiceAccountTokenStaleCacheMitigationOracle(problem=self)

    @mark_fault_injected
    def inject_fault(self):
        print("== Fault Injection ==")
        self._delete_fault_objects_quiet()
        self._ensure_resolver_rbac()
        self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace,
            body=self._build_resolver_deployment(cache_at_startup=True),
        )
        self._wait_for_deployment_ready(DEPLOYMENT_NAME)
        self._wait_for_stale_token_symptom()
        self._strip_sregym_jaeger_aliases()
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    @mark_fault_injected
    def recover_fault(self):
        print("== Fault Recovery ==")
        self._delete_fault_objects_quiet()
        self._wait_for_fault_objects_absent()
        self._restore_sregym_jaeger_aliases()
        self._ensure_resolver_rbac()
        self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace,
            body=self._build_resolver_deployment(cache_at_startup=False),
        )
        self._wait_for_deployment_ready(DEPLOYMENT_NAME)
        self._wait_for_api_success_lines(min_success=4, timeout_s=90)
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    def _wait_for_api_success_lines(self, *, min_success: int, timeout_s: int) -> None:
        print(f"Waiting for {min_success} consecutive http=200 lines from {DEPLOYMENT_NAME}...")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            logs = self._read_resolver_logs(tail_lines=30)
            recent = [line for line in logs.splitlines() if "http=" in line][-min_success:]
            if len(recent) >= min_success and all("http=200" in line for line in recent):
                return
            time.sleep(INJECT_POLL_INTERVAL_S)
        print("⚠️ Timed out waiting for sustained http=200 responses after recovery.")

    def _resolver_script(self, *, cache_at_startup: bool) -> str:
        template = _FAULT_SCRIPT if cache_at_startup else _RECOVERED_SCRIPT
        return template.replace("{{NAMESPACE}}", self.namespace)

    def _build_resolver_deployment(self, *, cache_at_startup: bool) -> dict:
        script = self._resolver_script(cache_at_startup=cache_at_startup)
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
                "replicas": 1,
                "selector": {"matchLabels": {"app.kubernetes.io/name": DEPLOYMENT_NAME}},
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": DEPLOYMENT_NAME}},
                    "spec": {
                        "serviceAccountName": SA_NAME,
                        "automountServiceAccountToken": False,
                        "containers": [
                            {
                                "name": "resolver",
                                "image": "curlimages/curl:8.11.1",
                                "command": ["sh", "-c", script],
                                "volumeMounts": [
                                    {
                                        "name": "sa-tokens",
                                        "mountPath": TOKEN_MOUNT_DIR,
                                        "readOnly": True,
                                    }
                                ],
                                "readinessProbe": {
                                    "exec": {"command": ["test", "-f", TOKEN_PATH]},
                                    "periodSeconds": 5,
                                    "initialDelaySeconds": 3,
                                },
                                "resources": {
                                    "requests": {"cpu": "10m", "memory": "32Mi"},
                                    "limits": {"cpu": "100m", "memory": "64Mi"},
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "sa-tokens",
                                "projected": {
                                    "defaultMode": 420,
                                    "sources": [
                                        {
                                            "serviceAccountToken": {
                                                "path": "token",
                                                "expirationSeconds": EXPIRATION_SECONDS,
                                            }
                                        },
                                        {
                                            "configMap": {
                                                "name": "kube-root-ca.crt",
                                                "items": [
                                                    {"key": "ca.crt", "path": "ca.crt"},
                                                ],
                                            }
                                        },
                                    ],
                                },
                            }
                        ],
                    },
                },
            },
        }

    def _strip_sregym_jaeger_aliases(self) -> None:
        """Remove observe-stack Jaeger DNS aliases so agents focus on secret-resolver."""
        for name in _JAEGER_ALIAS_SERVICES:
            try:
                self.core_v1.delete_namespaced_service(name=name, namespace=self.namespace)
            except ApiException as e:
                if e.status != 404:
                    print(f"  warning: could not delete Service {name}: {e.reason}")

    def _restore_sregym_jaeger_aliases(self) -> None:
        Jaeger().create_external_name_service(self.namespace)

    def _wait_for_stale_token_symptom(self) -> None:
        print(
            "Waiting for projected token expiry while the resolver keeps its "
            "startup-cached bearer token (expect http=401 after token TTL)..."
        )
        deadline = time.monotonic() + INJECT_MAX_WAIT_S
        min_age_s = EXPIRATION_SECONDS + 30
        while time.monotonic() < deadline:
            pod_age = self._resolver_pod_age_seconds()
            if pod_age is not None and pod_age < min_age_s:
                time.sleep(INJECT_POLL_INTERVAL_S)
                continue
            logs = self._read_resolver_logs(tail_lines=40)
            recent = [line for line in logs.splitlines() if "http=" in line][-6:]
            failures = sum(1 for line in recent if "http=401" in line)
            successes = sum(1 for line in recent if "http=200" in line)
            if failures >= INJECT_MIN_FAILURE_LINES and failures > successes:
                print(f"  Observed {failures} recent http=401 lines (pod age ~{pod_age:.0f}s).")
                return
            time.sleep(INJECT_POLL_INTERVAL_S)
        print("⚠️ Timed out waiting for http=401 lines; continuing anyway.")

    def _resolver_pod_age_seconds(self) -> float | None:
        try:
            dep = self.apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, self.namespace)
        except ApiException:
            return None
        selector = ",".join(f"{k}={v}" for k, v in (dep.spec.selector.match_labels or {}).items())
        pods = self.core_v1.list_namespaced_pod(namespace=self.namespace, label_selector=selector).items
        if not pods or not pods[0].status.start_time:
            return None
        return time.time() - pods[0].status.start_time.timestamp()

    def _read_resolver_logs(self, *, tail_lines: int) -> str:
        try:
            dep = self.apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, self.namespace)
        except ApiException:
            return ""
        selector = ",".join(f"{k}={v}" for k, v in (dep.spec.selector.match_labels or {}).items())
        pods = self.core_v1.list_namespaced_pod(namespace=self.namespace, label_selector=selector).items
        if not pods:
            return ""
        try:
            return self.core_v1.read_namespaced_pod_log(
                name=pods[0].metadata.name,
                namespace=self.namespace,
                tail_lines=tail_lines,
            )
        except ApiException:
            return ""

    def _ensure_resolver_rbac(self):
        try:
            self.core_v1.read_namespaced_service_account(SA_NAME, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise
            self.core_v1.create_namespaced_service_account(
                namespace=self.namespace,
                body=client.V1ServiceAccount(
                    metadata=client.V1ObjectMeta(
                        name=SA_NAME,
                        namespace=self.namespace,
                        labels=PLATFORM_LABELS,
                    ),
                ),
            )

        try:
            self.rbac_v1.read_namespaced_role(ROLE_NAME, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise
            self.rbac_v1.create_namespaced_role(
                namespace=self.namespace,
                body=client.V1Role(
                    metadata=client.V1ObjectMeta(name=ROLE_NAME, namespace=self.namespace),
                    rules=[
                        client.V1PolicyRule(
                            api_groups=[""],
                            resources=["pods"],
                            verbs=["get", "list"],
                        ),
                    ],
                ),
            )

        try:
            self.rbac_v1.read_namespaced_role_binding(BINDING_NAME, self.namespace)
        except ApiException as e:
            if e.status != 404:
                raise
            self.rbac_v1.create_namespaced_role_binding(
                namespace=self.namespace,
                body=client.V1RoleBinding(
                    metadata=client.V1ObjectMeta(name=BINDING_NAME, namespace=self.namespace),
                    role_ref=client.V1RoleRef(
                        api_group="rbac.authorization.k8s.io",
                        kind="Role",
                        name=ROLE_NAME,
                    ),
                    subjects=[
                        client.RbacV1Subject(
                            kind="ServiceAccount",
                            name=SA_NAME,
                            namespace=self.namespace,
                        )
                    ],
                ),
            )

    def _wait_for_deployment_ready(self, name: str) -> None:
        deadline = time.monotonic() + RECOVERY_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                dep = self.apps_v1.read_namespaced_deployment(name, self.namespace)
            except ApiException:
                time.sleep(RECOVERY_POLL_INTERVAL_S)
                continue
            desired = dep.spec.replicas or 1
            ready = dep.status.ready_replicas or 0
            if ready >= desired and desired > 0:
                return
            time.sleep(RECOVERY_POLL_INTERVAL_S)
        print(f"⚠️ Timed out waiting for Deployment '{name}' to become ready")

    def _wait_for_fault_objects_absent(self):
        deadline = time.monotonic() + RECOVERY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._get_deployment() is None:
                return
            time.sleep(RECOVERY_POLL_INTERVAL_S)
        print("⚠️ Timed out waiting for fault Deployment to be deleted before recovery recreate.")

    def _get_deployment(self):
        try:
            return self.apps_v1.read_namespaced_deployment(DEPLOYMENT_NAME, self.namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            raise

    def _delete_fault_objects_quiet(self):
        for kind, delete_fn, name in [
            ("Deployment", self.apps_v1.delete_namespaced_deployment, DEPLOYMENT_NAME),
            ("RoleBinding", self.rbac_v1.delete_namespaced_role_binding, BINDING_NAME),
            ("Role", self.rbac_v1.delete_namespaced_role, ROLE_NAME),
            ("ServiceAccount", self.core_v1.delete_namespaced_service_account, SA_NAME),
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
