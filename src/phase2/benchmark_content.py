from __future__ import annotations

import pandas as pd

from .queries import DEFAULT_QUERY_CONFIG


TOPIC_QUERY_BANK = {
    "authentication_and_identity": [
        {
            "query_text": "What is the safest way for human administrators to authenticate to a production Kubernetes cluster?",
            "reference_answer": "The safest default is to use a federated identity provider such as OIDC or managed cloud IAM with MFA and individual accounts. Avoid shared kubeconfigs or shared static credentials because they weaken auditability and make revocation harder.",
        },
        {
            "query_text": "Why should all Kubernetes API traffic use TLS?",
            "reference_answer": "TLS protects tokens, certificates, and cluster data while it is moving between clients and the API server. Without TLS, an attacker on the network can intercept or modify sensitive traffic.",
        },
        {
            "query_text": "When should I prefer short-lived tokens over long-lived static credentials for Kubernetes access?",
            "reference_answer": "Short-lived credentials are usually safer because they expire quickly and are easier to rotate or revoke. Long-lived static credentials increase the impact of theft and are harder to manage safely at scale.",
        },
        {
            "query_text": "How should I reduce risk from copied kubeconfig files on administrator laptops?",
            "reference_answer": "Use central identity, MFA, least-privilege RBAC, and short-lived credentials instead of permanent broad-access kubeconfigs. Lost or copied kubeconfigs should be revocable quickly, and endpoints should use disk encryption and secure storage.",
        },
        {
            "query_text": "How should an external automation service authenticate to the Kubernetes API?",
            "reference_answer": "Use a dedicated non-human identity such as workload identity federation or a narrowly scoped service account with short-lived audience-scoped tokens. Do not reuse an administrator kubeconfig or a personal account for automation.",
        },
        {
            "query_text": "What controls make bootstrap authentication safer during cluster or node provisioning?",
            "reference_answer": "Bootstrap tokens should be short-lived, narrowly scoped, and removed after provisioning. Their use should also be audited and limited to the expected network path and bootstrap workflow.",
        },
        {
            "query_text": "How should I rotate certificates and authentication secrets without breaking the cluster?",
            "reference_answer": "Rotate them in a planned staged workflow with overlapping validity and tested rollback steps. The process should cover certificates, signing keys, and tokens, and it should be automated where possible to reduce outages.",
        },
        {
            "query_text": "How do I prevent unintended or anonymous access to the Kubernetes API server?",
            "reference_answer": "Disable anonymous authentication if it is not needed, restrict API server network exposure, and require proper authentication and authorization for all requests. The API should be reachable only from the expected management paths.",
        },
        {
            "query_text": "How should I separate identities for humans, workloads, and nodes in Kubernetes?",
            "reference_answer": "Use different identity mechanisms and different permissions for each class of actor. Humans should use federated login, workloads should use service accounts or workload identity, and nodes should use their own tightly scoped identities with node-specific controls.",
        },
        {
            "query_text": "What is the safest way for an external service to validate Kubernetes service account tokens?",
            "reference_answer": "Use the TokenReview API or validate through OIDC discovery and JWKS with issuer, audience, and expiry checks. It is not enough to trust the token contents without verifying signature and claims properly.",
        },
    ],
    "rbac_and_service_accounts": [
        {
            "query_text": "What does least-privilege RBAC mean in Kubernetes?",
            "reference_answer": "Least privilege means granting only the exact verbs, resources, and namespaces that a user or workload needs. The default posture should be minimal access, with additional permissions added only when clearly justified.",
        },
        {
            "query_text": "Why is using the default service account risky for application pods?",
            "reference_answer": "Pods may inherit the default service account automatically even when they do not need API access. If that service account has more rights than expected, a pod compromise can spread further than intended.",
        },
        {
            "query_text": "What is the difference between a Role and a ClusterRole in Kubernetes RBAC?",
            "reference_answer": "A Role is scoped to one namespace, while a ClusterRole is cluster-scoped or reusable across namespaces. This distinction matters because cluster-wide permissions usually have a much larger blast radius.",
        },
        {
            "query_text": "How should I grant a workload access to only one namespace?",
            "reference_answer": "Create a dedicated service account, define a Role in the target namespace, and bind that Role to the service account with a RoleBinding. This keeps permissions local to the namespace instead of widening them unnecessarily.",
        },
        {
            "query_text": "Why can list or watch access on Secrets be as sensitive as get access?",
            "reference_answer": "Because list and watch can expose many secret values at once, not just one object. In practice, broad read-style access to Secrets often gives an attacker nearly the same sensitive data as direct get access.",
        },
        {
            "query_text": "How do I reduce privilege escalation through workload creation rights?",
            "reference_answer": "Be very careful with create rights on Pods, Deployments, Jobs, and similar workload resources. A user who can create workloads may be able to mount Secrets, use powerful service accounts, or request dangerous host access indirectly.",
        },
        {
            "query_text": "How should I audit dangerous RBAC permissions first?",
            "reference_answer": "Review cluster-admin style access first, then check for impersonate, bind, escalate, Secrets access, nodes/proxy access, and workload creation rights. Those permissions are common starting points for privilege escalation and lateral movement.",
        },
        {
            "query_text": "When should I use separate service accounts for different applications?",
            "reference_answer": "Use separate service accounts whenever the applications have different trust levels or different permission needs. This reduces blast radius and makes reviews, rotation, and auditing easier.",
        },
        {
            "query_text": "How do the bind and escalate verbs increase RBAC risk?",
            "reference_answer": "They let a subject grant roles or create stronger roles beyond its ordinary permissions, which can bypass normal RBAC guardrails. These verbs should be treated as highly sensitive and granted rarely.",
        },
        {
            "query_text": "How should I control cross-namespace access for an operator?",
            "reference_answer": "Give the operator a dedicated identity and explicitly bind only the namespaces and resources it truly needs. Avoid broad cluster-admin grants when a narrower cross-namespace design is possible.",
        },
    ],
    "pod_security_and_admission_control": [
        {
            "query_text": "What does Pod Security Admission do in Kubernetes?",
            "reference_answer": "Pod Security Admission checks pod specifications against the Pod Security Standards at admission time. It helps block risky settings such as privileged mode, dangerous host access, and other weak isolation choices.",
        },
        {
            "query_text": "Why should most application namespaces aim for the Restricted Pod Security policy?",
            "reference_answer": "Restricted blocks many common privilege escalation paths and reflects modern pod-hardening expectations. For ordinary application workloads it is usually the best default target if compatibility allows it.",
        },
        {
            "query_text": "Why are privileged containers dangerous in a hardened cluster?",
            "reference_answer": "Privileged containers can access host-level capabilities and often break the normal isolation boundary between pod and node. That makes container compromise far more serious.",
        },
        {
            "query_text": "How should I roll out Pod Security Admission without breaking existing workloads?",
            "reference_answer": "Start with audit and warn modes, inspect the violations, fix the manifests, and only then move to enforce mode. That phased rollout reduces surprise failures while still improving security steadily.",
        },
        {
            "query_text": "Which securityContext fields should I review first when hardening pods?",
            "reference_answer": "Start with privileged, allowPrivilegeEscalation, runAsNonRoot, capabilities, host namespaces, hostPath volumes, and seccomp or similar runtime controls. Those fields often determine whether a pod can bypass basic isolation.",
        },
        {
            "query_text": "How can admission control help beyond Pod Security Admission?",
            "reference_answer": "Custom admission policies can enforce organization-specific rules such as approved registries, required labels, image signatures, resource policies, or restricted annotations. That lets you cover controls that Pod Security Admission does not enforce by itself.",
        },
        {
            "query_text": "When should I use namespace exemptions from Pod Security Admission?",
            "reference_answer": "Only when there is a clear operational reason, such as essential system components that cannot yet meet the policy. Exemptions should be narrow, documented, and reviewed regularly.",
        },
        {
            "query_text": "How do RuntimeClass and sandboxed runtimes fit into pod hardening?",
            "reference_answer": "They add another isolation layer by running selected workloads with a stronger runtime boundary such as gVisor or Kata. They do not replace policy controls, but they can reduce impact for higher-risk workloads.",
        },
        {
            "query_text": "How should I combine Pod Security Admission with custom policy engines such as Kyverno or Gatekeeper?",
            "reference_answer": "Use Pod Security Admission for broad baseline platform hardening, then use a custom policy engine for rules that are specific to your organization. This layered approach keeps the baseline simple while allowing targeted extra controls.",
        },
        {
            "query_text": "What is a safe way to harden legacy workloads that still fail the Restricted policy?",
            "reference_answer": "Inventory the failing controls, fix the easiest ones first, isolate the legacy workloads, and keep pressure on remediation with clear milestones. Use audit and warn modes during the transition, not permanent broad exceptions.",
        },
    ],
    "secrets_handling": [
        {
            "query_text": "Why should I avoid storing plaintext secrets in Git for Kubernetes deployments?",
            "reference_answer": "Git spreads secrets widely through clones, backups, and history, and removal later is difficult. Plaintext secrets in source control often become long-term exposure points.",
        },
        {
            "query_text": "What is the main limitation of a Kubernetes Secret by itself?",
            "reference_answer": "A Secret is an API object with basic access control and encoding, but it is not a full secret-management system on its own. Its safety still depends heavily on RBAC, encryption, and how the secret is delivered and rotated.",
        },
        {
            "query_text": "Why should I avoid mounting more secrets than a pod actually needs?",
            "reference_answer": "Every extra secret increases blast radius if the pod is compromised. The pod should receive only the specific secret material it needs for that workload.",
        },
        {
            "query_text": "How should I deliver database credentials to an application pod safely?",
            "reference_answer": "Use a dedicated secret or an external secret manager integration, scope it to the right namespace and identity, and mount only the values the pod needs. The design should also include a rotation plan.",
        },
        {
            "query_text": "When should I prefer short-lived credentials over long-lived secret values in Kubernetes?",
            "reference_answer": "Use short-lived credentials whenever the system supports them because they reduce the theft window and make rotation safer. Long-lived static secrets should be the exception, not the default.",
        },
        {
            "query_text": "How should I restrict who can read secrets in a Kubernetes cluster?",
            "reference_answer": "Use tight RBAC, separate identities, and namespace boundaries, and avoid broad get, list, or watch access on Secret objects. Secret readers should be explicit and rare.",
        },
        {
            "query_text": "What is the risk of storing service account tokens or cloud credentials as generic static Secrets?",
            "reference_answer": "Static copies bypass stronger lifetime and audience controls and are harder to rotate and audit. They can turn short-lived or context-bound credentials into long-lived stolen artifacts.",
        },
        {
            "query_text": "How should I rotate secrets without causing outages for workloads?",
            "reference_answer": "Use overlapping validity, staged rollout, health checks, and an application design that can reload or restart cleanly. Rotation should be tested regularly so it is not a one-time emergency procedure.",
        },
        {
            "query_text": "What is a safe pattern for syncing an external secret manager into Kubernetes?",
            "reference_answer": "Keep the sync scope narrow, map values only into the required namespaces and identities, and audit both sync activity and secret consumers. Avoid over-privileged secret-sync controllers that can read everything in the cluster.",
        },
        {
            "query_text": "How do I reduce secret exposure in logs, debug output, and troubleshooting data?",
            "reference_answer": "Redact sensitive fields, avoid printing secret values, restrict who can access logs and support bundles, and review observability pipelines for accidental secret leakage. Debugging should never require exposing the secret itself.",
        },
    ],
    "network_policy_and_traffic_isolation": [
        {
            "query_text": "What does a Kubernetes NetworkPolicy control?",
            "reference_answer": "A NetworkPolicy controls which pods or endpoints can communicate with a pod for selected directions and ports, depending on CNI support. It is one of the main tools for reducing lateral movement inside the cluster.",
        },
        {
            "query_text": "Why is default-allow pod networking risky in a multi-service cluster?",
            "reference_answer": "If most workloads can talk to everything, a compromised pod can scan and move laterally much more easily. Default-allow networking creates unnecessary internal exposure.",
        },
        {
            "query_text": "What is a safe first step for network isolation in Kubernetes?",
            "reference_answer": "A common safe start is to apply default-deny for ingress and then allow only the required application flows. This makes connectivity intentional instead of accidental.",
        },
        {
            "query_text": "How should I allow only a frontend service to talk to a backend database?",
            "reference_answer": "Use labels and a default-deny policy, then create a narrow allow rule from the frontend pods or namespace to the database port only. Avoid broader allow rules that include unrelated workloads.",
        },
        {
            "query_text": "Why should I check CNI support before relying on NetworkPolicy?",
            "reference_answer": "Not every CNI enforces the same policy features, so unsupported policies can create a false sense of safety. The cluster team should confirm what is actually enforced in the chosen network stack.",
        },
        {
            "query_text": "How do egress policies help with Kubernetes security?",
            "reference_answer": "They limit outbound connections so workloads can talk only to approved destinations. This reduces exfiltration paths and makes command-and-control traffic harder after compromise.",
        },
        {
            "query_text": "How should I handle DNS when I introduce strict network policies?",
            "reference_answer": "You usually need an explicit rule that still allows pods to reach the cluster DNS service. Otherwise normal application name resolution can fail and cause confusion during rollout.",
        },
        {
            "query_text": "Why are hostNetwork pods a concern in a hardened cluster?",
            "reference_answer": "hostNetwork shares the node network namespace, which can bypass normal pod-level network isolation. It should be reserved for carefully justified cases.",
        },
        {
            "query_text": "How should I isolate tenant namespaces from each other on the network layer?",
            "reference_answer": "Use namespace-scoped default-deny rules, explicit ingress and egress allowlists, and narrowly defined shared-service exceptions. Tenant isolation should be intentional, not left to default connectivity.",
        },
        {
            "query_text": "What extra controls are useful when NetworkPolicy alone is not enough?",
            "reference_answer": "Additional controls can include service mesh policy, egress gateways, firewalls, admission rules, and stronger identity and runtime controls. NetworkPolicy is important, but it is not the only layer.",
        },
    ],
    "etcd_and_data_protection": [
        {
            "query_text": "Why is etcd one of the most sensitive components in a Kubernetes cluster?",
            "reference_answer": "etcd stores core cluster state, including Secrets and control-plane data, so strong access to etcd often means strong access to the whole cluster. Direct etcd compromise can become cluster compromise quickly.",
        },
        {
            "query_text": "Why should confidential Kubernetes data be encrypted at rest?",
            "reference_answer": "Encryption at rest reduces exposure if disks, snapshots, or storage backups are accessed by an attacker. It adds protection beyond ordinary API-layer access control.",
        },
        {
            "query_text": "What is the minimum safe network posture for etcd?",
            "reference_answer": "etcd should be private, authenticated, and encrypted with TLS, and it should not be reachable from workloads. Only the intended control-plane components and administrators should reach it.",
        },
        {
            "query_text": "How should I protect etcd backups in Kubernetes?",
            "reference_answer": "Treat them like high-sensitivity secrets: encrypt them, restrict access tightly, separate storage, and test restore procedures. A backup can be just as dangerous as the live data if exposed.",
        },
        {
            "query_text": "What is the risk of leaving old encryption keys active forever in a cluster?",
            "reference_answer": "Compromised old keys may still decrypt stored data or complicate confident recovery. Key rotation needs an end state where obsolete keys are removed after safe migration.",
        },
        {
            "query_text": "How should I rotate encryption keys or providers for Kubernetes secrets at rest?",
            "reference_answer": "Add the new key or provider, re-encrypt the data, validate that reads and writes still work, and then retire the old key after the migration is confirmed. Rotation should be planned, not improvised.",
        },
        {
            "query_text": "Why do API server and etcd certificate settings matter for cluster security?",
            "reference_answer": "Weak, expired, or mis-scoped certificates can break trust or expose control-plane traffic to interception. Certificate management is part of protecting the control-plane data path.",
        },
        {
            "query_text": "How should I reduce exposure from control-plane snapshots and stored state copies?",
            "reference_answer": "Limit who can create and access snapshots, encrypt them, set retention rules, and log administrative access. Every copy of cluster state should be treated as sensitive data.",
        },
        {
            "query_text": "What is a safe disaster-recovery practice for encrypted Kubernetes data?",
            "reference_answer": "Test restoring both the data and the required keys or configuration together in an isolated environment. Recovery plans that ignore the key material are incomplete and risky.",
        },
        {
            "query_text": "How do I stop workloads from reaching control-plane data paths indirectly?",
            "reference_answer": "Use network segmentation, harden nodes, and prevent dangerous pod features such as unnecessary hostPath or hostNetwork access. The goal is to stop workloads from bridging into control-plane systems.",
        },
    ],
    "image_and_supply_chain_security": [
        {
            "query_text": "Why should I avoid using latest tags for production container images?",
            "reference_answer": "latest is mutable, so the same deployment can pull different code at different times. That weakens traceability, change control, and incident response.",
        },
        {
            "query_text": "What is the advantage of pinning container images by digest?",
            "reference_answer": "A digest pins the exact image artifact, which makes deployments reproducible and auditable. It also reduces the risk of silently pulling changed content under the same tag.",
        },
        {
            "query_text": "Why should I scan images before deployment to Kubernetes?",
            "reference_answer": "Pre-deployment scanning helps identify known vulnerabilities and risky packages before the workload reaches the cluster. It is an important early filter in supply-chain defense.",
        },
        {
            "query_text": "How should I control which registries workloads can pull images from?",
            "reference_answer": "Use admission policy, CI/CD controls, and registry allowlists so workloads can pull only from approved image sources. That reduces the chance of running unreviewed or malicious images.",
        },
        {
            "query_text": "What is the risk of running container images as root by default?",
            "reference_answer": "Root inside the container increases the impact of a compromise and can make escape attempts more dangerous. Hardening should prefer non-root execution wherever possible.",
        },
        {
            "query_text": "How should I handle base-image updates when new vulnerabilities are disclosed?",
            "reference_answer": "Rebuild from maintained base images, retest, and redeploy rather than patching running containers manually. A repeatable rebuild pipeline is safer than one-off fixes inside containers.",
        },
        {
            "query_text": "Why are signed images or provenance records useful in Kubernetes supply-chain security?",
            "reference_answer": "They help verify who built the image and whether the artifact was tampered with between build and deployment. This strengthens trust in the software delivery path.",
        },
        {
            "query_text": "How should I stop developers from deploying unreviewed images directly to the cluster?",
            "reference_answer": "Require CI/CD promotion, enforce registry policy, and use admission checks for approved or signed images. Direct deployment paths should not bypass the release controls.",
        },
        {
            "query_text": "What is a safe response when a critical image vulnerability is announced?",
            "reference_answer": "Find which workloads use the affected image, prioritize privileged and internet-facing workloads, rebuild and redeploy patched images, and monitor for exploitation signs during the response window.",
        },
        {
            "query_text": "How do SBOMs help in Kubernetes supply-chain security?",
            "reference_answer": "SBOMs make impact analysis faster because they show which components and versions are inside each image. That helps security teams identify affected workloads quickly when a dependency issue appears.",
        },
    ],
    "logging_auditing_and_detection": [
        {
            "query_text": "What does Kubernetes audit logging record?",
            "reference_answer": "Audit logging records API requests and the actions taken by the control plane in response. It is one of the most valuable sources for cluster security investigation.",
        },
        {
            "query_text": "Why should security logs be exported off the cluster?",
            "reference_answer": "An attacker who compromises the cluster may tamper with or delete local logs. Off-cluster storage improves integrity and makes incident investigation more reliable.",
        },
        {
            "query_text": "What should I log first if I am designing basic Kubernetes security monitoring?",
            "reference_answer": "Start with authentication, authorization, sensitive resource access, workload creation, and configuration changes. Those events usually give the best early security signal.",
        },
        {
            "query_text": "How should I tune audit policy without overwhelming storage and analysis systems?",
            "reference_answer": "Use higher detail for high-value events and lower detail or narrower selection for lower-value events. Audit policy should be targeted enough to stay useful over time.",
        },
        {
            "query_text": "Why is kubectl exec activity important to monitor in a production cluster?",
            "reference_answer": "Interactive exec sessions can indicate risky manual behavior or post-compromise activity. In many environments, exec is a high-signal event worth special review.",
        },
        {
            "query_text": "How should I detect suspicious use of service accounts?",
            "reference_answer": "Baseline expected namespaces, verbs, and source patterns, then alert on abnormal token use, strange audiences, unusual API calls, or unexpected administrative actions. Service account misuse often appears as behavior that does not match the workload’s normal role.",
        },
        {
            "query_text": "What is the value of correlating audit logs with workload and network telemetry?",
            "reference_answer": "Correlation helps connect API changes to runtime and network behavior, which makes investigations faster and more accurate. It turns isolated signals into a timeline of activity.",
        },
        {
            "query_text": "How should I protect log pipelines that may contain sensitive cluster information?",
            "reference_answer": "Apply access control, retention rules, encryption, and redaction where possible. Logging systems themselves can become a secondary data-exposure path if they are not protected carefully.",
        },
        {
            "query_text": "What cluster actions are strong indicators of potential compromise?",
            "reference_answer": "Examples include unusual secret access, unexpected privilege grants, policy changes, suspicious exec sessions, new privileged pods, and newly exposed services. These are often stronger warning signs than routine application noise.",
        },
        {
            "query_text": "How should I design detection coverage for Kubernetes-specific attacks?",
            "reference_answer": "Detection should cover identity misuse, RBAC escalation, secret access, admission or policy changes, node interaction, and suspicious workload behavior. The design should reflect the main attack paths specific to Kubernetes rather than only generic host alerts.",
        },
    ],
    "multi_tenancy_and_isolation": [
        {
            "query_text": "Why is namespace separation alone not full tenant isolation in Kubernetes?",
            "reference_answer": "Namespaces separate many Kubernetes objects, but they do not isolate every kernel, node, or control-plane risk. Strong tenant isolation needs more than object grouping.",
        },
        {
            "query_text": "What is the first rule for safer multi-tenancy in Kubernetes?",
            "reference_answer": "Do not share powerful identities or broad permissions across tenants. Identity separation is one of the most important foundations for tenant safety.",
        },
        {
            "query_text": "Why should I use resource quotas and limits in a multi-tenant cluster?",
            "reference_answer": "They reduce noisy-neighbor abuse and lower the chance that one tenant can consume shared capacity aggressively. Quotas are not full security controls, but they support tenant stability and resilience.",
        },
        {
            "query_text": "How should I separate tenant access to secrets and configuration data?",
            "reference_answer": "Use dedicated namespaces, dedicated service accounts, and narrowly scoped RBAC so each tenant can access only its own material. Shared broad readers should be avoided.",
        },
        {
            "query_text": "What is the risk of placing very different trust levels on the same worker nodes?",
            "reference_answer": "If one workload escapes or abuses the node, co-located tenants may be exposed. Shared-node scheduling increases blast radius when trust boundaries are weak.",
        },
        {
            "query_text": "How can taints, tolerations, and node pools help tenant isolation?",
            "reference_answer": "They keep selected workloads on selected nodes and reduce unnecessary co-location between different trust groups. This is useful when some tenants or workloads need stronger separation.",
        },
        {
            "query_text": "Why are shared cluster-wide controllers a multi-tenancy concern?",
            "reference_answer": "Controllers with broad cluster permissions can read or change objects across many tenants. If they are compromised or misconfigured, the blast radius can be large.",
        },
        {
            "query_text": "How should I handle ingress and shared gateways in multi-tenant environments?",
            "reference_answer": "Shared gateways need strict routing, authentication, and configuration review so one tenant cannot affect another tenant’s exposure. Gateway misconfiguration can become a shared security boundary failure.",
        },
        {
            "query_text": "When should I prefer separate clusters instead of one multi-tenant cluster?",
            "reference_answer": "Separate clusters are often better when trust levels differ strongly, compliance boundaries are strict, or the cost of cross-tenant blast radius is too high. Operational simplicity alone should not override strong isolation requirements.",
        },
        {
            "query_text": "What does defense in depth look like for Kubernetes tenant isolation?",
            "reference_answer": "It combines namespaces, RBAC, network policy, pod security, node isolation, quotas, monitoring, and sometimes separate clusters. No single control is enough by itself for strong tenant separation.",
        },
    ],
    "vulnerability_management_and_incident_response": [
        {
            "query_text": "What should I do first when a Kubernetes security issue is reported?",
            "reference_answer": "Confirm scope quickly, identify affected clusters and workloads, and prioritize the most exposed or highest-impact systems first. Early triage should focus on what is actually reachable and risky.",
        },
        {
            "query_text": "Why is asset inventory important for Kubernetes incident response?",
            "reference_answer": "Without a clear map of workloads, namespaces, images, and owners, teams cannot quickly identify what is affected. Inventory is what turns a report into an actionable response plan.",
        },
        {
            "query_text": "What is a safe initial containment step for a suspicious workload in Kubernetes?",
            "reference_answer": "A safe first step is to isolate its network reach, stop new rollout if needed, preserve logs and evidence, and reduce exposure without immediately destroying everything. Fast containment should still leave room for investigation.",
        },
        {
            "query_text": "How should I prioritize remediation when many Kubernetes findings appear at once?",
            "reference_answer": "Prioritize internet-facing, privileged, secret-accessing, and business-critical workloads first. Not all findings have the same real-world exposure or blast radius.",
        },
        {
            "query_text": "What evidence should I preserve during a Kubernetes incident?",
            "reference_answer": "Preserve audit logs, workload specs, container image identifiers, node or runtime logs, timelines of changes, and relevant snapshots. Evidence should support both root-cause analysis and scope assessment.",
        },
        {
            "query_text": "How should I respond if a service account token may have been stolen?",
            "reference_answer": "Rotate or revoke the credential, reduce unnecessary permissions, search for token use in logs, and assess which workloads or namespaces may have been affected. The response should cover both containment and scope validation.",
        },
        {
            "query_text": "What is the risk of deleting a compromised pod too quickly during incident response?",
            "reference_answer": "Deleting it immediately can destroy useful forensic evidence before you understand the attack path or lateral movement. Containment should be balanced with evidence preservation.",
        },
        {
            "query_text": "How should I handle a vulnerable admission controller or operator?",
            "reference_answer": "Assess its privilege level and scope first, isolate or disable it carefully, patch it quickly, and review what objects it could have changed. Highly privileged controllers deserve urgent review during incidents.",
        },
        {
            "query_text": "What is a strong recovery check after containment and patching in a Kubernetes incident?",
            "reference_answer": "Verify that risky privileges are removed, credentials are rotated, policies are back in place, and monitoring no longer shows suspicious behavior. Recovery should prove that the cluster state is genuinely back under control.",
        },
        {
            "query_text": "How do I turn one Kubernetes incident into lasting security improvement?",
            "reference_answer": "Write the timeline, root cause, control gaps, and follow-up actions, then convert those findings into changes in RBAC, policy, logging, image controls, or response playbooks. The incident should improve the platform rather than end as a one-time fix.",
        },
    ],
}


def build_reviewed_benchmark() -> pd.DataFrame:
    rows = []
    query_counter = 1

    for topic in DEFAULT_QUERY_CONFIG.topics:
        topic_entries = TOPIC_QUERY_BANK[topic]
        if len(topic_entries) != len(DEFAULT_QUERY_CONFIG.difficulty_pattern_per_topic):
            raise ValueError(f"Topic {topic} does not have the required 10 benchmark entries.")

        for slot_index, (difficulty, entry) in enumerate(
            zip(DEFAULT_QUERY_CONFIG.difficulty_pattern_per_topic, topic_entries),
            start=1,
        ):
            rows.append(
                {
                    "query_id": f"Q{query_counter:03d}",
                    "topic": topic,
                    "difficulty": difficulty,
                    "topic_slot": f"{topic}__{slot_index:02d}",
                    "query_text": entry["query_text"],
                    "reference_answer": entry["reference_answer"],
                    "review_status": "reviewed_seed",
                    "draft_source": "structured_seed_then_manual_refinement",
                    "notes": "",
                }
            )
            query_counter += 1

    benchmark_df = pd.DataFrame(rows)
    return benchmark_df
