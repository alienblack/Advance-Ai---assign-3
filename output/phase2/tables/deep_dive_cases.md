# Deep-Dive Cases

## Q052
**Question:** Why should confidential Kubernetes data be encrypted at rest?
**Rationale:** Clear V2 win: improved retrieval focus and best balanced answer on etcd encryption-at-rest.

| Variant | Retrieval | Proxy Composite | Context Relevance | Answer Relevance | Faithfulness | Top Sources |
|---|---|---:|---:|---:|---:|---|
| V2 | V2 | 0.894 | 0.818 | 0.864 | 1.000 | k8s_operational_security_docs, k8s_security_observability_book |
| QW_V1 | V1 | 0.785 | 0.533 | 0.822 | 1.000 | k8s_operational_security_docs |
| V1 | V1 | 0.774 | 0.533 | 0.789 | 1.000 | k8s_operational_security_docs |
| QW_V3 | V3 | 0.735 | 0.471 | 0.734 | 1.000 | owasp_k8s_cheatsheet, k8s_security_docs, k8s_security_observability_book |
| V3 | V3 | 0.716 | 0.471 | 0.676 | 1.000 | owasp_k8s_cheatsheet, k8s_security_docs, k8s_security_observability_book |

## Q048
**Question:** Why are hostNetwork pods a concern in a hardened cluster?
**Rationale:** Clear V3 win: intent-enriched retrieval better surfaces the hostNetwork / NetworkPolicy interaction.

| Variant | Retrieval | Proxy Composite | Context Relevance | Answer Relevance | Faithfulness | Top Sources |
|---|---|---:|---:|---:|---:|---|
| V3 | V3 | 0.913 | 0.846 | 0.893 | 1.000 | k8s_security_observability_book, k8s_security_docs |
| QW_V3 | V3 | 0.815 | 0.846 | 0.847 | 0.750 | k8s_security_observability_book, k8s_security_docs |
| V1 | V1 | 0.789 | 0.478 | 0.888 | 1.000 | k8s_security_docs, aalto_k8s_security_thesis, k8s_security_observability_book |
| V2 | V2 | 0.789 | 0.478 | 0.888 | 1.000 | k8s_security_docs, aalto_k8s_security_thesis, k8s_security_observability_book |
| QW_V1 | V1 | 0.743 | 0.478 | 0.750 | 1.000 | k8s_security_docs, aalto_k8s_security_thesis, k8s_security_observability_book |

## Q050
**Question:** What extra controls are useful when NetworkPolicy alone is not enough?
**Rationale:** Failure case: retrieval is moderately relevant, but answers still under-specify the requested extra controls beyond NetworkPolicy.

| Variant | Retrieval | Proxy Composite | Context Relevance | Answer Relevance | Faithfulness | Top Sources |
|---|---|---:|---:|---:|---:|---|
| QW_V1 | V1 | 0.725 | 0.500 | 0.674 | 1.000 | k8s_operational_security_docs |
| QW_V3 | V3 | 0.704 | 0.562 | 0.549 | 1.000 | k8s_security_observability_book |
| V3 | V3 | 0.668 | 0.562 | 0.442 | 1.000 | k8s_security_observability_book |
| V1 | V1 | 0.634 | 0.500 | 0.651 | 0.750 | k8s_operational_security_docs |
| V2 | V2 | 0.634 | 0.500 | 0.651 | 0.750 | k8s_operational_security_docs |
