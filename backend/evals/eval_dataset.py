"""
Multi-Domain Evaluation datasets for measuring RAG quality.

Contains curated golden test sets across multiple technical engineering domains:
1. Docker Architecture & Containerization
2. Kubernetes & Cluster Orchestration
3. Distributed Systems & Database Storage Engines
"""

# Domain 1: Docker Architecture
DOCKER_EVAL_DATASET = [
    {
        "question": "What is Docker?",
        "ground_truth": "Docker is a platform for developing, shipping, and running applications in containers. Containers package code, runtime, system tools, libraries, and settings together.",
        "expected_concepts": ["containerization", "application packaging"],
    },
    {
        "question": "What is the difference between Docker containers and virtual machines?",
        "ground_truth": "Docker containers share the host OS kernel and are lightweight (MBs, seconds to start), while VMs include a full guest OS with hypervisor and are heavier (GBs, minutes to start).",
        "expected_concepts": ["kernel sharing", "OS isolation", "hypervisor"],
    },
    {
        "question": "What is a Docker image?",
        "ground_truth": "A Docker image is a read-only template with instructions for creating a Docker container. It is an immutable snapshot that includes the application code, runtime, libraries, and dependencies.",
        "expected_concepts": ["immutable template", "layers", "read-only"],
    },
    {
        "question": "What is the bridge network in Docker?",
        "ground_truth": "The bridge network is Docker's default network driver. Containers on the same bridge network can communicate using IP addresses. It provides network isolation between containers and the host.",
        "expected_concepts": ["default driver", "container communication", "IP addresses"],
    },
    {
        "question": "What is the difference between CMD and ENTRYPOINT in a Dockerfile?",
        "ground_truth": "CMD provides default arguments that can be overridden at runtime, while ENTRYPOINT configures the container to run as an executable and cannot be easily overridden. They can be used together, with CMD providing default arguments to ENTRYPOINT.",
        "expected_concepts": ["default arguments", "executable", "override behavior"],
    },
    {
        "question": "What are Docker volumes used for?",
        "ground_truth": "Docker volumes provide persistent data storage that survives container restarts and removal. They are managed by Docker and stored outside the container's filesystem, allowing data sharing between containers.",
        "expected_concepts": ["persistent storage", "data sharing", "container lifecycle"],
    },
]

# Domain 2: Kubernetes Orchestration
KUBERNETES_EVAL_DATASET = [
    {
        "question": "What is a Kubernetes Pod?",
        "ground_truth": "A Pod is the smallest deployable compute unit in Kubernetes representing a single instance of a running process. It can contain one or more tightly coupled containers sharing network and storage.",
        "expected_concepts": ["smallest compute unit", "shared network", "shared storage"],
    },
    {
        "question": "What is the difference between a Deployment and a StatefulSet?",
        "ground_truth": "Deployments manage stateless applications with interchangeable replica pods, while StatefulSets manage stateful applications requiring unique, stable network identifiers and persistent storage.",
        "expected_concepts": ["stateless vs stateful", "stable network identity", "persistent storage"],
    },
    {
        "question": "What role does the kube-apiserver play in the control plane?",
        "ground_truth": "The kube-apiserver is the central gateway and front-end for the Kubernetes control plane that validates and configures API objects and coordinates cluster state.",
        "expected_concepts": ["control plane front-end", "API gateway", "cluster state validation"],
    },
    {
        "question": "What is the difference between ClusterIP, NodePort, and LoadBalancer services?",
        "ground_truth": "ClusterIP exposes a service internally inside the cluster, NodePort exposes it on each node IP at a static port, and LoadBalancer provisions an external cloud load balancer routing into the cluster.",
        "expected_concepts": ["internal cluster IP", "static node port", "external load balancer"],
    },
    {
        "question": "What is a Kubernetes Ingress controller?",
        "ground_truth": "An Ingress controller is an API object and reverse proxy that manages external HTTP and HTTPS routing, SSL termination, and name-based virtual hosting to cluster services.",
        "expected_concepts": ["reverse proxy", "HTTP routing", "SSL termination"],
    },
    {
        "question": "How do ConfigMaps and Secrets differ in Kubernetes?",
        "ground_truth": "ConfigMaps store non-confidential configuration data as plaintext key-value pairs, whereas Secrets store sensitive data like tokens and passwords base64-encoded with restricted RBAC access.",
        "expected_concepts": ["non-confidential configuration", "sensitive tokens", "base64 encoding"],
    },
]

# Domain 3: Distributed Systems & Database Storage Engines
DISTRIBUTED_SYSTEMS_EVAL_DATASET = [
    {
        "question": "What does the CAP theorem state?",
        "ground_truth": "The CAP theorem states that a distributed data store can simultaneously provide at most two out of three guarantees: Consistency, Availability, and Partition Tolerance.",
        "expected_concepts": ["consistency", "availability", "partition tolerance"],
    },
    {
        "question": "How does the Raft consensus algorithm elect a leader?",
        "ground_truth": "Raft uses randomized election timeouts. When a follower node detects a heartbeat timeout, it increments term and transitions to candidate, requesting votes from peers until receiving a majority.",
        "expected_concepts": ["randomized timeout", "candidate term", "majority votes"],
    },
    {
        "question": "What is the difference between B-Trees and LSM-Trees in storage engines?",
        "ground_truth": "B-Trees perform in-place updates optimized for read-heavy workloads, whereas LSM-Trees append writes to in-memory memtables and flush sequentially to immutable SSTables, optimizing write throughput.",
        "expected_concepts": ["in-place updates", "append-only memtable", "immutable SSTables"],
    },
    {
        "question": "What is the purpose of a Write-Ahead Log in database systems?",
        "ground_truth": "A Write-Ahead Log (WAL) ensures durability by recording all data modifications sequentially to persistent disk before changes are applied to in-memory database pages.",
        "expected_concepts": ["durability", "sequential disk logging", "crash recovery"],
    },
    {
        "question": "What is the difference between Strong Consistency and Eventual Consistency?",
        "ground_truth": "Strong consistency guarantees that any read operation immediately observes the most recent write, while eventual consistency allows replicas to diverge temporarily before converging.",
        "expected_concepts": ["immediate latest read", "temporary replica divergence", "convergence"],
    },
    {
        "question": "How does two-phase locking prevent concurrency anomalies?",
        "ground_truth": "Two-phase locking (2PL) enforces a growing phase where transactions only acquire locks and a shrinking phase where they only release locks, guaranteeing conflict serializability.",
        "expected_concepts": ["growing phase", "shrinking phase", "serializability"],
    },
]

# Map of all registered evaluation datasets by technical domain
ALL_EVAL_DATASETS = {
    "Docker Architecture": DOCKER_EVAL_DATASET,
    "Kubernetes Orchestration": KUBERNETES_EVAL_DATASET,
    "Distributed Systems": DISTRIBUTED_SYSTEMS_EVAL_DATASET,
}