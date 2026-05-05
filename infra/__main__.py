"""
infra/__main__.py
Upgraded from your friend's placeholder to a REAL Pulumi script
that provisions all containers needed for the full project.

Run:
  cd infra
  pulumi login --local
  pulumi stack init dev
  pulumi up

🔑 PRIVATE DATA NEEDED:
   None for local Docker deployment.
   If you upgrade to AWS later → set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY in .env
"""

import pulumi
import pulumi_docker as docker

# ──────────────────────────────────────────────────────────────────
#  CONFIG  (matches your friend's Pulumi.yaml: name=rag-local)
# ──────────────────────────────────────────────────────────────────

config       = pulumi.Config()
project_name = "rag-local"        # kept from friend's Pulumi.yaml
environment  = "dev"

# ──────────────────────────────────────────────────────────────────
#  SHARED DOCKER NETWORK
# ──────────────────────────────────────────────────────────────────

network = docker.Network("aegis-net", name="aegis-net")

# ──────────────────────────────────────────────────────────────────
#  1. CHROMADB  (your friend's vector store)
# ──────────────────────────────────────────────────────────────────

chroma = docker.Container(
    "chromadb",
    image="chromadb/chroma:latest",
    name="aegis-chroma",
    ports=[docker.ContainerPortArgs(internal=8000, external=8100)],   # 8100 to avoid clash
    volumes=[docker.ContainerVolumeArgs(
        container_path="/chroma/chroma",
        host_path="./backend/chroma_db",
    )],
    networks_advanced=[docker.ContainerNetworksAdvancedArgs(name=network.name)],
)

# ──────────────────────────────────────────────────────────────────
#  2. FASTAPI BACKEND  (your friend's + your combined api.py)
# ──────────────────────────────────────────────────────────────────

backend_image = docker.Image(
    "backend-image",
    build=docker.DockerBuildArgs(context=".", dockerfile="docker/Dockerfile.backend"),
    image_name="aegis-backend:latest",
    skip_push=True,
)

backend = docker.Container(
    "fastapi-backend",
    image=backend_image.image_name,
    name="aegis-api",
    ports=[docker.ContainerPortArgs(internal=8000, external=8000)],
    envs=[
        "CHROMA_HOST=aegis-chroma",
        "CHROMA_PORT=8100",
        "CONFIDENCE_THRESHOLD=0.65",
        # 🔑 Set these in .env — Pulumi reads them as stack config or env vars
        f"ARIZE_API_KEY={config.get('arize_api_key') or ''}",
        f"ARIZE_SPACE_ID={config.get('arize_space_id') or ''}",
        f"WANDB_API_KEY={config.get('wandb_api_key') or ''}",
        f"WANDB_PROJECT={config.get('wandb_project') or 'aegis-autochat'}",
    ],
    networks_advanced=[docker.ContainerNetworksAdvancedArgs(name=network.name)],
    opts=pulumi.ResourceOptions(depends_on=[chroma]),
)

# ──────────────────────────────────────────────────────────────────
#  3. PREFECT SERVER  (orchestrator for your pipeline)
# ──────────────────────────────────────────────────────────────────

prefect_server = docker.Container(
    "prefect-server",
    image="prefecthq/prefect:3-latest",
    name="aegis-prefect",
    ports=[docker.ContainerPortArgs(internal=4200, external=4200)],
    command=["prefect", "server", "start", "--host", "0.0.0.0"],
    networks_advanced=[docker.ContainerNetworksAdvancedArgs(name=network.name)],
)

# ──────────────────────────────────────────────────────────────────
#  4. PROMETHEUS  (friend already uses this for /metrics scraping)
# ──────────────────────────────────────────────────────────────────

prometheus = docker.Container(
    "prometheus",
    image="prom/prometheus:latest",
    name="aegis-prometheus",
    ports=[docker.ContainerPortArgs(internal=9090, external=9090)],
    volumes=[docker.ContainerVolumeArgs(
        container_path="/etc/prometheus/prometheus.yml",
        host_path="./docker/prometheus.yml",
    )],
    networks_advanced=[docker.ContainerNetworksAdvancedArgs(name=network.name)],
)

# ──────────────────────────────────────────────────────────────────
#  5. GRAFANA  (friend uses this; admin/admin by default)
# ──────────────────────────────────────────────────────────────────

grafana = docker.Container(
    "grafana",
    image="grafana/grafana:latest",
    name="aegis-grafana",
    ports=[docker.ContainerPortArgs(internal=3000, external=3000)],
    envs=["GF_SECURITY_ADMIN_PASSWORD=admin"],
    networks_advanced=[docker.ContainerNetworksAdvancedArgs(name=network.name)],
    opts=pulumi.ResourceOptions(depends_on=[prometheus]),
)

# ──────────────────────────────────────────────────────────────────
#  EXPORTS  (kept from friend + your additions)
# ──────────────────────────────────────────────────────────────────

pulumi.export("project",            project_name)
pulumi.export("environment",        environment)
pulumi.export("api_url",            "http://localhost:8000")
pulumi.export("prefect_ui",         "http://localhost:4200")
pulumi.export("grafana_ui",         "http://localhost:3000")
pulumi.export("prometheus_ui",      "http://localhost:9090")
pulumi.export("chromadb_port",      "8100")
