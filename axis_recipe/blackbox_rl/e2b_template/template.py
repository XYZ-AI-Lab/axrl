from e2b import Template

template = (
    Template()
    .from_python_image("3.12-slim")
    .apt_install(["bash", "ca-certificates", "curl", "git", "tmux"], no_install_recommends=True)
    .run_cmd("curl -fsSL https://install.openhands.dev/install.sh | sh", user="root")
    .make_dir("/workspace", user="root")
    .run_cmd("chown user:user /workspace", user="root")
    .set_workdir("/workspace")
)
