from setuptools import find_packages, setup

setup(
    name="axrl",
    version="0.1.0",
    packages=find_packages(include=["axrl", "axrl.*", "axis_recipe", "axis_recipe.*"]),
    python_requires=">=3.12",
    install_requires=[
        "e2b>=2.34.0",
    ],
    description="Agentic RL post-training framework built on SGLang rollout, Megatron training, and real-world agent workflows.",
    url="https://github.com/XYZ-AI-Lab/axrl",
)
