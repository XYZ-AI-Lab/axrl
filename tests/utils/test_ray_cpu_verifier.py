from axrl.ray.ray_cpu_worker import RayCPUWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import setup_logger
from axrl.verifier.base_verifier import VerifierInput
from axrl.verifier.hf_math_verifier import HfMathVerifier


def test_cpu_worker() -> None:
    import asyncio

    setup_logger("info")
    resource_group = ResourceGroup([Request(gpu=0, cpu=2), Request(gpu=0, cpu=2)])
    worker = RayCPUWorker(HfMathVerifier, config=None, resource_group=resource_group)
    worker.initialize()
    reqs = [
        VerifierInput(label="123", output_text="123"),
        VerifierInput(label="456", output_text="789"),
        VerifierInput(label="10", output_text="5 + 5"),
    ]
    for req in reqs:
        score = asyncio.run(worker.generate(req=req))
        print(f"Req: {req}, Score: {score}")

    async def batch_generate() -> list:
        tasks = [worker.generate(req=req) for req in reqs]
        return await asyncio.gather(*tasks)

    results: list = asyncio.run(batch_generate())
    for req, score in zip(reqs, results, strict=True):
        print(f"[Batch] Req: {req}, Score: {score}")

    assert [r.score for r in results] == [1.0, 0.0, 1.0]


if __name__ == "__main__":
    test_cpu_worker()
