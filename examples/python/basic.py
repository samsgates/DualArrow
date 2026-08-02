import asyncio

from dualarrow import DualArrow, QueryRequest


async def main() -> None:
    async with DualArrow.from_config("dualarrow.yaml") as engine:
        await engine.ingest()
        result = await engine.query(
            QueryRequest(
                query="Which bearing is approved for P-104?",
                retrieval_profile="balanced",
            )
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
