import streamlit as st
from jaide_v27_ultimate_orchestrator import UltimateOrchestrator
import asyncio

async def main():
    orchestrator = UltimateOrchestrator("config.json")
    await orchestrator.streamlit_interface()

if __name__ == "__main__":
    asyncio.run(main())
