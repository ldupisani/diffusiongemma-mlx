#!/usr/bin/env python3
"""A tiny stdio MCP server for the DiffusionGemma agent demo.

Exposes two trivial tools. The agent (mcp_agent.py) launches this file as a
subprocess and talks to it over stdio using the MCP protocol — so this is a
real MCP server, not a mock. Swap these tools for anything (filesystem, HTTP,
a database) and the agent loop is unchanged.
"""

import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("difg-demo")


@mcp.tool()
def random_int(low: int, high: int) -> int:
    """Return a random integer between low and high (inclusive).

    The result is unpredictable, so the model must call this and read the result
    before deciding its next move — it cannot pre-compute a chain of calls.
    """
    return random.randint(low, high)


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Canned data so the demo is deterministic and offline.
    table = {
        "paris": "18°C, light rain",
        "tokyo": "26°C, clear",
        "reykjavik": "7°C, windy",
    }
    return table.get(city.lower(), f"No data for {city}; assume 20°C, partly cloudy.")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
