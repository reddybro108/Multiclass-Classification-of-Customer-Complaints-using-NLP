from mcp.server import Server
import asyncio

server = Server(name="mcp-chat-server")

async def chat_tool(input: dict) -> dict:
    user_message = input.get("message", "")
    reply = f"You said: {user_message}. This is the MCP server responding."
    return {"reply": reply}

# Manually register the tool
server._tools["chat"] = chat_tool

if __name__ == "__main__":
    server.run()
