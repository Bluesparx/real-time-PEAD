import sys
import asyncio
import uvicorn
from app import app 

if __name__ == '__main__':
    if sys.platform == 'win32':
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(uvicorn.Server(config=uvicorn.Config(app=app, host="0.0.0.0", port=8000)).serve())
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
