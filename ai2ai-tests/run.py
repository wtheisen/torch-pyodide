import asyncio, sys
from playwright.async_api import async_playwright
code=open(sys.argv[1]).read()
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(args=['--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-webgpu-adapter=swiftshader','--use-angle=swiftshader','--ignore-gpu-blocklist'])
        pg=await b.new_page(); pg.on('console', lambda m: print('console:',m.text) if 'rror' in m.text else None)
        await pg.goto('http://127.0.0.1:8901/'+(sys.argv[2] if len(sys.argv)>2 else 'index.html')+''); await pg.wait_for_function('window.__ready',timeout=120000)
        if await pg.evaluate('!!window.py'):
            await pg.evaluate('async c=>{try{await py.runPythonAsync(c)}catch(e){document.getElementById("o").textContent+="PYERR "+e.message.slice(-1500)}}', code)
        print(await pg.inner_text('#o')); await b.close()
asyncio.run(main())
