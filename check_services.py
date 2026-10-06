"""Quick health check of every free service free_agent.py can use."""
import asyncio, json, os, sys, tempfile, time, urllib.parse
import requests

def show(name, fn):
    t = time.time()
    try:
        print(f"OK   {name}: {fn()}  ({time.time() - t:.1f}s)", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}  ({time.time() - t:.1f}s)", flush=True)

def gh_catalog():
    r = requests.get("https://models.github.ai/catalog/models",
                     headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}, timeout=30)
    ids = [m.get("id") for m in r.json()] if r.ok else []
    return f"{r.status_code} {len(ids)} models e.g. {ids[:15]}"

def gh_chat(model):
    def f():
        r = requests.post("https://models.github.ai/inference/chat/completions",
                          headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "messages": [{"role": "user", "content": "Say hi in 3 words"}]},
                          timeout=60)
        return f"{r.status_code} {r.headers.get('content-type')} {r.text[:200]!r}"
    return f

def poll_image(base):
    def f():
        r = requests.get(base + urllib.parse.quote("cute 3d cartoon baby girl, pixar style") +
                         "?width=512&height=896&seed=1&model=flux&nologo=true", timeout=180,
                         headers={"User-Agent": "LittlePixelsFreeAgent/1.0"})
        ctype = r.headers.get("content-type", "")
        body = "" if ctype.startswith("image") else repr(r.text[:120])
        return f"{r.status_code} {ctype} {len(r.content)} bytes {body}"
    return f

def hf_flux():
    from gradio_client import Client
    c = Client("black-forest-labs/FLUX.1-schnell")
    res = c.predict(prompt="cute 3d cartoon baby girl, pixar style", seed=1, randomize_seed=False,
                    width=512, height=896, num_inference_steps=4, api_name="/infer")
    return repr(res)[:200]

def edge():
    import edge_tts
    out = os.path.join(tempfile.gettempdir(), "t.mp3")
    asyncio.run(edge_tts.Communicate("Daddy! That my cookie!", "en-US-AnaNeural", pitch="+12Hz").save(out))
    return f"{os.path.getsize(out)} bytes"

show("GitHub Models catalog", gh_catalog)
for m in ("openai/gpt-4.1-mini", "openai/gpt-4o-mini", "meta/Llama-3.3-70B-Instruct"):
    show(f"GitHub Models chat {m}", gh_chat(m))
show("Pollinations image (image.pollinations.ai)", poll_image("https://image.pollinations.ai/prompt/"))
show("Pollinations image (gen.pollinations.ai)", poll_image("https://gen.pollinations.ai/image/"))
show("Hugging Face FLUX.1-schnell space", hf_flux)
show("Edge TTS child voice", edge)
