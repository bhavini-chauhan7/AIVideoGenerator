"""Find free Hugging Face Spaces that turn a picture into an animated clip, and test them.

Usage: python check_video_spaces.py <test_image.jpg>
"""
import json
import signal
import sys
import time

import requests
from gradio_client import Client, handle_file

IMG = sys.argv[1]
PROMPT = ("3D Pixar-style animation. The big dad chews a cookie guiltily while the little "
          "red-haired girl points at him and stomps her foot. Smooth natural motion.")
KNOWN = [
    "zerogpu-aoti/wan2-2-fp8da-aoti-faster",
    "r3gm/wan2-2-fp8da-aoti-preview",
    "multimodalart/wan2-1-fast",
    "Lightricks/ltx-video-distilled",
    "Lightricks/LTX-Video-0.9.8-13B-distilled",
    "Wan-AI/Wan-2.2-5B",
    "alexnasa/Wan2.2-I2V-Fast",
]


class Timeout(Exception):
    pass


def alarm(seconds):
    def handler(*_):
        raise Timeout(f"timed out after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def discover():
    found = list(KNOWN)
    for q in ("image-to-video", "image to video", "wan2.2", "wan 2.2 i2v", "ltx-video", "i2v"):
        try:
            r = requests.get("https://huggingface.co/api/spaces",
                             params={"search": q, "sort": "likes", "direction": -1, "limit": 25}, timeout=30)
            found += [s["id"] for s in r.json()]
        except Exception as e:
            print("search failed", q, e)
    seen, out = set(), []
    for sid in found:
        if sid in seen:
            continue
        seen.add(sid)
        try:
            info = requests.get(f"https://huggingface.co/api/spaces/{sid}", timeout=20).json()
            rt = info.get("runtime", {})
            hw = (rt.get("hardware") or {}).get("current")
            stage = rt.get("stage")
            if stage == "RUNNING" and hw and ("zero" in hw or "a10" in hw or "a100" in hw or "l4" in hw or "h100" in hw):
                out.append((sid, hw, info.get("likes", 0)))
        except Exception:
            pass
    out.sort(key=lambda x: -x[2])
    return out


def endpoints(client):
    api = client.view_api(print_info=False, return_format="dict")
    res = []
    for name, ep in (api.get("named_endpoints") or {}).items():
        params = ep.get("parameters", [])
        rets = ep.get("returns", [])
        has_img = any((p.get("component") or "").lower() == "image" for p in params)
        has_vid = any("video" in (r.get("component") or "").lower() or "video" in json.dumps(r).lower() for r in rets)
        if has_img and has_vid:
            res.append((name, params))
    return res


def try_call(client, name, params):
    kwargs = {}
    for p in params:
        pname = p.get("parameter_name") or p.get("label")
        comp = (p.get("component") or "").lower()
        label = (p.get("label") or pname or "").lower()
        if comp == "image" and "image" not in str(kwargs.values()):
            kwargs[pname] = handle_file(IMG)
        elif comp in ("textbox", "text") and "prompt" in label and "neg" not in label:
            kwargs[pname] = PROMPT
    t = time.time()
    job = client.submit(api_name=name, **kwargs)
    result = job.result(timeout=420)
    return result, time.time() - t, list(kwargs)


cands = discover()
print(f"Found {len(cands)} running GPU spaces:")
for c in cands[:30]:
    print("  ", c)

wins = 0
for sid, hw, likes in cands[:14]:
    print(f"\n=== {sid} ({hw}, {likes} likes)", flush=True)
    try:
        alarm(60)
        client = Client(sid)
        eps = endpoints(client)
        signal.alarm(0)
        print("   picture->video endpoints:", [(n, [(p.get('parameter_name'), p.get('component'), p.get('parameter_default') if 'parameter_default' in p else '-') for p in ps]) for n, ps in eps])
        for name, params in eps[:1]:
            alarm(480)
            result, secs, used = try_call(client, name, params)
            signal.alarm(0)
            print(f"   OK {name} in {secs:.0f}s using {used}: {str(result)[:300]}", flush=True)
            wins += 1
    except Exception as e:
        signal.alarm(0)
        print(f"   FAIL {type(e).__name__}: {str(e)[:400]}", flush=True)
    if wins >= 3:
        break
print(f"\n{wins} working picture->video spaces")
