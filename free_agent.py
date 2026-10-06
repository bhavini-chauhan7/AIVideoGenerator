#!/usr/bin/env python3
"""FREE video agent: no API keys, no payment.

    python free_agent.py "baby catches daddy eating her cookies"
    python free_agent.py            # no idea? it invents one (or uses a built-in demo)

How it stays free:
  story    -> Claude if ANTHROPIC_API_KEY is set, otherwise a small open-source
              AI (Qwen) running on this computer's CPU
  pictures -> free Hugging Face FLUX space (fallback: free Pollinations image API)
  voices   -> free Microsoft Edge read-aloud voices (edge-tts), incl. a child voice
  video    -> ffmpeg: slow camera moves on each picture + voices + big subtitles

The result is a "talking storybook" style short. For true animation with
lip-sync, use the paid agent.py instead.
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.parse
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
MUSIC_DIR = ROOT / "assets" / "music"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H, FPS = 1080, 1920, 30
UA = {"User-Agent": "LittlePixelsFreeAgent/1.0"}


def log(msg):
    print(f"[free-agent] {msg}", flush=True)


def slugify(text, n=40):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "video"


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# --------------------------------------------------------------------------
# Story
# --------------------------------------------------------------------------

DEMO_PLAN = {
    "title": "Daddy's Cookie Crime",
    "idea": "baby catches daddy eating her cookies",
    "hook_caption": "\"I tell Mommy on you!\" 😂🍪❤️",
    "description": "She caught me red-handed... and negotiated like a pro. Daddy's cookie crime is safe, for now.",
    "hashtags": ["pixar", "dadlife", "toddlersbelike", "cutebaby", "fatherdaughter", "funnybaby"],
    "scenes": [
        {
            "characters": ["daddy", "lily"],
            "image_prompt": "Cozy kitchen at night under a warm lamp. The dad is frozen mid-bite with a cookie in his mouth and his hand inside a pink cookie jar, eyes wide and guilty. In the doorway the little girl stands holding a teddy bear, pointing at him with a shocked face. Medium-wide shot.",
            "action": "slow push in on the dad's guilty face",
            "dialogue": [{"speaker": "lily", "line": "Daddyyy! Dat MY cookie!"}],
        },
        {
            "characters": ["daddy", "lily"],
            "image_prompt": "Cozy kitchen at night. The dad crouches down to the little girl's height with cookie crumbs on his chin, holding a finger to his lips with an awkward smile. The little girl has her arms crossed and squints at him suspiciously, chin up. Close two-shot.",
            "action": "gentle drift between their faces",
            "dialogue": [
                {"speaker": "daddy", "line": "Shhh... our little secret?"},
                {"speaker": "lily", "line": "Hmph! I tell Mommy on you!"},
            ],
        },
        {
            "characters": ["daddy", "lily"],
            "image_prompt": "Cozy kitchen at night with golden bokeh. The little girl happily hugs the dad's neck while holding a half-eaten cookie, giggling with eyes closed. The dad melts with a huge happy smile. Close-up, heartwarming.",
            "action": "slow zoom out",
            "dialogue": [
                {"speaker": "lily", "line": "Otay... secret, Daddy. Hee hee!"},
                {"speaker": "daddy", "line": "Best partner in crime ever."},
            ],
        },
    ],
}


def story_prompt(cast, idea, n_scenes, language):
    chars = "\n".join(f"- id \"{cid}\" ({c['name']}): {c['look'].strip()}"
                      for cid, c in cast["characters"].items())
    return f"""You write viral, wholesome, funny 3D Pixar-style family shorts for the
channel "{cast['channel_name']}". Recurring cast:
{chars}

Write a {n_scenes}-scene short about: {idea or 'an idea of your own that would go viral'}.
Formula: a cute, slightly sassy toddler line, a big exaggerated adult reaction,
a sweet ending. Each scene is ONE still picture with at most 2 short spoken lines
(under 15 words total). Dialogue language: {language}.
image_prompt must fully describe the shot (who, pose, expression, setting,
framing) on its own and contain no text or captions.

Reply with ONLY a JSON object, no markdown, in exactly this shape:
{{"title": str, "hook_caption": str (with emojis), "description": str,
 "hashtags": [str], "scenes": [{{"characters": [character ids],
 "image_prompt": str, "action": str, "dialogue": [{{"speaker": character id, "line": str}}]}}]}}"""


def parse_json(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


IDEAS = [
    "baby catches daddy eating her cookies",
    "Lily puts makeup on daddy while he naps",
    "daddy pretends Lily's tiny roar is the scariest thing ever",
    "mommy asks who made the mess and Lily points at daddy",
    "Lily tries to say 'I love you' but mispronounces it",
    "daddy tries to braid Lily's hair for the first time",
    "Lily refuses to share her ice cream with daddy",
    "Lily teaches daddy how to dance",
    "daddy pretends to cry so Lily will give him a hug",
    "Lily hides daddy's phone and acts innocent",
    "Lily wants daddy to wear her pink bow",
    "daddy and Lily have a tickle fight",
    "Lily insists her teddy bear is hungry at the restaurant",
    "Lily tells mommy that daddy said a bad word",
    "Lily wakes daddy up at 5am to play",
]


def llm_local(prompt, model_id):
    """Small open-source AI that runs right here on the CPU: free, no account."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    example = json.dumps(DEMO_PLAN, ensure_ascii=False)
    msgs = [
        {"role": "system", "content": "You are a funny, warm screenwriter. You answer with valid JSON only."},
        {"role": "user", "content": prompt + "\n\nExample of the exact JSON format (write a NEW story, "
                                             "do not copy this one):\n" + example},
    ]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1100, do_sample=True, temperature=0.8, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def validate_plan(plan, cast, n_scenes):
    ids = set(cast["characters"])
    scenes = []
    for s in plan["scenes"][:n_scenes]:
        chars = [c for c in s.get("characters", []) if c in ids] or list(ids)[:1]
        lines = [d for d in s.get("dialogue", []) if d.get("speaker") in ids and d.get("line", "").strip()]
        scenes.append({"characters": chars, "image_prompt": s["image_prompt"],
                       "action": s.get("action", ""), "dialogue": lines})
    if not scenes:
        raise ValueError("plan has no scenes")
    plan["scenes"] = scenes
    plan.setdefault("hashtags", [])
    plan.setdefault("description", "")
    plan.setdefault("hook_caption", plan.get("title", ""))
    return plan


def write_plan(cfg, cast, idea, n_scenes):
    if not idea and os.environ.get("USE_DEMO", "") == "1":
        return json.loads(json.dumps(DEMO_PLAN))
    language = cfg.get("language", "English")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import agent  # the paid agent's Claude writer
            log("Writing the story with Claude...")
            plan = agent.write_plan(cfg, cast, idea or agent.brainstorm(cfg, cast, 1)[0], n_scenes)
            return validate_plan(plan, cast, n_scenes)
        except Exception as e:  # fall through to the free writers
            log(f"Claude unavailable ({e}); using a free writer")

    if not idea:
        idea = random.choice(IDEAS)
        log(f"No idea given, the agent picked: {idea}")
    prompt = story_prompt(cast, idea, n_scenes, language)
    model_id = cfg.get("free_story_model", "Qwen/Qwen2.5-1.5B-Instruct")
    for attempt in range(3):
        try:
            log(f"Writing the story with a free AI on this computer ({model_id})...")
            plan = validate_plan(parse_json(llm_local(prompt, model_id)), cast, n_scenes)
            plan["idea"] = idea
            return plan
        except Exception as e:
            log(f"  attempt {attempt + 1} failed: {e}")
    if idea == DEMO_PLAN["idea"]:
        return json.loads(json.dumps(DEMO_PLAN))
    sys.exit("[free-agent] The free story writer could not write a story. Please try again.")


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------

def scene_prompt(cfg, cast, scene):
    looks = " ".join(f"{cast['characters'][c]['name']}: {cast['characters'][c]['look'].strip()}"
                     for c in scene["characters"])
    return f"{cfg['style'].strip()} {scene['image_prompt']} Characters: {looks}"


def image_pollinations(prompt, seed, dest):
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:1800]) +
           f"?width=768&height=1344&seed={seed}&model=flux&nologo=true&private=true")
    r = requests.get(url, headers=UA, timeout=240)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/") or len(r.content) < 10_000:
        raise RuntimeError(f"not an image ({r.headers.get('content-type')}, {len(r.content)} bytes)")
    dest.write_bytes(r.content)


def image_huggingface(prompt, seed, dest):
    from gradio_client import Client

    client = Client("black-forest-labs/FLUX.1-schnell", hf_token=os.environ.get("HF_TOKEN") or None)
    result = client.predict(prompt=prompt[:1800], seed=seed, randomize_seed=False, width=768,
                            height=1344, num_inference_steps=4, api_name="/infer")
    path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(path, dict):
        path = path.get("path") or path.get("url")
    shutil.copy(path, dest)


def make_picture(cfg, cast, scene, seed, dest):
    prompt = scene_prompt(cfg, cast, scene)
    raw = dest.with_suffix(".download")
    for name, fn in (("Hugging Face", image_huggingface), ("Pollinations", image_pollinations)):
        for attempt in range(3):
            try:
                fn(prompt, seed, raw)
                ffmpeg("-i", raw, "-q:v", "2", dest)
                raw.unlink()
                return
            except Exception as e:
                log(f"  {name} image attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
    sys.exit("[free-agent] Could not get a picture from any free service. Try again later.")


# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------

async def _tts(text, voice, rate, pitch, dest):
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(dest))


def make_voice(cast, speaker, line, dest):
    c = cast["characters"][speaker]
    for attempt in range(3):
        try:
            asyncio.run(_tts(line, c.get("tts_voice", "en-US-JennyNeural"),
                             c.get("tts_rate", "+0%"), c.get("tts_pitch", "+0Hz"), dest))
            return
        except Exception as e:
            log(f"  voice attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    sys.exit("[free-agent] Could not create the voices. Try again later.")


# --------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------

LEAD, GAP, TAIL = 0.6, 0.35, 0.9
MOVES = [
    # (zoom expression, x expression, y expression)
    ("min(1+0.0009*on,1.18)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),            # push in
    ("1.15", "(iw-iw/zoom)*on/{n}", "ih/2-(ih/zoom/2)"),                             # pan right
    ("max(1.18-0.0009*on,1)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),            # pull out
    ("1.15", "(iw-iw/zoom)*(1-on/{n})", "ih/2-(ih/zoom/2)"),                         # pan left
]


def build_scene(cast, scene, idx, img, folder):
    # 1) voices, laid out on a timeline
    parts, cues, t = [], [], LEAD
    for j, d in enumerate(scene["dialogue"]):
        mp3 = folder / f"s{idx}_line{j}.mp3"
        make_voice(cast, d["speaker"], d["line"], mp3)
        dur = duration(mp3)
        parts.append((mp3, t))
        cues.append((d, t, t + dur + 0.25))
        t += dur + GAP
    total = max(t - GAP + TAIL, 3.5) if parts else 4.0

    audio = folder / f"s{idx}_audio.wav"
    if parts:
        inputs, filters = [], []
        for k, (mp3, start) in enumerate(parts):
            inputs += ["-i", mp3]
            ms = int(start * 1000)
            filters.append(f"[{k}:a]aresample=44100,aformat=channel_layouts=stereo,adelay={ms}|{ms}[a{k}]")
        mix = "".join(f"[a{k}]" for k in range(len(parts)))
        filters.append(f"{mix}amix=inputs={len(parts)}:normalize=0,apad,atrim=0:{total:.3f}[out]")
        ffmpeg(*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", audio)
    else:
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{total:.3f}", audio)

    # 2) picture with a slow camera move + subtitles
    frames = int(total * FPS)
    z, x, y = MOVES[idx % len(MOVES)]
    vf = [f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2}",
          f"zoompan=z='{z}':x='{x.format(n=frames)}':y='{y.format(n=frames)}':d={frames}:s={W}x{H}:fps={FPS}"]
    for k, (d, start, end) in enumerate(cues):
        color = cast["characters"][d["speaker"]].get("subtitle_color", "white")
        lines = textwrap.wrap(d["line"], 18)
        for li, text in enumerate(lines):
            tf = folder / f"s{idx}_cue{k}_{li}.txt"
            tf.write_text(text)
            ypos = int(H * 0.70) + li * 92 - (len(lines) - 1) * 46
            vf.append(f"drawtext=fontfile={FONT}:textfile={tf}:expansion=none:fontsize=72:"
                      f"fontcolor={color}:borderw=7:bordercolor=black@0.85:"
                      f"x=(w-text_w)/2:y={ypos}:enable='between(t,{start:.2f},{end:.2f})'")
    vf += [f"fade=t=in:st=0:d=0.25", f"fade=t=out:st={total - 0.25:.2f}:d=0.25", "format=yuv420p"]

    out = folder / f"scene_{idx + 1:02d}.mp4"
    ffmpeg("-loop", "1", "-framerate", FPS, "-i", img, "-i", audio, "-vf", ",".join(vf),
           "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
           "-c:a", "aac", "-b:a", "160k", "-shortest", out)
    return out


def join(clips, dest, music_volume):
    listfile = dest.with_name("clips.txt")
    listfile.write_text("".join(f"file '{c.name}'\n" for c in clips))
    joined = dest.with_name("joined.mp4")
    ffmpeg("-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", "-movflags", "+faststart", joined)
    listfile.unlink()
    music = sorted(MUSIC_DIR.glob("*.mp3"))
    if music:
        track = random.choice(music)
        log(f"Adding background music: {track.name}")
        ffmpeg("-i", joined, "-stream_loop", "-1", "-i", track, "-filter_complex",
               f"[1:a]volume={music_volume}[m];[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-movflags", "+faststart", dest)
        joined.unlink()
    else:
        joined.rename(dest)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="FREE Pixar-style family video agent")
    p.add_argument("idea", nargs="*")
    p.add_argument("-n", "--scenes", type=int, default=None)
    p.add_argument("--out", default="output", help="folder to put videos in")
    p.add_argument("--demo", action="store_true", help="use the built-in demo story")
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cast = yaml.safe_load((ROOT / "characters.yaml").read_text())
    n_scenes = args.scenes or cfg.get("scenes_per_video", 3)
    idea = " ".join(args.idea).strip()
    if args.demo:
        os.environ["USE_DEMO"] = "1"
        idea = ""
    if not shutil.which("ffmpeg"):
        sys.exit("[free-agent] ffmpeg is not installed.")

    plan = write_plan(cfg, cast, idea, n_scenes)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(args.out) / f"{stamp}-{slugify(plan['title'])}"
    work = folder / "work"
    work.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    log(f"Title: {plan['title']}")

    seed = random.randint(1, 10**6)  # same seed for every scene helps faces stay similar
    clips = []
    for i, scene in enumerate(plan["scenes"]):
        log(f"Scene {i + 1}/{len(plan['scenes'])}: " +
            (" / ".join(f"{d['speaker']}: {d['line']}" for d in scene["dialogue"]) or "(no lines)"))
        img = folder / f"scene_{i + 1:02d}.jpg"
        log("  drawing picture (free)...")
        make_picture(cfg, cast, scene, seed, img)
        log("  adding voices, camera move and subtitles...")
        clips.append(build_scene(cast, scene, i, img, work))
        time.sleep(2)  # be polite to the free services

    final = folder / "final.mp4"
    join(clips, work / "final.mp4", cfg.get("music_volume", 0.12))
    shutil.move(str(work / "final.mp4"), final)
    shutil.rmtree(work)

    caption = (f"{plan['hook_caption']}\n\n{plan['description']}\n\n"
               + " ".join(h if h.startswith("#") else f"#{h}" for h in plan["hashtags"])
               + "\n\n(AI-generated content)")
    (folder / "caption.txt").write_text(caption)
    log(f"DONE -> {final}  ({duration(final):.1f}s)")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"folder={folder}\n")


if __name__ == "__main__":
    main()
