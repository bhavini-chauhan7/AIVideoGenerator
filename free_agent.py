#!/usr/bin/env python3
"""FREE video agent: no API keys, no payment.

    python free_agent.py "baby catches daddy eating her cookies"
    python free_agent.py            # no idea? it invents one (or uses a built-in demo)

How it stays free:
  story    -> Claude if ANTHROPIC_API_KEY is set, otherwise a small open-source
              AI (Qwen) running on this computer's CPU
  pictures -> free Hugging Face FLUX space (fallback: free Pollinations image API)
  voices   -> free Microsoft Edge read-aloud voices (edge-tts), incl. a child voice
  motion   -> free Hugging Face Wan 2.2 spaces animate each picture (a few clips
              per day free; falls back to slow camera moves when used up)
  video    -> ffmpeg: clips + voices + big subtitles

No lip-sync (that needs the paid agent.py).
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

STORIES = yaml.safe_load((ROOT / "stories.yaml").read_text())


def copy(obj):
    return json.loads(json.dumps(obj))


def made_titles(out_dir):
    titles = set()
    for f in Path(out_dir).glob("*/plan.json"):
        try:
            titles.add(json.loads(f.read_text()).get("title"))
        except Exception:
            pass
    return titles


def pick_story(idea, out_dir):
    """A ready-made story: the one matching the idea, or the next one not made yet."""
    if idea:
        key = idea.lower().strip(" .!")
        return next((copy(st) for st in STORIES if st["idea"].lower() == key), None)
    done = made_titles(out_dir)
    fresh = [st for st in STORIES if st["title"] not in done]
    return copy(random.choice(fresh) if fresh else random.choice(STORIES))


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
    return json.loads(m.group(0) if m else text, strict=False)


def llm_local(prompt, model_id):
    """Small open-source AI that runs right here on the CPU: free, no account."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    example = json.dumps(STORIES[0], ensure_ascii=False)
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


def write_plan(cfg, cast, idea, n_scenes, out_dir):
    story = pick_story(idea, out_dir)
    if story:
        log(f"Using ready-made story: {story['title']}")
        return validate_plan(story, cast, len(story["scenes"]))
    language = cfg.get("language", "English")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import agent  # the paid agent's Claude writer
            log("Writing the story with Claude...")
            return validate_plan(agent.write_plan(cfg, cast, idea, n_scenes), cast, n_scenes)
        except Exception as e:  # fall through to the free writer
            log(f"Claude unavailable ({e}); using the free writer")

    prompt = story_prompt(cast, idea, n_scenes, language)
    model_id = cfg.get("free_story_model", "Qwen/Qwen2.5-3B-Instruct")
    best = None
    for attempt in range(3):
        try:
            log(f"Writing the story with a free AI on this computer ({model_id})...")
            plan = validate_plan(parse_json(llm_local(prompt, model_id)), cast, n_scenes)
            plan["idea"] = idea
            if len(plan["scenes"]) == n_scenes:
                return plan
            log(f"  got {len(plan['scenes'])} scenes instead of {n_scenes}, trying again")
            best = best or plan
        except Exception as e:
            log(f"  attempt {attempt + 1} failed: {e}")
    if best:
        return best
    sys.exit("[free-agent] The free story writer could not write a story. Please try again.")


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------

def scene_prompt(cfg, cast, scene):
    looks = " ".join(f"{cast['characters'][c]['name']}: {cast['characters'][c]['look'].strip()}"
                     for c in scene["characters"])
    n = len(scene["characters"])
    count = "Only one character in the picture." if n == 1 else f"Exactly {n} different characters in the picture, each appearing once, no one else."
    return f"{cfg['style'].strip()} {scene['image_prompt']} {count} Characters: {looks}"


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

    token = os.environ.get("HF_TOKEN")
    client = Client("black-forest-labs/FLUX.1-schnell", **({"token": token} if token else {}))
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


# --------------------------------------------------------------------------
# Animation (free Hugging Face "ZeroGPU" spaces running Wan 2.2)
# --------------------------------------------------------------------------

DEFAULT_VIDEO_SPACES = [
    "zerogpu-aoti/wan2-2-fp8da-aoti-faster",
    "r3gm/wan2-2-fp8da-aoti-preview",
    "Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom",
]
_clients = {}
_quota_used_up = False


def find_mp4(result):
    if isinstance(result, str):
        return result if result.lower().endswith((".mp4", ".webm", ".mov")) else None
    if isinstance(result, dict):
        result = list(result.values())
    if isinstance(result, (list, tuple)):
        for item in result:
            found = find_mp4(item)
            if found:
                return found
    return None


def animate(cfg, scene, img, dest, seconds):
    """Turn the still picture into a moving clip. Returns True on success."""
    global _quota_used_up
    if _quota_used_up or not cfg.get("free_animation", True):
        return False
    from gradio_client import Client, handle_file

    token = os.environ.get("HF_TOKEN")
    prompt = (f"3D Pixar-style animation. {scene.get('action', '')} Smooth natural lively motion, "
              "expressive faces, characters keep the same look, steady camera.")
    for space in cfg.get("free_video_spaces", DEFAULT_VIDEO_SPACES):
        try:
            if space not in _clients:
                _clients[space] = Client(space, **({"token": token} if token else {}))
            job = _clients[space].submit(input_image=handle_file(str(img)), prompt=prompt,
                                         duration_seconds=seconds, api_name="/generate_video")
            path = find_mp4(job.result(timeout=420))
            if not path:
                raise RuntimeError("no video returned")
            shutil.copy(path, dest)
            log(f"  animated with {space}")
            return True
        except Exception as e:
            msg = str(e)
            log(f"  animation via {space} failed: {msg[:200]}")
            if "quota" in msg.lower():
                _quota_used_up = True  # the free daily allowance is shared by all spaces
                log("  free daily animation allowance used up; remaining scenes use camera moves")
                return False
    return False


# --------------------------------------------------------------------------
# Scene assembly
# --------------------------------------------------------------------------

def subtitle_filters(cast, cues, idx, folder):
    vf = []
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
    return vf


def build_scene(cfg, cast, scene, idx, img, folder):
    # 1) voices, laid out on a timeline
    parts, cues, t = [], [], LEAD
    for j, d in enumerate(scene["dialogue"]):
        mp3 = folder / f"s{idx}_line{j}.mp3"
        make_voice(cast, d["speaker"], d["line"], mp3)
        dur = duration(mp3)
        parts.append((mp3, t))
        cues.append((d, t, t + dur + 0.25))
        t += dur + GAP
    voice_total = max(t - GAP + TAIL, 3.5) if parts else 4.0

    # 2) try to animate the picture; fall back to a slow camera move
    clip = folder / f"s{idx}_anim.mp4"
    want = round(min(max(voice_total, 3.5), cfg.get("free_clip_seconds", 5.0)), 1)
    log("  animating (free)...")
    animated = animate(cfg, scene, img, clip, want)

    if animated:
        cd = duration(clip)
        total = max(voice_total, cd)
        slow = min(total / cd, 1.3)  # stretch motion a little if the talking runs long
        hold = max(total - cd * slow, 0)
        vf = [f"setpts=PTS*{slow:.3f}", f"fps={FPS}",
              f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"]
        if hold > 0.01:
            vf.append(f"tpad=stop_mode=clone:stop_duration={hold:.3f}")
        src = ["-i", clip]
    else:
        total = voice_total
        frames = int(total * FPS)
        z, x, y = MOVES[idx % len(MOVES)]
        vf = [f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2}",
              f"zoompan=z='{z}':x='{x.format(n=frames)}':y='{y.format(n=frames)}':d={frames}:s={W}x{H}:fps={FPS}"]
        src = ["-loop", "1", "-framerate", FPS, "-i", img]

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

    vf += subtitle_filters(cast, cues, idx, folder)
    vf += ["fade=t=in:st=0:d=0.25", f"fade=t=out:st={total - 0.25:.2f}:d=0.25", "format=yuv420p"]

    out = folder / f"scene_{idx + 1:02d}.mp4"
    ffmpeg(*src, "-i", audio, "-map", "0:v", "-map", "1:a", "-vf", ",".join(vf),
           "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
           "-c:a", "aac", "-b:a", "160k", out)
    return out, animated


def join(clips, dest, music_volume):
    listfile = dest.with_name("clips.txt")
    listfile.write_text("".join(f"file '{c.name}'\n" for c in clips))
    joined = dest.with_name("joined.mp4")
    # loudnorm: make voices as loud as typical Reels/Shorts audio
    ffmpeg("-f", "concat", "-safe", "0", "-i", listfile, "-c:v", "copy",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "44100", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", joined)
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
    p.add_argument("--demo", action="store_true", help="make the first ready-made story")
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cast = yaml.safe_load((ROOT / "characters.yaml").read_text())
    n_scenes = args.scenes or cfg.get("scenes_per_video", 3)
    idea = " ".join(args.idea).strip()
    if args.demo:
        idea = STORIES[0]["idea"]
    if not shutil.which("ffmpeg"):
        sys.exit("[free-agent] ffmpeg is not installed.")

    plan = write_plan(cfg, cast, idea, n_scenes, args.out)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(args.out) / f"{stamp}-{slugify(plan['title'])}"
    work = folder / "work"
    work.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    log(f"Title: {plan['title']}")

    seed = random.randint(1, 10**6)  # same seed for every scene helps faces stay similar
    clips, n_animated = [], 0
    for i, scene in enumerate(plan["scenes"]):
        log(f"Scene {i + 1}/{len(plan['scenes'])}: " +
            (" / ".join(f"{d['speaker']}: {d['line']}" for d in scene["dialogue"]) or "(no lines)"))
        img = folder / f"scene_{i + 1:02d}.jpg"
        log("  drawing picture (free)...")
        make_picture(cfg, cast, scene, seed, img)
        log("  adding voices and subtitles...")
        clip, was_animated = build_scene(cfg, cast, scene, i, img, work)
        clips.append(clip)
        n_animated += was_animated
        time.sleep(2)  # be polite to the free services

    final = folder / "final.mp4"
    join(clips, work / "final.mp4", cfg.get("music_volume", 0.12))
    shutil.move(str(work / "final.mp4"), final)
    shutil.rmtree(work)

    caption = (f"{plan['hook_caption']}\n\n{plan['description']}\n\n"
               + " ".join(h if h.startswith("#") else f"#{h}" for h in plan["hashtags"])
               + "\n\n(AI-generated content)")
    (folder / "caption.txt").write_text(caption)
    log(f"DONE -> {final}  ({duration(final):.1f}s, {n_animated}/{len(clips)} scenes animated)")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"folder={folder}\n")


if __name__ == "__main__":
    main()
