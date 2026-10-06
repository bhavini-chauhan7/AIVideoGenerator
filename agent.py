#!/usr/bin/env python3
"""Little Pixels video agent: one command in, finished Pixar-style short out.

    python agent.py characters                 # once: draw your cast's reference pictures
    python agent.py make "baby tells on daddy"  # idea -> finished video
    python agent.py make                        # no idea? the agent invents one
    python agent.py auto 5                      # invent + make 5 videos in a row
    python agent.py ideas                       # just brainstorm 10 ideas

Pipeline: Claude writes the scene plan -> fal.ai draws each keyframe with your
characters -> fal.ai animates each keyframe with spoken dialogue -> ffmpeg
stitches a vertical 1080x1920 MP4 plus a ready-to-paste caption.
"""
import argparse
import datetime as dt
import json
import random
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CHAR_DIR = ROOT / "characters"
MUSIC_DIR = ROOT / "assets" / "music"
OUT_DIR = ROOT / "output"


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def load_yaml(name):
    return yaml.safe_load((ROOT / name).read_text())


def log(msg):
    print(f"[agent] {msg}", flush=True)


def slugify(text, n=40):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "video"


def download(url, dest):
    urllib.request.urlretrieve(url, dest)
    return dest


# --------------------------------------------------------------------------
# Claude: story writer
# --------------------------------------------------------------------------

def ask_claude(cfg, system, prompt, schema):
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=cfg["claude_model"],
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={
            "effort": cfg.get("claude_effort", "medium"),
            "format": {"type": "json_schema", "schema": schema},
        },
        # If a request is declined, the API retries it on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        sys.exit("[agent] Claude declined this idea. Try rephrasing it.")
    if resp.stop_reason == "max_tokens":
        sys.exit("[agent] Claude's answer was cut off. Try fewer scenes.")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def writer_system(cfg, cast):
    chars = "\n".join(
        f"- id `{cid}` — {c['name']}: {c['look'].strip()} Voice: {c['voice']}."
        for cid, c in cast["characters"].items()
    )
    return f"""You are the head writer of "{cast['channel_name']}", a viral short-video
channel of heart-warming, funny 3D Pixar-style family moments (Reels/Shorts/TikTok).

Recurring cast:
{chars}

What performs well: a relatable parenting moment, a cute toddler line that is
funny because it is a little bit sassy or mispronounced, an exaggerated adult
reaction, and a sweet "aww" ending. Clean, wholesome, family-friendly only.

Each scene is animated from ONE still image into an ~8 second clip with sound,
so per scene: one clear moment, one or two short spoken lines at most (under
~20 words total), simple motion. Dialogue is in {cfg.get('language', 'English')}.
Image prompts must fully describe the shot (who, pose, expression, setting,
camera framing) without relying on other scenes, and must not contain text."""


def plan_schema(char_ids):
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "hook_caption", "description", "hashtags", "scenes"],
        "properties": {
            "title": {"type": "string"},
            "hook_caption": {"type": "string", "description": "short on-post caption with emojis, like a quote of the funniest line"},
            "description": {"type": "string", "description": "1-2 sentence post description"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["characters", "image_prompt", "action", "dialogue"],
                    "properties": {
                        "characters": {"type": "array", "items": {"type": "string", "enum": char_ids}},
                        "image_prompt": {"type": "string"},
                        "action": {"type": "string", "description": "what moves / happens during the clip, camera motion"},
                        "dialogue": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["speaker", "line"],
                                "properties": {
                                    "speaker": {"type": "string", "enum": char_ids},
                                    "line": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def brainstorm(cfg, cast, n=10):
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ideas"],
        "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
    }
    data = ask_claude(
        cfg, writer_system(cfg, cast),
        f"Brainstorm {n} fresh, specific video ideas (one sentence each) that are "
        "likely to go viral. Vary settings and which characters appear.",
        schema,
    )
    return data["ideas"][:n]


def write_plan(cfg, cast, idea, n_scenes):
    log(f"Writing the story with Claude: {idea!r}")
    plan = ask_claude(
        cfg, writer_system(cfg, cast),
        f"Write a {n_scenes}-scene short video for this idea: {idea}\n"
        f"Return exactly {n_scenes} scenes.",
        plan_schema(list(cast["characters"])),
    )
    plan["idea"] = idea
    return plan


# --------------------------------------------------------------------------
# fal.ai: images and video
# --------------------------------------------------------------------------

def fal_run(model, args):
    import fal_client

    def on_update(status):
        if isinstance(status, fal_client.InProgress):
            for entry in status.logs or []:
                print("    " + entry.get("message", ""), flush=True)

    return fal_client.subscribe(model, arguments=args, with_logs=True, on_queue_update=on_update)


_upload_cache = {}


def fal_upload(path):
    import fal_client

    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key not in _upload_cache:
        _upload_cache[key] = fal_client.upload_file(path)
    return _upload_cache[key]


def char_ref(cid):
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = CHAR_DIR / f"{cid}.{ext}"
        if p.exists():
            return p
    return None


def make_characters(cfg, cast, redo=False):
    CHAR_DIR.mkdir(parents=True, exist_ok=True)
    for cid, c in cast["characters"].items():
        if char_ref(cid) and not redo:
            log(f"{c['name']}: reference exists ({char_ref(cid).name}), skipping")
            continue
        log(f"Drawing reference picture for {c['name']}...")
        prompt = (f"{cfg['style'].strip()} Character reference portrait, waist-up, "
                  f"facing camera, neutral friendly expression, plain soft background. "
                  f"{c['look'].strip()}")
        res = fal_run(cfg["character_model"], {"prompt": prompt, **cfg.get("character_args", {})})
        download(res["images"][0]["url"], CHAR_DIR / f"{cid}.png")
        log(f"  saved characters/{cid}.png")


def make_keyframe(cfg, cast, scene, dest):
    looks = " ".join(f"{cast['characters'][c]['name']}: {cast['characters'][c]['look'].strip()}"
                     for c in scene["characters"])
    prompt = f"{cfg['style'].strip()} {scene['image_prompt']} Characters: {looks}"
    refs = [char_ref(c) for c in scene["characters"] if char_ref(c)]
    if refs:
        prompt = ("Keep each character exactly identical to the reference images "
                  "(face, hair, outfit). " + prompt)
        args = {"prompt": prompt, "image_urls": [fal_upload(r) for r in refs], **cfg.get("image_args", {})}
        res = fal_run(cfg["image_model"], args)
    else:
        res = fal_run(cfg["character_model"], {"prompt": prompt, **cfg.get("character_args", {})})
    download(res["images"][0]["url"], dest)


def make_clip(cfg, cast, scene, keyframe, dest):
    lines = " ".join(
        f'{cast["characters"][d["speaker"]]["name"]} ({cast["characters"][d["speaker"]]["voice"]}) says: "{d["line"]}"'
        for d in scene["dialogue"]
    )
    prompt = (f"3D Pixar-style animation. {scene['action']} {lines} "
              "Natural lip-sync, expressive faces, gentle cinematic camera, "
              "soft ambient room sound, no background music, no subtitles.")
    args = {"prompt": prompt, "image_url": fal_upload(keyframe), **cfg.get("video_args", {})}
    res = fal_run(cfg["video_model"], args)
    download(res["video"]["url"], dest)


# --------------------------------------------------------------------------
# ffmpeg: stitch
# --------------------------------------------------------------------------

def has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(out.stdout.strip())


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def stitch(cfg, clips, dest):
    w, h = cfg["output_width"], cfg["output_height"]
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=30,format=yuv420p"
    normalized = []
    for clip in clips:
        norm = clip.with_name(clip.stem + "_norm.mp4")
        if has_audio(clip):
            ffmpeg("-i", clip, "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                   "-c:a", "aac", "-ar", "44100", "-ac", "2", norm)
        else:
            ffmpeg("-i", clip, "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest",
                   "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                   "-c:a", "aac", norm)
        normalized.append(norm)

    listfile = dest.with_name("clips.txt")
    listfile.write_text("".join(f"file '{p.name}'\n" for p in normalized))
    joined = dest.with_name("joined.mp4")
    ffmpeg("-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", joined)

    music = sorted(MUSIC_DIR.glob("*.mp3"))
    if music:
        track = random.choice(music)
        log(f"Adding background music: {track.name}")
        ffmpeg("-i", joined, "-stream_loop", "-1", "-i", track, "-filter_complex",
               f"[1:a]volume={cfg['music_volume']}[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", dest)
        joined.unlink()
    else:
        joined.rename(dest)
    listfile.unlink()
    for p in normalized:
        p.unlink()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def confirm_cost(cfg, n_scenes, yes):
    est = n_scenes * (cfg["cost_per_image"] + cfg["cost_per_clip"])
    log(f"Estimated fal.ai cost: ~${est:.2f} for {n_scenes} scenes (rough; see config.yaml)")
    if yes:
        return
    if input("[agent] Go ahead? [Y/n] ").strip().lower() in ("n", "no"):
        sys.exit("[agent] Cancelled.")


def make_video(cfg, cast, idea, n_scenes, yes=False, plan_only=False, resume=None):
    if resume:
        folder = Path(resume)
        plan = json.loads((folder / "plan.json").read_text())
        log(f"Resuming {folder}")
    else:
        if not idea:
            idea = brainstorm(cfg, cast, 1)[0]
            log(f"No idea given, the agent picked: {idea}")
        plan = write_plan(cfg, cast, idea, n_scenes)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = OUT_DIR / f"{stamp}-{slugify(plan['title'])}"
        folder.mkdir(parents=True)
        (folder / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    log(f"Title: {plan['title']}")
    for i, s in enumerate(plan["scenes"], 1):
        said = " / ".join(f"{d['speaker']}: {d['line']}" for d in s["dialogue"]) or "(no lines)"
        log(f"  Scene {i}: {said}")
    if plan_only:
        log(f"Plan saved to {folder / 'plan.json'} (no images/videos generated)")
        return folder

    missing_refs = [c for c in cast["characters"] if not char_ref(c)]
    if missing_refs:
        log(f"Tip: run `python agent.py characters` first for consistent faces (missing: {', '.join(missing_refs)})")

    confirm_cost(cfg, len(plan["scenes"]), yes)

    clips = []
    for i, scene in enumerate(plan["scenes"], 1):
        img = folder / f"scene_{i:02d}.png"
        clip = folder / f"scene_{i:02d}.mp4"
        if not img.exists():
            log(f"Scene {i}/{len(plan['scenes'])}: drawing keyframe...")
            make_keyframe(cfg, cast, scene, img)
        if not clip.exists():
            log(f"Scene {i}/{len(plan['scenes'])}: animating with voices (takes a few minutes)...")
            make_clip(cfg, cast, scene, img, clip)
        clips.append(clip)

    final = folder / "final.mp4"
    log("Stitching final video...")
    stitch(cfg, clips, final)

    caption = (f"{plan['hook_caption']}\n\n{plan['description']}\n\n"
               + " ".join(h if h.startswith("#") else f"#{h}" for h in plan["hashtags"])
               + "\n\n(AI-generated content)")
    (folder / "caption.txt").write_text(caption)
    log(f"DONE -> {final}")
    log(f"Caption -> {folder / 'caption.txt'}")
    return folder


def check_setup(need_fal=True):
    import os
    missing = [k for k in ("ANTHROPIC_API_KEY", "FAL_KEY") if not os.environ.get(k) and (need_fal or k != "FAL_KEY")]
    if missing:
        sys.exit(f"[agent] Missing {', '.join(missing)}. Copy .env.example to .env and fill it in.")
    if need_fal and not shutil.which("ffmpeg"):
        sys.exit("[agent] ffmpeg is not installed (https://ffmpeg.org/download.html).")


def main():
    load_env()
    cfg, cast = load_yaml("config.yaml"), load_yaml("characters.yaml")

    p = argparse.ArgumentParser(description="One-command Pixar-style family video agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("characters", help="draw reference pictures for your cast (run once)")
    c.add_argument("--redo", action="store_true", help="redraw even if pictures exist")

    m = sub.add_parser("make", help="make one video from an idea (or let the agent pick)")
    m.add_argument("idea", nargs="*", help="e.g. baby tells on daddy to mommy")
    m.add_argument("-n", "--scenes", type=int, default=cfg["scenes_per_video"])
    m.add_argument("-y", "--yes", action="store_true", help="don't ask before spending")
    m.add_argument("--plan-only", action="store_true", help="only write the story (cheap test)")
    m.add_argument("--resume", metavar="FOLDER", help="finish an interrupted video folder")

    a = sub.add_parser("auto", help="invent and make several videos in a row")
    a.add_argument("count", type=int, nargs="?", default=3)
    a.add_argument("-n", "--scenes", type=int, default=cfg["scenes_per_video"])
    a.add_argument("-y", "--yes", action="store_true")

    i = sub.add_parser("ideas", help="brainstorm video ideas")
    i.add_argument("count", type=int, nargs="?", default=10)

    args = p.parse_args()

    if args.cmd == "characters":
        check_setup()
        make_characters(cfg, cast, args.redo)
    elif args.cmd == "make":
        check_setup(need_fal=not args.plan_only)
        make_video(cfg, cast, " ".join(args.idea), args.scenes, args.yes, args.plan_only, args.resume)
    elif args.cmd == "auto":
        check_setup()
        ideas = brainstorm(cfg, cast, args.count)
        confirm_cost(cfg, args.count * args.scenes, args.yes)
        for n, idea in enumerate(ideas, 1):
            log(f"=== Video {n}/{len(ideas)}: {idea}")
            make_video(cfg, cast, idea, args.scenes, yes=True)
    elif args.cmd == "ideas":
        check_setup(need_fal=False)
        for n, idea in enumerate(brainstorm(cfg, cast, args.count), 1):
            print(f"{n:2d}. {idea}")


if __name__ == "__main__":
    main()
