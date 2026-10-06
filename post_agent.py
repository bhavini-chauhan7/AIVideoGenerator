#!/usr/bin/env python3
"""
FREE motivational carousel agent (hand-drawn picture + handwritten quote).

  python post_agent.py                      # next ready-made post from motivation.yaml
  python post_agent.py "never give up"      # a new post about your theme
  python post_agent.py --layout-test        # check the layout with blank pictures (no internet)

  writing  -> motivation.yaml, or Claude (if ANTHROPIC_API_KEY is set), or a free AI on this computer
  pictures -> free Hugging Face FLUX space (fallback: free Pollinations image API)
  slides   -> Pillow: 1080x1350 Instagram slides with a handwritten quote
  reel     -> ffmpeg: the same slides as a 1080x1920 slideshow video
"""
import argparse
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
HAND_FONT = ROOT / "assets" / "fonts" / "PatrickHand-Regular.ttf"
MUSIC_DIR = ROOT / "assets" / "music"
W, H = 1080, 1350                     # Instagram portrait (4:5)
PIC_W, PIC_H = 1024, 768              # picture size asked from the AI
INK = (28, 28, 28)
PAPER = (255, 255, 255)
UA = {"User-Agent": "MotivationPostAgent/1.0"}
POSTS = yaml.safe_load((ROOT / "motivation.yaml").read_text())

STYLE = ("Hand-drawn illustration in coloured pencil and wax crayon on clean white paper. "
         "Visible sketchy textured strokes, soft scribbled shadow under the subject, vivid but "
         "limited colours, charming children's book feel. One simple centered scene with lots "
         "of empty white space around it and a plain pure white background. "
         "No text, no letters, no words, no border, no frame.")


def log(msg):
    print(f"[post-agent] {msg}", flush=True)


def slugify(text, n=40):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "post"


# --------------------------------------------------------------------------
# Writing the post
# --------------------------------------------------------------------------

def made_titles(out_dir):
    titles = set()
    for f in Path(out_dir).glob("*/post.json"):
        try:
            titles.add(json.loads(f.read_text()).get("title"))
        except Exception:
            pass
    return titles


def post_prompt(theme, n_slides):
    return f"""Write a viral Instagram motivational carousel about: {theme}.
It has a cover and {n_slides} slides. Each slide is ONE simple, symbolic, hand-drawn
picture (nature, animals, small objects, e.g. ants marching past a giant candy for
"focus") plus ONE short original quote (under 14 words) that the picture explains.
The cover has a picture and a short hook line that makes people swipe.
"scene" describes only the picture, in one sentence, and never contains words or text.

Reply with ONLY a JSON object, no markdown, in exactly this shape:
{{"title": str, "cover": {{"scene": str, "text": str}}, "caption": str (with emojis),
 "hashtags": [str], "slides": [{{"scene": str, "quote": str}}]}}"""


POST_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "cover": {"type": "object", "properties": {"scene": {"type": "string"}, "text": {"type": "string"}},
                  "required": ["scene", "text"], "additionalProperties": False},
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "slides": {"type": "array", "items": {
            "type": "object", "properties": {"scene": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["scene", "quote"], "additionalProperties": False}},
    },
    "required": ["title", "cover", "caption", "hashtags", "slides"],
    "additionalProperties": False,
}


def llm_local(prompt, model_id):
    """Small open-source AI that runs right here on the CPU: free, no account."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    example = json.dumps(POSTS[0], ensure_ascii=False)
    msgs = [
        {"role": "system", "content": "You write short, powerful, original motivational quotes. "
                                      "You answer with valid JSON only."},
        {"role": "user", "content": prompt + "\n\nExample of the exact JSON format (write a NEW post, "
                                             "do not copy this one):\n" + example},
    ]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1100, do_sample=True, temperature=0.8, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def parse_json(text):
    m = re.search(r"\{.*\}", text.strip(), re.S)
    return json.loads(m.group(0) if m else text, strict=False)


def validate(post, n_slides):
    slides = [{"scene": s["scene"].strip(), "quote": s["quote"].strip()}
              for s in post["slides"] if s.get("scene", "").strip() and s.get("quote", "").strip()]
    if not slides:
        raise ValueError("post has no slides")
    post["slides"] = slides[:n_slides]
    post.setdefault("hashtags", [])
    post.setdefault("caption", post.get("title", ""))
    if not post.get("cover") or not post["cover"].get("text"):
        post["cover"] = None
    return post


def write_post(cfg, theme, n_slides, out_dir):
    if not theme:
        done = made_titles(out_dir)
        fresh = [p for p in POSTS if p["title"] not in done]
        post = json.loads(json.dumps(random.choice(fresh) if fresh else random.choice(POSTS)))
        log(f"Using ready-made post: {post['title']}")
        return validate(post, len(post["slides"]))
    prompt = post_prompt(theme, n_slides)

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import agent  # the paid agent's Claude helper
            log("Writing the post with Claude...")
            system = "You write viral, original, heartfelt motivational Instagram carousels."
            return validate(agent.ask_claude(cfg, system, prompt, POST_SCHEMA), n_slides)
        except (Exception, SystemExit) as e:  # fall through to the free writer
            log(f"Claude unavailable ({e}); using the free writer")

    model_id = cfg.get("free_story_model", "Qwen/Qwen2.5-3B-Instruct")
    for attempt in range(3):
        try:
            log(f"Writing the post with a free AI on this computer ({model_id})...")
            return validate(parse_json(llm_local(prompt, model_id)), n_slides)
        except Exception as e:
            log(f"  attempt {attempt + 1} failed: {e}")
    sys.exit("[post-agent] The free writer could not write a post. Please try again.")


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------

def image_huggingface(prompt, seed, dest):
    from gradio_client import Client

    token = os.environ.get("HF_TOKEN")
    client = Client("black-forest-labs/FLUX.1-schnell", **({"token": token} if token else {}))
    result = client.predict(prompt=prompt[:1800], seed=seed, randomize_seed=False, width=PIC_W,
                            height=PIC_H, num_inference_steps=4, api_name="/infer")
    path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(path, dict):
        path = path.get("path") or path.get("url")
    shutil.copy(path, dest)


def image_pollinations(prompt, seed, dest):
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:1800]) +
           f"?width={PIC_W}&height={PIC_H}&seed={seed}&model=flux&nologo=true&private=true")
    r = requests.get(url, headers=UA, timeout=240)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/") or len(r.content) < 10_000:
        raise RuntimeError(f"not an image ({r.headers.get('content-type')}, {len(r.content)} bytes)")
    dest.write_bytes(r.content)


def draw_picture(scene, seed, dest):
    prompt = f"{STYLE} Scene: {scene}"
    for name, fn in (("Hugging Face", image_huggingface), ("Pollinations", image_pollinations)):
        for attempt in range(3):
            try:
                fn(prompt, seed, dest)
                Image.open(dest).convert("RGB").save(dest, "PNG")
                return
            except Exception as e:
                log(f"  {name} picture attempt {attempt + 1} failed: {e}")
                time.sleep(8 * (attempt + 1))
    sys.exit("[post-agent] Could not get a picture from any free service. Try again later.")


def blank_picture(scene, dest):
    """Grey placeholder, used by --layout-test."""
    img = Image.new("RGB", (PIC_W, PIC_H), (236, 236, 236))
    ImageDraw.Draw(img).text((40, 40), textwrap.fill(scene, 50), fill=(120, 120, 120),
                             font=font(30))
    img.save(dest, "PNG")


# --------------------------------------------------------------------------
# Slides
# --------------------------------------------------------------------------

def font(size):
    try:
        return ImageFont.truetype(str(HAND_FONT), size)
    except OSError:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def fit_text(draw, text, box_w, box_h, max_size, min_size=40):
    """Biggest font size (and line wrap) at which the text fits the box."""
    for size in range(max_size, min_size - 1, -2):
        f = font(size)
        words, lines, cur = text.split(), [], ""
        for w in words:
            test = f"{cur} {w}".strip()
            if draw.textlength(test, font=f) <= box_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        line_h = int(size * 1.08)
        if line_h * len(lines) <= box_h and all(draw.textlength(l, font=f) <= box_w for l in lines):
            return f, lines, line_h
    return f, lines, line_h


def draw_lines(draw, lines, f, line_h, top, box_h):
    y = top + (box_h - line_h * len(lines)) // 2
    for line in lines:
        x = (W - draw.textlength(line, font=f)) / 2
        # a thin outline makes the handwriting look like a felt-tip marker
        draw.text((x, y), line, font=f, fill=INK, stroke_width=1, stroke_fill=INK)
        y += line_h
    return y


def draw_ornament(draw, y):
    """A small ── ∞ ── flourish under the quote."""
    cx, half = W // 2, 150
    draw.line((cx - half, y, cx - 34, y), fill=INK, width=3)
    draw.line((cx + 34, y, cx + half, y), fill=INK, width=3)
    for dx in (-12, 12):
        draw.ellipse((cx + dx - 13, y - 9, cx + dx + 13, y + 9), outline=INK, width=3)
    draw.ellipse((cx - 4, y - 4, cx + 4, y + 4), fill=INK)


def paste_picture(canvas, pic_path, top, height):
    """Place the drawing on the white page with soft edges so it blends into the paper."""
    pic = Image.open(pic_path).convert("RGB")
    scale = min(W / pic.width, height / pic.height)
    pic = pic.resize((int(pic.width * scale), int(pic.height * scale)), Image.LANCZOS)
    mask = Image.new("L", pic.size, 0)
    m = 40
    ImageDraw.Draw(mask).rectangle((m, m, pic.width - m, pic.height - m), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(m // 2))
    canvas.paste(pic, ((W - pic.width) // 2, top + (height - pic.height) // 2), mask)


def footer(draw, handle):
    if handle:
        f = font(34)
        draw.text(((W - draw.textlength(handle, font=f)) / 2, H - 70), handle, font=f, fill=(150, 150, 150))


def make_slide(pic_path, quote, dest, handle):
    canvas = Image.new("RGB", (W, H), PAPER)
    paste_picture(canvas, pic_path, top=60, height=790)
    draw = ImageDraw.Draw(canvas)
    f, lines, line_h = fit_text(draw, quote, box_w=860, box_h=330, max_size=96)
    bottom = draw_lines(draw, lines, f, line_h, top=860, box_h=330)
    draw_ornament(draw, bottom + 50)
    footer(draw, handle)
    canvas.save(dest, "PNG")


def make_cover(pic_path, text, dest, handle):
    canvas = Image.new("RGB", (W, H), PAPER)
    paste_picture(canvas, pic_path, top=50, height=640)
    draw = ImageDraw.Draw(canvas)
    f, lines, line_h = fit_text(draw, text, box_w=900, box_h=400, max_size=100)
    bottom = draw_lines(draw, lines, f, line_h, top=700, box_h=400)
    draw_ornament(draw, bottom + 50)
    swipe = "swipe"
    fs = font(46)
    sw = draw.textlength(swipe, font=fs)
    x, y = (W - sw - 60) / 2, H - 140
    draw.text((x, y), swipe, font=fs, fill=(90, 90, 90))
    ax, ay = x + sw + 16, y + 30  # a little hand-drawn arrow
    draw.line((ax, ay, ax + 44, ay), fill=(90, 90, 90), width=4)
    draw.line((ax + 30, ay - 12, ax + 44, ay, ax + 30, ay + 12), fill=(90, 90, 90), width=4)
    footer(draw, handle)
    canvas.save(dest, "PNG")


# --------------------------------------------------------------------------
# Reel (optional slideshow video of the same slides)
# --------------------------------------------------------------------------

def make_reel(slides, dest, seconds, music_volume):
    if not shutil.which("ffmpeg"):
        log("ffmpeg not installed, skipping the reel")
        return
    inputs, parts = [], []
    for i, s in enumerate(slides):
        inputs += ["-loop", "1", "-t", str(seconds), "-i", str(s)]
        parts.append(f"[{i}:v]scale=1080:1350,pad=1080:1920:0:285:white,setsar=1,fps=30,"
                     f"fade=t=in:st=0:d=0.4:color=white,fade=t=out:st={seconds - 0.4}:d=0.4:color=white[v{i}]")
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(slides))) + \
        f"concat=n={len(slides)}:v=1:a=0[v]"
    music = sorted(MUSIC_DIR.glob("*.mp3")) if MUSIC_DIR.exists() else []
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(random.choice(music))]
        graph += f";[{len(slides)}:a]volume={max(music_volume, 0.5)},afade=t=out:st={seconds * len(slides) - 1.5}:d=1.5[a]"
        cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-filter_complex", graph, "-map", "[v]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="FREE motivational carousel agent")
    p.add_argument("theme", nargs="*")
    p.add_argument("-n", "--slides", type=int, default=None, help="slides for your own theme")
    p.add_argument("--out", default="output", help="folder to put posts in")
    p.add_argument("--layout-test", action="store_true", help="grey pictures, no internet needed")
    p.add_argument("--no-reel", action="store_true", help="don't make the slideshow video")
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    theme = " ".join(args.theme).strip()
    n_slides = args.slides or cfg.get("post_slides", 8)
    handle = cfg.get("post_handle", "") or ""

    post = write_post(cfg, theme, n_slides, args.out) if not args.layout_test else validate(
        json.loads(json.dumps(POSTS[0])), 99)
    post["seed"] = random.randint(1, 10**6)
    post["theme"] = theme
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(args.out) / f"{stamp}-{slugify(post['title'])}"
    pics = folder / "pictures"
    pics.mkdir(parents=True)

    pages = ([("cover", post["cover"]["scene"], post["cover"]["text"])] if post.get("cover") else []) + \
        [("quote", s["scene"], s["quote"]) for s in post["slides"]]
    slides = []
    for i, (kind, scene, text) in enumerate(pages, 1):
        log(f"Slide {i}/{len(pages)}: {text}")
        pic = pics / f"picture_{i:02d}.png"
        if args.layout_test:
            blank_picture(scene, pic)
        else:
            draw_picture(scene, post["seed"] + i, pic)
            time.sleep(2)  # be polite to the free services
        dest = folder / f"slide_{i:02d}.png"
        (make_cover if kind == "cover" else make_slide)(pic, text, dest, handle)
        slides.append(dest)

    if not args.no_reel:
        log("Making the reel version...")
        make_reel(slides, folder / "reel.mp4", cfg.get("post_seconds_per_slide", 3.0),
                  cfg.get("music_volume", 0.12))

    (folder / "post.json").write_text(json.dumps(post, indent=2, ensure_ascii=False))
    caption = (f"{post['caption']}\n\n"
               + "\n".join(f"{i}. {s['quote']}" for i, s in enumerate(post["slides"], 1)) + "\n\n"
               + " ".join(h if h.startswith("#") else f"#{h}" for h in post["hashtags"]))
    (folder / "caption.txt").write_text(caption)
    log(f"DONE -> {folder}  ({len(slides)} slides)")
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"folder={folder}\n")


if __name__ == "__main__":
    main()
