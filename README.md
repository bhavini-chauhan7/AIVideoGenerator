# AI Video Generator: Little Pixels Family agent

Type one line. Get back a finished, vertical, Pixar-style family video with voices,
plus a caption ready to paste into Reels, Shorts or TikTok.

## 🆓 FREE version: no keys, no money, nothing to install

1. Open this repo on GitHub (the website or the GitHub app) and tap **Actions**.
2. Tap **Make a video (free)**, then **Run workflow**.
3. Type your idea, for example *daddy tries to braid Lily's hair*, and tap the green **Run workflow** button.
   Leave the idea **empty** to get the next ready-made story from `stories.yaml` (12 funny, hand-written stories).
4. Wait about 5 minutes. Your video appears in the **`videos/`** folder. Tap the newest folder, then `final.mp4`.
   The caption to paste is in `caption.txt` in the same folder.

The free version draws Pixar-style pictures, **animates** them (the characters run, eat, hug and fall over)
with free Hugging Face AI, and adds character voices (including a little girl's voice) and big coloured
subtitles. Each scene runs on its own free GitHub computer so it gets its own free animation allowance.
If that allowance runs out, the scene uses a slow camera move instead.
The characters don't move their lips; for lip-sync, use the paid version below.
Free AI is sometimes weird (a character drawn twice, or sliding out of the picture). If a scene looks odd,
just run it again.

Note: this repo is public, so the videos in `videos/` can be seen by anyone with the link.

---

## Paid version (real animation + lip-sync)

```
python agent.py make "baby tells daddy she'll tell on him to mommy"
```

## What the agent does

1. **Writes the story.** Claude turns your idea into 3 short scenes, each with a camera shot, an action and spoken lines.
2. **Draws each scene.** fal.ai (Nano Banana) draws each scene using your fixed characters, so Daddy and Lily look the same in every video.
3. **Animates with voices.** fal.ai (Veo 3) brings each picture to life as an 8-second clip with lip-synced dialogue.
4. **Edits.** ffmpeg joins the clips into a 1080×1920 MP4 and adds background music if you supply some.
5. **Writes the caption.** It saves `caption.txt` with a hook line, a description and hashtags.

Everything for a video lands in `output/<date>-<title>/`: `final.mp4`, `caption.txt`, and the scene images and clips.

## One-time setup (about 5 minutes)

1. Install Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html).
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and paste in two keys:
   - `ANTHROPIC_API_KEY` from https://console.anthropic.com/
   - `FAL_KEY` from https://fal.ai/dashboard/keys (add some credit)
4. Draw your cast once: `python agent.py characters`
   Open the pictures in `characters/`. If you don't like one, edit its description in
   `characters.yaml` and run `python agent.py characters --redo`. You can also drop in
   your own picture as `characters/lily.png` and so on.

## Daily use

| You want | Command |
|---|---|
| A video from your idea | `python agent.py make "daddy tries to braid Lily's hair"` |
| A video, agent picks the idea | `python agent.py make` |
| 5 videos while you sleep | `python agent.py auto 5 -y` |
| Just ideas | `python agent.py ideas` |
| Check the story before paying for video | `python agent.py make --plan-only "idea"` |
| Finish a video that got interrupted | `python agent.py make --resume output/<folder>` |

Before spending money the agent shows an estimated cost and asks first. Add `-y` to skip that question.

### Even less work: use it from Claude Code

Open this folder in Claude Code and type:

```
/video baby catches daddy eating her cookies
/video auto 3
/video
```

## Customising

- **`characters.yaml`** holds your cast's looks and voices. Add a grandma, a dog, anything.
- **`config.yaml`** sets the art style, scenes per video, dialogue language, which AI models to use, and costs.
- **`assets/music/`** is for royalty-free `.mp3` files. One is picked at random and mixed in quietly.

## Costs (rough)

Each 3-scene video (about 24 seconds) costs about **$3–4** on fal.ai. Almost all of that is the Veo 3
video step. Claude adds a few cents. For cheaper clips, change `video_model` in `config.yaml`, for example
to a Kling image-to-video model on fal. Most of those are silent, so you'd lose the voices.

## Tips for viral clips

- The format that works: a cute, sassy toddler line, then a big adult reaction, then a sweet ending.
- Keep the same characters and channel name so viewers recognise the family.
- Platforms require labelling AI content. The caption already includes "(AI-generated content)". Also turn on the platform's AI label when you post.
