---
description: Make a Pixar-style family short from an idea (or let the agent invent one)
argument-hint: [idea, or "auto 3", or "ideas"]
---
You are operating the video agent in this repo (`agent.py`). The user's request: $ARGUMENTS

1. If `.env` is missing or lacks ANTHROPIC_API_KEY / FAL_KEY, tell the user to copy `.env.example` to `.env` and fill it in, then stop.
2. If any character in `characters.yaml` has no picture in `characters/`, run `python agent.py characters` first.
3. Run the matching command:
   - request is empty → `python agent.py make -y`
   - request starts with "auto" → `python agent.py auto <count> -y`
   - request is "ideas" → `python agent.py ideas`
   - otherwise → `python agent.py make -y "<request>"`
   Video generation takes several minutes. Run it in the background and wait for it to finish.
4. When it finishes, show the path to `final.mp4` and paste the contents of `caption.txt`.
   If it failed partway, fix the cause and rerun with `python agent.py make --resume <folder> -y`.
