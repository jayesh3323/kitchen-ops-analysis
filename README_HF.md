---
title: Kitchen Ops Dashboard
emoji: 🍜
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Kitchen Ops Dashboard

AI-powered video analysis dashboard for kitchen operations monitoring.

## Features
- **Pork Weighing Compliance** — Detects weighing events and extracts scale readings
- **Ramen Plating Time** — Measures plating duration from video footage
- **Service Time Analysis** — Tracks customer service timing

## Tech Stack
- FastAPI + Uvicorn (backend)
- Firebase Firestore (persistent database)
- Langfuse (LLM observability & prompt management)
- Gemini & OpenAI APIs (vision models)
