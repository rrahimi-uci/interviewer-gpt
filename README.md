# interviewer-gpt

**Guru.AI** — a GPT-powered mock-interview assistant for ML Engineering, Leadership/Managerial,
and Coding interviews. It generates a random interview question, lets you answer by typing
(or by recording audio that is transcribed locally with Whisper), and then evaluates your
answer with an LLM. For leadership/behavioural questions it also scores how closely your
answer maps to the Amazon Leadership Principles using embedding similarity.

The app is built with [Gradio](https://www.gradio.app/) and
[LangChain](https://python.langchain.com/) on top of the OpenAI API.

## Repository structure

- `interviewer_app.py` — the Gradio app (entry point).
- `utility/`
  - `helpers.py` — database, embedding/similarity, summarization, transcript and threading helpers.
  - `prompts.py` — the prompt templates used for each question type.
- `resources/`
  - `interview_questions.db` — SQLite database that the question tables are loaded into.
  - `sample_behavioral_questions.txt` — behavioural/leadership questions (one per line). Add more lines to extend the pool.
  - `sample_coding_questions.txt` — coding questions (one per line).
  - `sample_ML_System_Design_questions.txt` — ML system-design questions (one per line).
  - `leadership_principles.json` — the Amazon Leadership Principles used to evaluate behavioural answers.
  - `random_story.txt` — an unrelated story used as a similarity baseline.
  - `interview_youtube.txt` — a sample transcript used by the test.
- `tests/`
  - `test_youtube.py` — an end-to-end script that pulls a YouTube mock-interview transcript and runs the evaluation pipeline against it.
- `.env.example` — template for the local configuration (copy to `.env`).

> **Note:** the question text files seed `interview_questions.db` automatically the first time a
> question table is empty. The database ships pre-seeded and deduplicated, so to pick up questions
> you add to the `.txt` files, delete `resources/interview_questions.db` and run the app again
> (see [Adding your own questions](#adding-your-own-questions)).

## Getting started

This project targets **Python 3.10–3.12** (verified on 3.12). Some pinned dependencies do not
yet ship wheels for Python 3.14.

1. Clone the repository.

2. Create and activate a virtual environment:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create your local config from the template and add your OpenAI API key:

   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY=sk-...
   ```

5. Run the app:

   ```bash
   python interviewer_app.py
   ```

6. Open <http://127.0.0.1:7860> in your browser.

   To expose a temporary public share link, set `GRADIO_SHARE=1` before launching.

## Configuration

All settings are read from a local, git-ignored `.env` file (see `.env.example`):

| Variable          | Purpose                                              | Default          |
| ----------------- | ---------------------------------------------------- | ---------------- |
| `OPENAI_API_KEY`  | Your OpenAI API key (required for any LLM call)      | —                |
| `MODEL`           | OpenAI chat model used for evaluation/summarization  | `gpt-3.5-turbo`  |
| `TEMPERATURE`     | Sampling temperature                                 | `0.0`            |
| `MAX_TOKENS`      | Max tokens for generations                           | `3000`           |
| `CHUNK_SIZE`      | Char threshold/size for splitting long responses     | `600`            |
| `CHUNK_OVERLAP`   | Overlap between text chunks when summarizing         | `30`             |

## Adding your own questions

Each question type is backed by a plain-text file in `resources/` (one question per line):

- Coding → `sample_coding_questions.txt`
- Leadership/Behavioural → `sample_behavioral_questions.txt`
- ML System Design → `sample_ML_System_Design_questions.txt`

Add your questions on new lines, then rebuild the database so the new entries are picked up:

```bash
rm resources/interview_questions.db
python interviewer_app.py   # the tables are re-seeded from the .txt files on first use
```

Questions are inserted with parameterized queries, so apostrophes and quotes are handled safely.

## How it works

1. Choose the question type (Leadership & Behavioural, ML System Design, or Coding).
2. Click **Generate me a random interview question**.
3. Type your answer — or, for behavioural questions, record audio and it will be transcribed
   locally with the `openai/whisper-base.en` model. (The model downloads on first use.)
4. Click **Guru evaluation of the response** to get:
   - a high-level ranking (Weak / Average / Good / Excellent) with an explanation,
   - for behavioural questions, a per-principle breakdown against the Amazon Leadership
     Principles plus a similarity bar chart, and
   - a sample "Guru" answer for reference.

## Running the test

`tests/test_youtube.py` is an end-to-end script (not a unit test). It downloads a YouTube
transcript, summarizes it, and runs the behavioural evaluation, writing results into the
`tests/` directory. It requires a valid `OPENAI_API_KEY` and network access:

```bash
python tests/test_youtube.py
```

## Security note

Never commit secrets. The `.env` file is git-ignored. An earlier version of this repository
committed an OpenAI key in `resources/conf.env`; that file is no longer tracked and any key it
contained should be considered compromised and revoked at
<https://platform.openai.com/api-keys>.

## Maintenance notes

The codebase was modernized to current libraries and verified end-to-end (UI build, server boot,
live OpenAI call, and the full evaluation path):

- **LangChain** upgraded from `0.0.327` to the split packages (`langchain` 0.3,
  `langchain-openai`, `langchain-text-splitters`); calls use `.invoke()` instead of the removed
  `.predict()` / `llm(...)` / `.run()` APIs.
- **Gradio** upgraded from `4.0.2` to `5.x` (`gr.update`, `Audio(sources=...)`, removed `.style()`
  and `launch(width/height)`); the server launch is guarded under `if __name__ == "__main__"`.
- **youtube-transcript-api** updated to the v1 instance API (`YouTubeTranscriptApi().fetch(...)`).
- Chat models and the Whisper transcriber are built lazily, so the UI starts without an API key
  or a model download.
- Configuration moved from `resources/conf.env` to a git-ignored `.env`.
- Fixed a database bug where every question request re-appended the whole `.txt` file (the table
  is now seeded only when empty), and switched inserts to parameterized queries.
