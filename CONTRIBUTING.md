# Contributing to finetune-kit

Thanks for helping. This page covers how to propose a change, how to set up the project, and
the few rules every change must keep.

## Ways to help

- **Report a bug or rough edge:** open an issue with what you ran, what you expected and what
  happened. Paste the JSON the command printed, but never rows of your data.
- **Suggest a feature:** open an issue first and describe the problem it solves, so we can agree
  on the approach before you build it.
- **Send a fix or feature:** follow the steps below.
- **Report a security problem:** do not open a public issue. See [Security](#security).

## How a change gets in

1. **Fork** the repository (or, if you have write access, create a branch in it). Name the
   branch after the change: `fix/<short-name>`, `feat/<short-name>` or `docs/<short-name>`.
2. **Make the change** with tests (see below).
3. **Open a pull request** against `main`. Say what changed, why, and how you tested it.
4. **Review.** A maintainer reviews it. `main` is protected: nobody pushes to it directly, and
   only maintainers can merge. Expect questions; small, focused PRs get merged fastest.

Keep each PR to one change. A bug fix and an unrelated cleanup are two PRs.

## Set up the project

You need [uv](https://docs.astral.sh/uv/) and git. uv fetches Python 3.12 if needed.

```bash
git clone https://github.com/PhiGrad-AI/finetune-kit
cd finetune-kit
uv sync
```

Run the CLI from your checkout:

```bash
uv run finetune-kit --help
```

## Tests

Every change needs tests, and the whole suite must pass before review.

```bash
uv run pytest -q
```

Tests marked `model` download small Hugging Face models and train for real, so they are the slow
part. While you work, skip them:

```bash
uv run pytest -q -m "not model"
```

Then run the full suite once before you open the PR.

- New behaviour gets a test that fails without your change.
- A bug fix gets a test that reproduces the bug.
- Step prompts (`src/finetune_kit/templates/commands/*.md`) are tested too
  (`tests/test_installer.py`): if you change a question or a command in a prompt, update the test
  that checks it.

## Where things are

- `src/finetune_kit/cli.py`: the commands. Each prints exactly one JSON object.
- `src/finetune_kit/commands.py`, `experiment.py`, `data.py`, `sources.py`, `queue.py`,
  `promote.py`: the logic behind them.
- `src/finetune_kit/templates/commands/`: the step prompts the agent follows, one file per step.
  `_rules.md`, `_goto.md` and `_finish.md` are shared blocks included in every step.
- `src/finetune_kit/templates/memory/research-rules.md`: the rules the agent follows during
  research.
- `src/finetune_kit/profiles/models.yaml`: the shipped model profiles.
- `docs/design/specs/` and `docs/design/plans/`: design documents for larger features.

## Rules every change must keep

These are what make finetune-kit's results trustworthy. A PR that weakens one will not be
merged without a design discussion first.

- **The test set is touched once.** Only the final scoring (`promote`, run by the *Finish* step)
  reads the test split in the vault. Nothing else may read it.
- **The agent never reads data rows.** Commands report counts, column names and statistics,
  never example text. Keep it that way in new commands and error messages.
- **The CLI decides, not the agent.** The keep verdict, hard limits, budget, seeds and the
  tamper checks are enforced in code. Don't move any of them into prompt text.
- **One JSON object per command** on standard output (except `init`).
- **Data stays on the machine.** No telemetry, no uploads, no network calls beyond downloading
  models and datasets the user asked for.
- **Pinned models.** Shipped profiles pin an exact Hugging Face revision; a profile that runs
  remote code must be pinned.

## Larger changes

For a new step, command, loss, metric or anything that changes how research decides, open an
issue first. If we agree to build it, add a short design document to `docs/design/specs/`
(what and why, what it must not break, the approach) before or with the code.

## Style

- Follow the style of the code around your change; keep functions small and named plainly.
- Prompts and docs are written for people who are not ML engineers: short sentences, plain
  words, and explain a term the first time it appears.
- Commit messages: a short summary line in the form `fix: ...`, `feat: ...` or `docs: ...`,
  then a blank line and the why, if it isn't obvious.
- Don't add tool-generated attribution lines (co-author trailers or "generated with" notes) to
  commits, PR descriptions or files.
- Never commit data, model weights, tokens or `.env` files. Experiments live in `experiments/`,
  which git ignores. Secret scanning blocks pushes that contain tokens.

## Security

To report a vulnerability, use GitHub's private reporting: the **Security** tab of this
repository → **Report a vulnerability**. Please don't open a public issue for it.
