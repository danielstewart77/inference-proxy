# Inference Proxy

One endpoint in front of every upstream model, with its own keys.

Apps and agents authenticate with a proxy key this service issues. The proxy
holds the real upstream credentials, picks the right one per model, translates
between the OpenAI and Anthropic request shapes, and records what each caller
spent. A client never sees an upstream key, and rotating one is a single row
edit rather than a redeploy of everything that talks to it.

```
  frame-main, skippy, any future app
        |
        |  proxy key  (hmp-…)
        v
  inference proxy  ──  clients · credentials · models · usage
        |
        |  the real upstream credential
        v
  api.anthropic.com · api.openai.com · Azure OpenAI · Azure AI Foundry
```

## Concepts

**Clients** are apps and agents, not people. Each has a name, a kind
(`app`, `agent`, `mind`, `human`), and any number of proxy keys, so a key can
be rotated without downtime. Only the SHA-256 hash of a key is stored; the
plaintext is shown once at mint time and never again. A client marked
*privileged* may additionally resolve models flagged admin-only.

**Credentials** are the upstream secrets, shared across models:

| Kind | Secret comes from | Notes |
|---|---|---|
| `static` | a literal API key stored on the row | Azure, Foundry, real OpenAI |
| `anthropic_oauth` | a Claude CLI credential file, or a literal long-lived token in `secret` | file tokens are short-lived and refreshed automatically; a stored `secret` wins |
| `openai_oauth` | a Codex CLI auth file | the subscription OAuth token (refreshed automatically, rotated tokens written back), or the file's plain `OPENAI_API_KEY` when present |

**Models** are deployments. Each row carries its own target URI, the credential
to authenticate with, and an `auth_scheme` (`bearer`, `x_api_key`, or
`api_key_header`) saying how the secret is sent. The target URI's host and path
decide the upstream protocol, so one proxy fronts Anthropic Messages, the
OpenAI Responses API, and plain Chat Completions at the same time.

### Anthropic subscription OAuth

An Anthropic OAuth token is scoped to the Claude Code client. The Messages API
only honours one when the request also carries the `oauth-2025-04-20` beta flag
and declares that client identity in its first system block — without them it
answers `429`, not `401`. A model backed by an `anthropic_oauth` credential
gets both added automatically, and anything the caller sent as `system` is
preserved after the identity block.

### OpenAI subscription OAuth

Codex OAuth tokens address the ChatGPT backend rather than `api.openai.com`,
so a model backed by one targets
`https://chatgpt.com/backend-api/codex/responses` — Responses API shape, but
streaming-only and requiring the caller's ChatGPT identity. The proxy adds the
`chatgpt-account-id` header (read from the token's own JWT claims) plus the
`OpenAI-Beta: responses=experimental` and `originator` headers automatically.
When the auth file carries a plain `OPENAI_API_KEY` instead, that key is used
and models can target `api.openai.com` directly.

## Running it

```bash
cp .env.example .env        # set ADMIN_USERNAME, ADMIN_PASSWORD, SESSION_SECRET
pip install -r requirements.txt
python -m app.entrypoint
```

The schema is created on first start — SQLite, one file at `DB_PATH`, no
database server to stand up. Then sign into `/login` and, under `/admin`,
add a credential, add a model that uses it, add a client, and mint that client
a key.

Docker is supported too, with `docker compose up -d --build`. The default
compose file is safe on a fresh clone: it persists `data/`, does not require a
`.env` file just to start, and does not mount any host credentials. Copy
`.env.example` to `.env` and replace the admin and session placeholders before
exposing the service beyond localhost.

Subscription OAuth is an explicit opt-in because those files contain live
credentials and may not exist on every host:

```bash
docker compose -f docker-compose.yml -f docker-compose.oauth.yml up -d --build
```

Both credential files are mounted read-only and remain outside the repository.

Pick one or the other, never both. The container and a host process share the
same `data/` directory and the same published port, so a second deployment
cannot bind the port, and a container built before the newest migration will
crash-loop on `alembic upgrade head` against a database the host process has
already stamped forward. If the container restarts on a loop with
`Can't locate revision identified by '000N_…'`, it is running stale code:
rebuild it, or shut it down and leave the host process serving.

## Using it

Anything that speaks the Anthropic API points at the proxy with its proxy key:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8888
export ANTHROPIC_AUTH_TOKEN=hmp-…
```

Anything that speaks OpenAI does the same with `OPENAI_BASE_URL` and
`OPENAI_API_KEY`. Keys are accepted as `Authorization: Bearer`, `api-key`, or
`x-api-key`, so a client sends whichever its SDK already sends.

`GET /v1/models` lists the OpenAI-shaped deployments and
`GET /v1/anthropic/models` the Anthropic ones; both require a valid proxy key
and hide admin-only rows from unprivileged clients. `GET /health` is
unauthenticated, and `/status` is a public health and throughput page carrying
no per-client detail.

## Layout

```
app/
  main.py          application factory, model listing, catch-all logging
  auth.py          proxy-key cache + admin session
  credentials.py   upstream secret resolution and OAuth refresh
  deployments.py   per-model routing, protocol detection, header building
  orm.py           clients · keys · credentials · models · usage_log
  proxy/           anthropic · responses · chat_completions · compact · websocket
  admin/           the HTML console
migrations/        alembic, SQLite
tests/             pytest
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The repository's complete published history is scanned with Gitleaks before a
release. Runtime secrets, generated TLS keys, logs, and the SQLite database are
excluded by `.gitignore` and must never be committed.
