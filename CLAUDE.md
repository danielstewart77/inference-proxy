# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An inference proxy that fronts several upstream model providers behind one
endpoint and its own issued keys. Read `README.md` first — it covers the
client / credential / model model and the Anthropic OAuth quirk, and this file
only adds what matters when changing the code.

The repo began as an Azure OpenAI proxy for a corporate deployment. The
protocol-translation guts were kept; everything around them — SQL Server, the
human user table, the self-service portal, the installer downloads — was
removed. If you find a stale reference to any of that, delete it.

## Conventions

- **Tests land with the code.** Not as a follow-up. `pytest` must be green
  before anything is committed; a test that fails because the API moved gets
  patched or deleted, never skipped.
- **Postgres is the only engine.** `DATABASE_URL` is required and the tables
  live in `DB_SCHEMA`. Migrations are Alembic. The suite runs on a scratch
  database it creates from `PROXY_TEST_ADMIN_DSN` (process environment only,
  never `.env`); never point a test at the live database or the `proxy`
  schema.
- **Never render a secret back to a page.** The admin console shows a masked
  placeholder and only writes a column when a non-empty value is submitted.
  A newly minted proxy key is the one exception, shown once at mint time.
- **The key cache is authoritative on the hot path.** Any change to keys or to
  a client's `disabled` flag must call `refresh_key_cache(session)`, or the
  change won't take effect until restart. A change made by another process
  (`python -m app.cli`) reaches the server through `key_cache_refresh_loop`
  within `KEY_CACHE_REFRESH_SECONDS`.
- **Fail closed.** `resolve_deployment` defaults `is_admin=False`, and an
  admin-only model hit by an unprivileged client returns 404, not 403, so
  restricted deployments aren't enumerable.

## Where things live

`app/deployments.py` decides the upstream protocol from the target URI and
builds the auth headers; it is the single place either concern is expressed.
`app/credentials.py` turns a credential row into a secret and owns OAuth
refresh. Route-specific header or body rewriting belongs in the route module
under `app/proxy/`, not in either of those.

Adding a provider usually means no code at all — a credential row, a provider
row and a model row. It means code only when the provider needs a wire format
none of the existing routes speak. A model authenticates with its provider's
credential and auth scheme, never with columns of its own.

`app/cli.py` makes those rows and client keys from the command line. It calls
the `insert_*`, `apply_*_update` and `issue_key` functions the console's form
routes call; a create or update path's validation lives in that function, never
in the route or the CLI, so the two cannot drift. `check-model` goes through
`forward_chat_completion`, the chat route's own body after authentication.
