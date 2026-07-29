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
- **Migrations are Alembic, on SQLite.** `render_as_batch` is on because SQLite
  cannot `ALTER` most column properties. `BigInteger` must carry
  `.with_variant(Integer, "sqlite")` on any autoincrement primary key —
  SQLite only autoincrements a column declared exactly `INTEGER PRIMARY KEY`.
- **Never render a secret back to a page.** The admin console shows a masked
  placeholder and only writes a column when a non-empty value is submitted.
  A newly minted proxy key is the one exception, shown once at mint time.
- **The key cache is authoritative on the hot path.** Any change to keys or to
  a client's `disabled` flag must call `refresh_key_cache(session)`, or the
  change won't take effect until restart.
- **Fail closed.** `resolve_deployment` defaults `is_admin=False`, and an
  admin-only model hit by an unprivileged client returns 404, not 403, so
  restricted deployments aren't enumerable.

## Where things live

`app/deployments.py` decides the upstream protocol from the target URI and
builds the auth headers; it is the single place either concern is expressed.
`app/credentials.py` turns a credential row into a secret and owns OAuth
refresh. Route-specific header or body rewriting belongs in the route module
under `app/proxy/`, not in either of those.

Adding a provider usually means no code at all — a credential row and a model
row. It means code only when the provider needs a wire format none of the
existing routes speak.
