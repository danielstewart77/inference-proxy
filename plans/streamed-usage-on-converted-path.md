# Streamed turns on the converted path report their token usage

Opened 2026-10-07. Found while verifying Cypher's rotation hooks: her session
log recorded `{"inputTokens": 0, "outputTokens": 0}` for a turn the proxy's own
`usage_log` recorded as 16,271 input and 127 output.

## Where the defect is

Cypher's dsh profile talks to the proxy as `api: openai-completions`
(`hive/dsh-headless-resumable/cordis.patch.yml`), so her turns run through
`app/proxy/chat_to_anthropic.py`. pi-ai asks for counts correctly — it sets
`stream_options: {include_usage: true}` and reads `prompt_tokens` /
`completion_tokens`.

That path's streaming branch captures the upstream Anthropic usage into
`captured` (which is why the metering row is right), converts it correctly on
the *non*-streaming branch via `_usage_to_chat`, and then ends the stream with
`yield chunk({}, finish_reason)` followed by `[DONE]`. The local `chunk()`
builder has no `usage` key at all, so a streamed caller is never told anything.

The sibling path `app/proxy/chat_completions.py` already does this right, and
even forces `include_usage` on when a caller omits it so the metering row is
never empty. The gap is unique to the conversion path.

## Requirements

1. A mind whose streamed turn goes through the proxy's Anthropic-conversion
   path is told how many tokens that turn used.
2. The counts it is told match what the proxy records in its own usage log for
   that same turn.
3. When the model read from cache, the cached portion is reported separately
   from fresh input.
4. The counts arrive whether or not the mind explicitly asked for them.
5. A turn that errors partway, or ends before the upstream said it was
   finished, still reports whatever counts the upstream gave before it
   stopped. Daniel's call on 2026-10-07: a partial count is better than no
   count. This makes emission order load-bearing — the usage chunk goes out
   above the failure return — and he struck the test that would have guarded
   it, so nothing does.
6. Cypher's rotation hook, reading her own session log after a real turn,
   reports a non-zero context measurement.

## Deferred — raise with Daniel when this ships

Daniel paused these on 2026-10-07 and asked to be reminded once the proxy fix
is done. Both block Cypher being fully operational:

- `training_capture` fails every turn with "attempt to write a readonly
  database".
- `auto_remember` skips every turn with `reason: no-token`.

## Deltas from the spec grill (2026-10-07, three reviewers, all findings verified)

The grill changed the design materially. What survived:

**`prompt_tokens` semantics were the whole ballgame.** `_usage_to_chat` set
`prompt_tokens` from Anthropic's `input_tokens`, which *excludes* cache, while
every OpenAI consumer reads it as *inclusive* with `cached_tokens` as a subset
— a contract this repo already documents at `app/orm.py:287-290` and
`app/proxy/chat_completions.py:849-850`. pi-ai acts on it literally:
`input = max(0, prompt_tokens - cached_tokens - cache_write_tokens)`
(`openai-completions.js:1069`). So shipping requirement 3 naively would have
reported a cache-read turn's fresh input as zero.

Worse, `_usage_to_chat` had no wire field for `cache_creation_input_tokens` at
all, so on the first turn of any conversation — where Anthropic writes the whole
prompt into cache — the entire cached prefix was unrecoverable. The rotation
hook sums all four usage fields, so this never read as zero; it read as
systematically light, which for a threshold is worse because it looks fine.

Decided: `prompt_tokens` becomes `input + cache_read + cache_creation`, with
`cached_tokens` and `cache_write_tokens` named inside `prompt_tokens_details`.
pi-ai reads both. This also changes the non-streaming path, which shares the
helper, and the exact-equality assertion at `tests/test_chat_to_anthropic.py:573`
is patched to the corrected shape.

**Chunk placement.** Usage rides on the *existing* finish chunk on the normal
path — no new chunk, no shape change, so none of the twelve registered proxy
clients sees anything new. Only the failure and truncation exits, which have no
finish chunk, emit a usage-only chunk, and that one carries a single
`finish_reason: null` choice rather than the canonical empty `choices: []`,
because four existing tests index `choices[0]` on every chunk they see. Usage
goes out *above* the error line: the OpenAI SDK throws on `{"error": ...}` and
discards everything after it.

**Absent usage is reported as absent.** `_usage_to_chat({})` returns a
well-formed all-zero dict, and pi-ai *overwrites* usage on every usage-bearing
chunk, so emitting zeros would wipe a real measurement. Worse for Cypher
specifically: her hook reads an all-zero usage dict as absence and keeps walking
back the transcript, so she would silently inherit the *previous* turn's figure.
The emission is therefore gated on the upstream having actually reported counts.

**Two new requirements the original list did not cover:**

7. A turn aborted by the transport or by the caller hanging up still records
   whatever counts it had reached. There is no `try` around the event loop, so
   everything after it is skipped; the live `usage_log` carries real evidence
   (ids 103078 and 100774: status 200, no error type, NULL tokens, one of them
   81 seconds). `usage_holder` is updated inside the loop so metering survives
   an abort even when no chunk can be yielded — nothing can be yielded during
   `GeneratorExit`.
8. A later usage event never zeroes out counts an earlier one got right.
   `captured.update(usage_delta)` is a blind merge, so an upstream emitting a
   full usage object in `message_delta` with zeroed fields overwrites correct
   values from `message_start` — and the metering row would then agree with the
   chunk on a wrong number.

**Struck:** requirement 4 folded into requirement 1 (the converter never sees
the caller's request options, so no input can make it fail).

**Consequence accepted:** real usage arms pi-ai's `isContextOverflow`, dead
until now. Cypher is safe (1,000,000 declared, rotation at 10%), but a model row
with a stale or low `context_window` will begin turning successful turns into
loud context-exceeded failures.

## Deltas from the code grill (2026-10-07, three reviewers against the diff)

Requirement 2's wording was wrong, not the code. The metering row keeps
Anthropic's native fields and the caller is handed OpenAI's inclusive total, so
the two are only numerically equal on an uncached turn. Restated: **the caller
is told the same counts the proxy recorded, expressed in the convention of its
own wire** — the row's parts add up to what the caller was told. That is what
`test_the_caller_and_the_metering_row_describe_one_turn` checks, and dropping a
single field on the way to either side now fails it.

Six defects found in my own implementation, all fixed:

- An upstream reporting its prompt fields as **null** while reporting a real
  output count left a populated `captured`, so a gate asking "is anything
  known" emitted `prompt_tokens: 0` — the original bug, in its worst form. The
  gate now asks whether any part of the *prompt* was reported, and
  `_usage_to_chat` refuses to answer rather than inventing a zero.
- Counts arriving as **floats** were dropped by an `isinstance(..., int)`
  check. The blind merge this change replaced handled them correctly, so that
  was a regression I introduced. Floats and digit strings are now read; `bool`
  is refused, since it is an `int` subclass and would store as a count of one.
- The "don't let a zero overwrite" rule only refused literal zeros. A later
  event re-reporting the block with a **sentinel** walked through — and
  Anthropic's own `message_start` carries `output_tokens: 1` as a placeholder.
  Replaced with a per-key high-water mark, which is the honest semantic for
  counters that only grow.
- Counts on **`message_stop`** were never folded. Anthropic reports earlier,
  but shims on this route report there, and four live `/v1/messages`
  deployments point at one.
- Keeping the counts through an abort removed the operator's only tell: those
  rows land at status 200 with no error named, and an empty token column was
  how they were ever spotted. An abort now names itself
  (`upstream_stream_aborted`).
- The **non-streaming** sibling shares the helper and still fabricated zeros.
  It now inherits the same refusal.

Three test gaps closed: the upstream-`error` exit's emission had no test at all
(deleting it left every test green), the partial chunk's own gate had none, and
requirement 2 had none. One assertion deleted for having no detection power —
`prompt - cached - written == fresh` follows arithmetically from the three
assertions above it.

Rejected, with reasons:

- That naming the created portion misleads a canonical OpenAI client into
  reading 4,396 of fresh input rather than 300. It does not: cache-creation
  tokens genuinely *were* processed this turn, so 4,396 is the right answer to
  "how much was processed fresh" in OpenAI's own terms, while pi-ai recovers
  the 300 by subtracting the write as well. Both readers get a true number.
- That the row and the chunk disagreeing is a defect. They state the same turn
  in two conventions and cost is priced off the row's native fields, so nothing
  is double-counted. Worth knowing when reconciling a rotation threshold
  against the admin usage page; not worth changing.
- Chained double-pricing, if a deployment is ever pointed at this proxy
  itself: `_chat_usage_to_azure_usage` would read the inclusive total into
  `input_tokens` and the cached subset again into `cache_read_input_tokens`.
  No deployment targets this proxy today. Latent, recorded here, not fixed.

Suite: 231 passing, from a baseline of 218. Thirteen tests, each verified to
die on its own distinct one-line mutation.
