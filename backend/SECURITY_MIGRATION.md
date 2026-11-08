# Security hardening rollout

This branch includes application changes and database migrations. Do not deploy
only the frontend or skip migrations. This document will be extended as the
remaining hardening changes are integrated.

## Provider encryption and signing-key rotation

Provider credentials now use AES-GCM with a separate 32-byte key, a fresh nonce,
and authenticated column context. The web application refuses legacy CFB blobs;
legacy decoding is confined to the explicit migration command. The old committed
example signing key is rejected at startup.

Before replacing any key, stop/drain API and background workers and keep private
backups of both the database and the old environment file outside the checkout.
Keep backup files mode 600. Configure independently generated `SECRET_KEY` and
`DATA_ENCRYPTION_KEY` through your deployment's secret store. The signing key must
contain at least 32 bytes; the encryption key is base64 encoding of exactly 32
random bytes. Do not reuse the signing key for encryption or print either value
in job logs. Fresh installations also need both keys.

With the new keys configured and the new backend image/tool available, validate
old credentials without writing, then explicitly apply the migration:

```bash
python migrate_provider_credentials.py --legacy-env-file /run/secrets/previous.env
python migrate_provider_credentials.py --legacy-env-file /run/secrets/previous.env --apply
```

The private environment backup supplies the *old* signing key used by CFB. A file
containing only that key can be supplied with `--legacy-signing-key-file` instead.
The command reports counts only. All updates share one transaction; invalid JSON,
wrong keys or concurrent changes abort it. Historical CFB integrity cannot be
proven retrospectively. After a connection failure, run validation again to
establish whether the transaction committed. The command is idempotent.

For later AES-GCM key rotation, configure the new encryption key and pass the old
base64 encryption key in a private file using `--previous-data-key-file`, first
without and then with `--apply`. Keep recovery backups until verification passes.
Old application versions cannot read GCM ciphertext: coordinate any rollback with
a compatible data restore, and never restore compromised signing credentials.

Bootstrap no longer prints or generates an unrecoverable administrator password.
Use an interactive terminal or a mode-600 password file with `bootstrap.py
--password-file PATH`. The isolated E2E harness generates fresh signing,
encryption and administrator credentials for each run and never prints them.

## Sessions, API keys, and historical guest chats

1. Back up the database and record the deployed revision. Stop/drain the old API
   workers before running migrations; do not issue credentials with old and new
   authentication implementations serving traffic simultaneously.
2. Complete the provider-credential migration above, then run `alembic upgrade head` from the backend with the deployment's existing
   database configuration. The new revisions add credential versions, revoke
   unrestricted API keys, and make historical anonymous conversations private.
   Account-owned chat visibility is preserved. No conversation is deleted.
3. Deploy the matching backend and frontend. Use HTTPS and keep
   `SESSION_COOKIE_SECURE=true`. `false` is supported only in the local environment,
   including the isolated HTTP E2E stack.
4. Log in as an administrator and create replacement API keys, explicitly selecting
   knowledge bases for document/graph search, and chat engines for conversation
   permissions. Lifetimes are 1–90 days (30 by default). Keys cannot call
   management or document-download endpoints. A chat key can
   create conversations only with its selected engines and continue only chats
   created by that same key. Human-owned chats and other keys' chats are excluded.
   Update retrieval and evaluation integrations and verify that unrelated
   knowledge bases and administrative endpoints return 403.
5. Verify password reset/change: all existing sessions and API keys must stop
   working. Credential changes also increment a version, so a concurrent login
   authenticated before the change cannot leave a usable old session behind.
   Account deactivation, verification changes, email changes, and role changes also
   revoke credentials. The reset and verification hooks no longer print tokens;
   configure secure delivery before enabling a self-service mail flow. Operators
   can continue using the existing password-management command.
6. Existing unsigned guest browser cookies are replaced. Their old conversations
   cannot be securely reassigned based on the untrusted cookie. Administrators
   retain access to the preserved records; legacy public guest links stop working.
   New guest chats start private and can be shared explicitly by their owner.

For the regression workflow, issue a `chat:write` key restricted to the engine
under evaluation; it needs no management permissions.

New API-key digests use a versioned format that older, unrestricted workers cannot
authenticate. Rollbacks deliberately invalidate sessions/keys and do not republish
historical guest conversations. A rollback is not a way to restore old credentials.

## Public data behavior

The optional external-engine answer cache stays disabled by default. If enabled,
it reuses only completed, approved first-turn answers from the same account and
API key (or the same signed guest browser), engine, visibility, and keyed
configuration snapshot. The snapshot includes private engine settings, linked
knowledge-base and model revisions; only the digest is stored in message metadata.
There is no administrator or public-chat bypass. Both question and refined-goal
lookups return at most one answer from the last 15 days, and goal matching includes
the complete response format. Conversations with history bypass the cache.
Existing answers without the new context are not reused or backfilled. No database
migration or deletion is needed. Cache expiry is not immediate document-content
invalidation: keep the feature disabled when current corpus freshness is required.

Chat-flow diagnostic logs no longer copy question/goal/answer text, endpoint URLs
or exception bodies. This does not change the explicitly configured Langfuse
prompt/answer tracing policy or erase historical logs.

Public chat/config responses omit internal prompts, endpoint URLs, tokens, knowledge
base configuration and browser identity. Enabling `hide_sources` also hides source
and graph data in historical messages and streams. Non-admin file downloads require
an accessible chat that actually cites that document, an available public engine,
and visible sources. Files are served as attachments with `nosniff`.

## Model endpoints and credential updates

Inventory existing LLM, embedding and reranker configurations before restarting.
Set `PROVIDER_ALLOWED_ORIGINS` to a JSON list of exact trusted origins for Azure
resources, custom gateways and self-hosted services, for example
`["https://resource.openai.azure.com","http://ollama:11434"]`. Origins contain no
path, query, wildcard or credentials. Only register services controlled/trusted
by the deployment operator; this grants network access to that origin, including
its resolved addresses. Use HTTPS except for explicitly trusted private services.
Public built-in providers use their fixed official endpoints. New configuration
fields must be explicitly supported; SDK transport, proxy, header and credential
injection fields are rejected.

Every remote model requires its own explicit credentials. Missing values no
longer inherit process-wide OpenAI, Google or AWS credentials. Google service
accounts must use the standard Google OAuth token endpoint. Do not enable
`GOOGLE_GENAI_USE_VERTEXAI` for a Gemini API-key model; select an explicit Vertex
provider instead. Register the actual Ollama/local-sidecar origin before use.

Changing an endpoint, Azure deployment or Vertex destination requires fresh
credentials in the same update. The management UI supports this atomic update;
JSON credentials are stored in the encrypted credentials column. Old editors
cannot overwrite newer destination/credential combinations from stale state.
Model-test responses and logs no longer contain provider exception payloads.

JSON credential edits now update the credentials field correctly; the previous
field accessor mapped the submitted value to the wrong frontend field.

## Background and optional local-model services

Flower is no longer started by supervisord and no compose file publishes its
port. Stop/recreate existing background containers so old monitoring listeners
are removed. Any replacement monitoring must be operator-managed and authenticated.

The optional local-model profile now builds the reviewed source instead of using
the old unauthenticated `v4-with-cache` image. Build it with
`docker compose --profile local-embedding-reranker build local-embedding-reranker`.
The image downloads exact approved model revisions during the build; runtime
model loading is offline and remote Python code is disabled. Changing the model
manifest requires another image build. Model inputs are limited to 8192 characters
per text and 65536 characters per request, with at most 32 texts; the backend splits
larger ordinary batches without changing their order. Inference uses one worker,
one active request and batches of four, with a 2048-token model context.

Set a separate random `SIDECAR_AUTH_TOKEN` (at least 32 characters) in the private
compose environment, register `http://local-embedding-reranker:5001` in the
backend's provider origins, and save that token as the encrypted credentials for
each Local Embedding/Local Reranker provider. Placeholder `dummy` credentials no
longer work. Stop/remove old sidecar containers and recreate the profile; it has
no host-published port. Default compose CPU/memory limits are 2 CPUs / 8 GiB.

Credential-backed E2E no longer records traces or video, uploads HTML reports, or
publishes reports to Vercel. CI retains the test status and masked console output.
Historical reports may need separate review and credential rotation; disabling
new uploads does not revoke any credential captured previously.

## Prompt templates

Review custom engine templates before restarting. Templates now support text
variables, public data lookups, `if`, bounded nonrecursive `for` loops, and a small
set of pure formatting filters (including `tojson(indent=2)`). Python attribute
traversal, function calls other than dictionary `items`/`keys`/`values`, template
imports/macros, arithmetic expansion, and file/network/completion filters are
rejected both when saving engine settings and before rendering stored templates.
Banks chat/multimedia extension tags are not supported in engine text prompts.
All shipped templates retain their rendered content and LlamaIndex's partial
prompt/predict interface remains supported.

Limits are 32 KiB of template text, 512 KiB of string context, 1 MiB of rendered
output and 10,000 total loop iterations. Context must contain data values rather
than executable Python objects. Oversized or unsupported templates fail closed;
reduce their context or rewrite them using the supported text syntax.

## Web and sitemap imports

Rebuild the backend image to install the locked Chromium revision. The image now
stores browser binaries under `/opt/playwright`; development installations should
use `playwright install chromium` or set `PLAYWRIGHT_BROWSERS_PATH` explicitly.
Each URL/sitemap runs in a fresh process with no application credentials, a
45-second hard deadline, isolated browser state and bounded HTTP bodies. Network
connections resolve with a deadline, reject nonpublic/transition addresses and
connect only to the validated numeric address while retaining TLS hostname
verification. Redirects and browser subrequests use the same check. Chromium's
fallback network uses an unavailable proxy; service workers, server WebSockets,
QUIC and nonproxied WebRTC UDP are disabled. Cookies and authorization headers are
not forwarded. Dynamic pages that need authenticated sessions, POST requests or
live WebSockets are not supported by this public-document crawler.

Public destinations are restricted to ports 80/443. An operator may explicitly
register trusted internal origins in `CRAWLER_ALLOWED_PRIVATE_ORIGINS`; this
allows that exact origin's resolved addresses and port. Keep it empty in normal
public deployments. The local E2E fixture registers only `http://static-web-server`.
Do not register hostnames whose DNS is controlled by untrusted users.

Limits are 200 URLs per datasource, 10 minutes per import batch, 100 requests and
20 MiB per page, 5 MiB per HTTP response, and 2 MiB of rendered document text.
Servers must honor `Accept-Encoding: identity`. Split larger sitemaps or datasets
before import. Failed/blocked pages now leave the import failed rather than
silently reporting success; already imported pages remain available. Crawl errors
omit URL query strings and response contents. Document hashes are stable across
workers, avoiding unnecessary reindexing on subsequent imports.

## Browser write protection and generation context

Credentialed CORS now accepts only explicit HTTP(S) origins. Remove
`BACKEND_CORS_ORIGIN_REGEXP`; wildcard hosts and origins containing credentials,
paths, queries or fragments are rejected. Register trusted direct browser client
origins in `BACKEND_CORS_ORIGINS`. The frontend proxy verifies requests from its
own origin before forwarding them as explicit server requests, so same-origin
preview deployments do not require a wildcard backend origin rule.

Unsafe requests with Origin or Referer must match the backend origin or an
explicitly trusted origin. Originless cookie clients must send `X-CSRF-Guard: 1`
(or use an Authorization header); browser fetch metadata identifying a cross-site
or same-site request without an origin is rejected. Frontend browser/server
clients and the E2E login helper now send the guard. Headerless stateless clients
remain supported. Rejected requests do not create browser identities.

GET recommended-questions is now cache-only, including for owners. The UI requests
generation with POST `?refresh=false`; explicit refresh uses POST `?refresh=true`
(the existing POST default). Public readers can only obtain cached suggestions.
Continuation model context uses at most 19 recent history messages plus the
current user message, with a total 40,000-character budget. Only bounded text is
selected from the database; the stored transcript is retained.

## Shared generation admission

Every API replica must use the same Redis database, `SECRET_KEY` and
`GENERATION_QUOTA_NAMESPACE` (default `autoflow`). `GENERATION_REDIS_URL` is an
optional secret URL; otherwise admission uses `CELERY_BROKER_URL`. Use a separate
namespace for separate deployments. Redis must be private, authenticated where
appropriate, persistent, and configured without eviction of quota keys. Losing
or deleting these keys resets the current quotas. Redis connection failures
reject new generation with 503; there is no process-local fallback.

`GENERATION_LIMITS` accepts a JSON object overriding individual defaults. Limits
count admitted requests in fixed UTC minute/day windows: globally 60/1000,
per source IP 20/200, per authenticated account 10/100 and per signed guest 3/20.
Concurrent limits are global 20, IP 5, account 2 and guest 1. All API keys for an
account share its budget. IPv6 privacy addresses share a /64 IP bucket. Changing
the guest cookie cannot bypass the IP or global limit. These are request budgets;
choose deployment values that fit the configured models and expected usage.

Chat initialization is authorized before admission, then admission precedes chat
creation and provider initialization. Continuations and recommendation generation
share a per-chat lock. Cached recommendations and read-only GETs do not consume a
generation allowance. Failed generations consume their allowance. Refresh keeps
the previous recommendations until replacement commits successfully. Rejected
requests return 429 with Retry-After. Request bodies for these endpoints are
limited to 256 KiB and must arrive within 15 seconds.
Reading a missing legacy subgraph returns 404 rather than silently invoking paid
retrieval to reconstruct it. Existing stored graph data remains readable.

The configured generation deadline defaults to 180 seconds. Iteration and guarded
HTTP requests check the deadline; guarded transports limit outbound bodies to
2 MiB, responses to 4 MiB and provider attempts to 16. The OpenAI-compatible and
Cohere HTTPX clients reject compressed generation responses to prevent a
compressed body bypass. ProviderSession clients bound decoded response bytes.
A running synchronous SDK call can finish its bounded I/O before cleanup; this
is not a process-kill deadline. A disconnected SSE client does not release its
concurrency lease until the running iterator has returned and closed. Crashed
worker leases expire after the deadline plus a 120-second cleanup allowance.
Provider-specific coverage and the remaining DSPy transport path are still under
review while this PR is a draft.

Forwarded client IP headers are ignored by default. Only register exact proxy
addresses/CIDRs in `GENERATION_TRUSTED_PROXY_CIDRS` when that proxy sanitizes its
forwarding chain. The chain is inspected from the immediate peer backwards to
the first untrusted address. Do not trust a shared proxy range without an
authenticated, sanitized origin path. Docker and the CLI disable Uvicorn's
independent proxy-header rewriting; custom launch commands must also use
`--no-proxy-headers`. Without trusted forwarding, clients behind one proxy share
its conservative IP quota. Validate the actual deployment proxy before rollout.

The security tests launch an isolated local Redis with persistence disabled. Use
`SECURITY_REDIS_SERVER=/path/to/redis-server` if it is not on PATH; CI installs the
test binary explicitly. No deployment Redis database is accessed by these tests.

## Security events and committed audit records

Apply revision `31f7b624d9a0` before starting the new API or workers. It adds
nullable attribution fields and indexes to `staff_action_logs`; existing rows
remain explicitly unattributed. Preserve those historical rows. Downgrading this
revision removes the new attribution fields, so export the audit table before a
schema rollback if those records must be retained.

New records contain the authenticated actor ID, authentication kind, API-key ID
when applicable, a server-generated request ID, resource identifier and changed
field names. Graph edits additionally record the knowledge-base ID, which
disambiguates per-KB entity and relationship IDs. Values, credentials, documents,
prompts and before/after snapshots are not copied into new audit records. Legacy
numeric target IDs remain available; nonnumeric identifiers use
`target_identifier`. A bulk SQL operation without one explicit primary-key
predicate is identified as `bulk`.

The database audit hooks cover ORM and session-executed update/delete operations
for users, LLMs, embeddings, rerankers, chat engines, site settings, knowledge bases,
datasources, API keys and evaluation datasets/items/tasks within authenticated write
requests. Graph edits use an explicit writer. Audit insertion shares the mutation
transaction; failure to write the audit record prevents the mutation from
committing. Synopsis creation now commits its audit row with its graph changes.
Unattributed internal graph jobs are marked `system`; HTTP actor identities are
never taken from caller-supplied headers. Offline imports, raw connection-level
SQL and database administrator actions require their separate operator/database
audit facilities.

The `security.audit` logger emits a separate JSON event stream for login outcomes,
rejected sessions/API keys, accepted API-key requests, authorization denials and
privileged HTTP request outcomes. Request logs describe the HTTP outcome; the
database rows identify committed mutations. Events use route templates, omit
queries/bodies/cookies/credential values and HMAC the source address. Each response
gets a fresh `X-Request-ID`, overriding untrusted inbound correlation values.
Collect this stream and the audit table in the deployment's restricted logging
system, configure retention/alerts for authentication failures and privilege
denials, and validate actual collector ingestion during rollout.
