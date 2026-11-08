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

## Langfuse credentials and tracing

The Langfuse integration keeps intentional prompt/answer tracing. Its destination,
public key and encrypted secret now live in one versioned `site_settings` record;
no new table is required. AES-GCM authenticates the destination and public key as
well as the secret. Management responses never return plaintext or ciphertext.
The editor saves the configuration atomically, rejects stale revisions and
requires a fresh key when changing the destination. Removing the key explicitly
disables tracing for new chats. In-flight chats retain their starting snapshot;
drain them before an urgent revocation, and revoke exposed keys with Langfuse.

Before restarting workers, register trusted self-hosted destinations using
`LANGFUSE_ALLOWED_ORIGINS`, a JSON list of exact origins. The US and EU official
cloud origins are built in. This grants network access to the registered service
and its resolved addresses; register only operator-controlled/trusted services.
Base paths remain supported. Clients do not inherit proxy settings or keys from
the environment, follow redirects or accept oversized responses. Authentication
tests return generic failures rather than upstream error bodies.

With the same independent `DATA_ENCRYPTION_KEY` used for provider credentials,
run these commands while API/worker writers are stopped:

```bash
python migrate_langfuse_credentials.py
python migrate_langfuse_credentials.py --apply
```

The default dry-run executes only reads and in-memory encryption checks. Apply
encrypts the old `langfuse_secret_key` and clears its plaintext in the same
transaction. It is idempotent, reports counts only and rolls back on failure.
The runtime refuses an unmigrated non-empty legacy key. Later data-key rotations
must run **both** credential migration tools with `--previous-data-key-file`,
first as dry-run, then with `--apply`, before restarting workers. Back up the
database and old key privately beforehand. To roll back application versions,
restore the matching private database/key backup while all writers are stopped;
do not point older code at the new record or decrypt secrets into SQL/job logs.
Remove restored legacy plaintext again before bringing the hardened version back.

Tracing clients are scoped to one chat/configuration snapshot. Context-local
dispatchers route concurrent LlamaIndex events to their own clients, instead of
the SDK's first-client process singleton; completing one chat does not unregister
another's handlers. Clients, worker queues and HTTP connections are closed on
normal completion or failure. The SDK adapter uses interfaces of the locked
Langfuse 2.x release; retain the real-SDK concurrency and shutdown regressions
when upgrading it. This does not erase historical traces or revoke old keys.

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

Langfuse remains an intentional destination for prompt/answer tracing. Each chat
now has a 256-event pending queue, a five-second final flush allowance and a shared
three-second worker cleanup allowance. A stalled final flush discards pending
events with a value-free warning rather than retaining the chat indefinitely.
Only that chat's socket connections, workers and exit hooks are stopped. The
transport enforces response deadlines while reading headers as well as bodies;
TLS certificate and hostname checks remain enabled. OS DNS resolution and Python
SDK CPU work are not forcibly killed; cancelled clients cannot open later network
connections. These bounds can lose traces when Langfuse is unavailable or a chat
exceeds the queue capacity, without changing the generated answer.

Autoflow's text/JSON tracing keeps `data:` strings as text. Langfuse's automatic
conversion of those strings into separate media uploads is disabled because that
upload path bypasses the configured HTTP transport. Binary media tracing would
require a separately reviewed transport before enabling it.

The frontend now uses Zod 4.6.7 with compatible form resolvers. Existing optional
model configuration fields and numeric form values retain their API behavior.
The unused Zod 3 adapter was removed; the lockfile no longer contains Zod 3.
Do not use Zod 4's bundled v3 compatibility entry point to bypass the migration.

Typed JSON response handlers bound decoding to 64 MiB, error bodies to 64 KiB,
individual collections to 10,000 entries, total validation nodes to 100,000 and
nesting to 64 levels before schema traversal. The byte ceiling accommodates the
default 10 MiB text upload even with JSON escaping. Normal pages and document text
remain intact; very large result sets need pagination or a raw document download.
Deployments that increased document upload limits should review these limits with
their UI workloads. Limits reject oversized responses instead of truncating data.
Raw downloads and streaming chat/graph protocols remain separate from this JSON
handler. The package patch alone does not bound schemas without collection limits.

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

Sentry errors and performance transactions use an allowlist before transmission:
error classes, code locations (no local values/source lines), registered route
templates, HTTP method, trace identifiers and timing. Raw messages, exception
bodies, request URLs/queries/headers/cookies/bodies, user context, arbitrary tags,
SQL/model span descriptions, breadcrumbs and attachments are omitted. Unexpected
event shapes are dropped. Request-body, local-variable and source-code collection
are also disabled at the SDK. The locked Sentry 2.26.1 profile/session channels do
not use these hooks, so profiles, automatic sessions and experimental logs/metrics
are disabled. The legacy `SENTRY_PROFILES_SAMPLE_RATE` setting remains accepted
but cannot enable profiles; trace sampling continues to follow
`SENTRY_TRACES_SAMPLE_RATE`. Review historical Sentry and application-log access
and retention separately; this change neither retrieves nor deletes old events.

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

Google Gemini/Vertex SDK and DSPy calls now use the same explicit destination,
identity and configured output limit. Gemini keeps `v1beta`; Vertex keeps
`v1beta1` with its configured project/location, including the global endpoint.
Ambient Google base URLs, credentials, recording/replay and proxy settings do
not select another destination or persist prompts. Synchronous and asynchronous
model/metadata requests reject redirects and compressed responses, enforce the
shared generation deadline/call allowance and bound uploads/responses to 2/4 MiB.
Vertex service-account renewal, including later expiry and copied credentials,
uses the same bounded OAuth transport; renewal consumes a provider call.
DSPy retains JSON schemas, tool history and SSE usage reporting. Runtime output
overrides may lower the configured token limit but cannot raise it. The Google
SDK's internal retry multiplier is disabled; LlamaIndex keeps the configured
retry policy and each actual request is charged to the generation allowance.

OpenAI-compatible text providers keep `/completions`, their existing DSPy prompt
delimiter and text-only generation options. The compatibility path now uses
owned synchronous/asynchronous clients with the same response, request, deadline
and attempt bounds, explicit credentials and output cap. Token logprobs and usage
retain their original mapping/attribute shape. No LiteLLM global network client
or callback lifecycle is invoked. This uses the pinned DSPy text compatibility
hook because its native engine supports only chat/Responses; revalidate that
hook when upgrading DSPy, rather than silently switching text models to chat.

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
External chat engines and post-verification services require registration in
`EXTERNAL_ENGINE_ALLOWED_ORIGINS` (a JSON list of exact HTTP(S) origins, without
paths). Register the existing services before rollout, including intended private
services. Configuration writes and runtime requests both check this policy; no
external service is implicitly trusted. Paths are preserved, URL credentials and
query-string secrets are rejected, redirects and ambient proxies/netrc credentials
are disabled. External streams allow at most 4 MiB and a 120-second read budget
(or the shorter active generation deadline); post-verification allows 64 KiB and
10 seconds. Connections close on failure and consumer disconnect. ProviderSession
also applies standalone response/read limits when no chat generation scope exists.
Read budgets are checked between chunks, with bounded blocking socket reads as
described above; they are not process-kill deadlines.

DSPy OpenAI-compatible chat calls use explicit native engines with guarded sync
and async transports. Both keep the configured origin/key binding, ignore
ambient proxy and provider-base variables, reject redirects and count actual
outbound attempts against the active generation allowance. They enforce the
same body/deadline limits as the primary model clients. Borrowed engine
transports close each request on completion, error or early stream exit, so
copies and different event loops do not share live HTTP pools. Model tests and
background calls without a chat lease also have response-size and 60-second
read budgets. The normal admitted chat deadline remains unchanged.
Explicit native capability hints preserve OpenAI JSONAdapter schema/tool
selection using the bundled metadata snapshot, without a discovery request.
OpenAI-compatible and Azure native transports enforce the model's configured
output-token cap on the actual request, including canonical calls and LM copies.
They preserve the primary client's model-specific `max_tokens` or
`max_completion_tokens` spelling, allow smaller per-call limits, and cap larger
ones. This keeps reasoning models usable below DSPy's suggested 16k-token
constructor default without increasing Autoflow's configured allowance.

Azure DSPy calls use the same guarded native transport, while retaining the
configured legacy deployment URL, gateway prefix, `api-version`, deployment
model and API key. They do not silently switch existing installations to Azure's
newer `/openai/v1` endpoint. The primary Azure SDK clients also bind API-key
authentication explicitly: the underlying SDK otherwise prefers a process-wide
`AZURE_OPENAI_AD_TOKEN`, even when the model has its own configured key. Owned
client subclasses prevent this substitution, including after `with_options()`
copies; they do not change the process environment. Autoflow's supported Azure
configuration remains API-key based. Real sync/async tests cover the primary and
DSPy clients, separate deployment versions/keys, JSON schemas and tool history,
stream usage/early closure, proxy rejection and response/deadline/attempt bounds.

Bedrock primary clients keep the Converse/ConverseStream protocol and AWS SigV4
signing, using packaged botocore serializers and event parsers. Destinations are
derived from the configured region and AWS partition; server profile files,
endpoint/proxy/CA overrides and ambient session tokens cannot change them.
Each synchronous or native asynchronous call owns its guarded HTTP connection.
Async streaming no longer uses the SDK's unbounded background producer, which
also lost the active generation context. Early exit and cancellation close the
connection; binary stream checksums, incomplete streams, body/deadline/attempt
limits and the configured output cap are enforced. Inline images/documents and
tool/thinking/usage blocks remain supported. Supply media inline: SDK conversion
must not read arbitrary local files or download remote URLs before admission.
DSPy Bedrock now shares the same owned Converse transport and explicit identity.
Only LiteLLM's in-memory request/response codecs are used; its network clients,
ambient endpoint/session-token lookup and automatic retry lifecycle are bypassed.
Chat/JSON adapters, inline binary media, tools, thinking, usage and native or
`streamify` streaming remain supported. Missing reasoning-token counts stay
unknown instead of triggering a tokenizer download. The pinned DSPy legacy
compatibility hook preserves provider-shaped documents and file names that its
canonical request cannot represent; revalidate this hook when upgrading DSPy.

Ollama's primary clients and DSPy use guarded HTTPX transports. DSPy's explicit
native engines retain `/api/chat` through the locked SDK's pure request/response
codecs, without its completion lifecycle: that lifecycle also queried `/api/show`
through an unguarded global client. No global SDK client or model metadata is
changed. Exact configured credentials (including the keyless-local placeholder),
generation settings, JSON schemas, inline images, tool history and token counts
are preserved. NDJSON streams are converted incrementally, require a final done
record, and close their sockets on early exit. Custom engine capabilities are
declared so JSONAdapter continues sending format constraints. Real loopback
tests cover sync/async calls, streaming, concurrent credential bindings, copies,
multiple event loops, hostile proxy settings and the body/time/attempt limits.

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

Run `python scripts/test_security_tidb.py` from an isolated backend checkout with
its locked dependencies and Docker available. The script refuses a deployment
`.env`, starts a digest-pinned TiDB 8.5.4 container on ephemeral loopback ports,
and removes that container after the tests. Each case installs the shipped
Alembic migrations into a new disposable database. These tests cover concurrent
Langfuse configuration updates, encrypted credential migration and rollback,
provider destination/key updates, password-reset revocation and scoped answer
cache queries on TiDB. A populated-upgrade case starts at revision
`04947f9684ab`, migrates fake legacy provider and Langfuse credentials, upgrades
the schema, and checks existing sessions, revoked keys, guest privacy and audit
attribution. It also checks that schema rollback cannot reactivate unrestricted
keys or republish guests, and that re-upgrade is idempotent. This is not a
substitute for restoring the deployment's own private backup before rollout.
CI runs the same script; no deployment database URL or
credentials are accepted.

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
