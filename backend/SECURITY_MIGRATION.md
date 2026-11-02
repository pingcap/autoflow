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
