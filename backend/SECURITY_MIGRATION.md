# Security hardening rollout

This branch includes application changes and database migrations. Do not deploy
only the frontend or skip migrations. This document will be extended as the
remaining hardening changes are integrated.

## Sessions, API keys, and historical guest chats

1. Back up the database and record the deployed revision. Stop/drain the old API
   workers before running migrations; do not issue credentials with old and new
   authentication implementations serving traffic simultaneously.
2. Run `alembic upgrade head` from the backend with the deployment's existing
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
