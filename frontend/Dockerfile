FROM node:20-alpine AS base

# 1. Install dependencies only when needed
FROM base AS deps
# Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
RUN apk add --no-cache libc6-compat

WORKDIR /tidb.ai

RUN mkdir -p frontend/patches
RUN mkdir -p frontend/app
RUN mkdir -p frontend/packages/widget-react

# Install dependencies based on the preferred package manager
COPY frontend/package.json ./frontend
COPY frontend/pnpm-*.yaml ./frontend
COPY frontend/patches/* ./frontend/patches
COPY frontend/app/package.json ./frontend/app
COPY frontend/packages/widget-react/package.json ./frontend/packages/widget-react

WORKDIR /tidb.ai/frontend

RUN npm install -g corepack@latest
RUN corepack enable pnpm
RUN pnpm i --frozen-lockfile


# 2. Rebuild the source code only when needed
FROM base AS builder
# Need git client to fetch git revision info
RUN apk add --no-cache git
WORKDIR /tidb.ai
COPY --from=deps /tidb.ai/frontend/node_modules ./frontend/node_modules
COPY --from=deps /tidb.ai/frontend/app/node_modules ./frontend/app/node_modules
COPY --from=deps /tidb.ai/frontend/packages/widget-react/node_modules ./frontend/packages/widget-react/node_modules
# For symbol link of root README.md
COPY . .

# This will do the trick, use the corresponding env file for each environment.
#COPY .env.production.sample .env.production

WORKDIR /tidb.ai/frontend

ENV BASE_URL=""
ENV SITE_URL=""

RUN rm -f app/.env
RUN echo BASE_URL=${BASE_URL:-'""'} >> app/.env.production

RUN corepack enable pnpm
RUN pnpm run build:docker
RUN pnpm run build:widget-react

# 3. Production image, copy all the files and run next
FROM base AS runner
WORKDIR /tidb.ai

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /tidb.ai/frontend/app/.next/standalone .
COPY --from=builder --chown=nextjs:nodejs /tidb.ai/frontend/app/.next/static app/.next/static
COPY --from=builder /tidb.ai/frontend/app/public app/public

USER nextjs

EXPOSE 3000

CMD ["node", "app/server.js"]
