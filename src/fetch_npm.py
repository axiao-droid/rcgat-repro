"""Fetch npm package metadata via the huaweicloud mirror in a BFS from seed
packages, storing one JSON document per package under raw/.

Design:
  - node time   = time.created
  - edge set    = versions[dist-tags.latest].dependencies  (A depends on B)
  - edge time   = src node time (same convention as the HEP citation network)
  - text fields = description + keywords + readme prefix (content features)
  - resumable   : raw/<name>.json exists -> skip (rename to .partial first)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request

RAW = Path(os.environ.get("RAW_DIR", str(Path(__file__).resolve().parent / "raw"))).expanduser()
BASE = "https://repo.huaweicloud.com/repository/npm/"
TARGET = int(os.environ.get("TARGET", "4000"))
WORKERS = int(os.environ.get("WORKERS", "8"))
LOG = Path(os.environ.get("LOG_FILE", str(RAW.parent / "fetch_npm.log"))).expanduser()

SEEDS = [
    "lodash", "express", "react", "vue", "axios", "moment", "chalk", "debug",
    "uuid", "typescript", "webpack", "request", "async", "underscore", "bluebird",
    "commander", "minimist", "readable-stream", "inherits", "util-deprecate",
    "safe-buffer", "string_decoder", "isarray", "ms", "ansi-regex", "strip-ansi",
    "wrap-ansi", "string-width", "cliui", "yargs", "semver", "glob", "graceful-fs",
    "once", "wrappy", "inflight", "minimatch", "brace-expansion", "concat-map",
    "balanced-match", "path-is-absolute", "fs.realpath", "rimraf", "mkdirp",
    "supports-color", "has-flag", "color-convert", "color-name", "ansi-styles",
    "escape-string-regexp", "object-assign", "prop-types", "loose-envify", "js-tokens",
    "scheduler", "react-dom", "jest", "babel-core", "@babel/core", "eslint", "prettier",
    "rollup", "vite", "next", "nuxt", "gulp", "grunt", "mocha", "chai", "sinon",
    "enzyme", "redux", "react-redux", "zustand", "pinia", "router", "qs", "send",
    "depd", "etag", "fresh", "vary", "cookie", "accepts", "negotiator", "mime-types",
    "mime-db", "http-errors", "statuses", "toidentifier", "setprototypeof",
    "ipaddr.js", "proxy-addr", "forwarded", "merge-descriptors", "methods",
    "parseurl", "path-to-regexp", "encodeurl", "destroy", "on-finished", "ee-first",
    # --- expansion wave 2: broad ecosystem coverage ---
    "gatsby", "svelte", "@sveltejs/kit", "angular", "@angular/core", "preact",
    "solid-js", "qwik", "astro", "esbuild", "parcel", "webpack-cli", "tsup",
    "turbo", "lerna", "vitest", "playwright", "puppeteer", "cypress", "karma",
    "jasmine", "ava", "tape", "supertest", "nock", "knex", "sequelize", "typeorm",
    "prisma", "mongoose", "pg", "mysql2", "better-sqlite3", "redis", "ioredis",
    "mongodb", "levelup", "fastify", "koa", "hapi", "socket.io", "ws", "graphql",
    "apollo-server", "express-session", "cors", "helmet", "morgan", "body-parser",
    "cookie-parser", "serve-static", "finalhandler", "joi", "yup", "zod", "ajv",
    "validator", "dayjs", "date-fns", "luxon", "got", "node-fetch", "undici",
    "superagent", "ejs", "pug", "handlebars", "mustache", "nunjucks", "core-js",
    "regenerator-runtime", "mobx", "immer", "react-query", "swr", "recoil", "jotai",
    "antd", "element-ui", "element-plus", "bootstrap", "tailwindcss", "sass", "less",
    "postcss", "autoprefixer", "cssnano", "echarts", "chart.js", "d3", "three",
    "gsap", "framer-motion", "recharts", "dotenv", "cross-env", "shelljs", "execa",
    "globby", "fast-glob", "chokidar", "watchpack", "acorn", "terser", "uglify-js",
    "crypto-js", "bcrypt", "bcryptjs", "jsonwebtoken", "jose", "nanoid", "winston",
    "pino", "bunyan", "log4js", "sharp", "jimp", "canvas", "multer", "busboy",
    "firebase", "stripe", "twilio", "bull", "agenda", "amqplib", "kafkajs", "mqtt",
    "cheerio", "jsdoc", "typedoc", "markdown-it", "remark", "rehype", "gray-matter",
    "archiver", "tar", "yauzl", "yazl", "adm-zip", "split2", "pump", "cross-spawn",
    "get-stream", "is-stream", "p-map", "p-queue", "delay", "fast-deep-equal",
    "deepmerge", "clone", "flat", "camelcase", "pluralize", "inflection", "slugify",
    "i18next", "intl-messageformat", "@types/node", "@types/react", "@types/express",
    "@sentry/node", "@aws-sdk/client-s3", "@nestjs/core", "@mui/material",
    # --- expansion wave 3: application-layer packages (out-degree rich) ---
    "create-react-app", "react-scripts", "webpack-dev-server", "webpack-dev-middleware",
    "webpack-merge", "html-webpack-plugin", "mini-css-extract-plugin", "css-loader",
    "style-loader", "file-loader", "url-loader", "source-map-loader", "babel-loader",
    "ts-loader", "sass-loader", "postcss-loader", "eslint-loader", "eslint-plugin-react",
    "eslint-plugin-import", "eslint-config-airbnb", "eslint-config-airbnb-base",
    "eslint-config-prettier", "eslint-plugin-prettier", "eslint-plugin-jsx-a11y",
    "eslint-plugin-react-hooks", "@typescript-eslint/parser", "@typescript-eslint/eslint-plugin",
    "copy-webpack-plugin", "clean-webpack-plugin", "terser-webpack-plugin",
    "optimize-css-assets-webpack-plugin", "css-minimizer-webpack-plugin",
    "compression-webpack-plugin", "progress-bar-webpack-plugin", "case-sensitive-paths-webpack-plugin",
    "dotenv-webpack", "dotenv-expand", "react-app-polyfill", "react-refresh",
    "@pmmmwh/react-refresh-webpack-plugin", "fork-ts-checker-webpack-plugin",
    "babel-eslint", "@babel/preset-env", "@babel/preset-react", "@babel/preset-typescript",
    "@babel/plugin-transform-runtime", "@babel/runtime", "@babel/runtime-corejs3",
    "@babel/polyfill", "babel-preset-env", "babel-plugin-transform-runtime",
    "babel-plugin-module-resolver", "babel-plugin-import", "babel-plugin-styled-components",
    "postcss-preset-env", "postcss-import", "postcss-nested", "postcss-flexbugs-fixes",
    "autoprefixer", "tailwindcss", "purgecss", "@tailwindcss/postcss7-compat",
    "precss", "cssnano-preset-default", "stylelint", "stylelint-config-standard",
    "stylelint-config-prettier", "stylelint-webpack-plugin", "sass", "node-sass",
    "less-loader", "stylus-loader", "stylus", "jade", "coffee-script", "coffeescript",
    "pug-loader", "pug-plain-loader", "vue-loader", "vue-template-compiler", "vue-router",
    "vuex", "@vue/cli-service", "@vue/cli-plugin-babel", "@vue/cli-plugin-router",
    "@vue/cli-plugin-vuex", "@vue/cli-plugin-eslint", "@vue/test-utils", "vue-server-renderer",
    "react-hot-loader", "@hot-loader/react-dom", "react-router", "react-router-dom",
    "connected-react-router", "history", "react-helmet", "react-helmet-async",
    "react-loadable", "react-lazyload", "react-transition-group", "react-spring",
    "react-use", "react-hook-form", "formik", "react-final-form", "react-dropzone",
    "react-select", "react-datepicker", "react-virtualized", "react-window", "react-dnd",
    "react-beautiful-dnd", "react-grid-layout", "react-slick", "react-swipeable-views",
    "next-common-path", "@next/font", "next-auth", "next-i18next", "gatsby-link",
    "gatsby-image", "gatsby-source-filesystem", "gatsby-transformer-sharp",
    "gatsby-plugin-sharp", "gatsby-plugin-manifest", "gatsby-plugin-offline",
    "gatsby-plugin-react-helmet", "gatsby-plugin-typescript", "gatsby-plugin-sass",
    "gatsby-plugin-postcss", "gatsby-plugin-image", "gatsby-plugin-mdx", "@mdx-js/react",
    "mdx-bundler", "remark-gfm", "remark-html", "rehype-stringify", "unified",
    "hast-util-sanitize", "katex", "react-katex", "mathjax", "highlight.js",
    "prismjs", "shiki", "react-syntax-highlighter", "react-markdown",
    "swagger-ui-dist", "swagger-ui-react", "openapi-types", "@apollo/client",
    "apollo-link-http", "graphql-tag", "graphql-tools", "type-graphql", "mercurius",
    "prisma-client-js", "@prisma/client", "pg-promise", "knex-paginate", "objection",
    "bookshelf", "waterline", "sails", "sails-hook-orm", "sails-hook-sockets",
    "feathers", "@feathersjs/express", "@feathersjs/socketio", "loopback", "loopback-datasource-juggler",
    "socket.io-client", "socket.io-redis", "engine.io", "engine.io-client", "uWebSockets.js",
    "ws", "http-proxy", "http-proxy-middleware", "http-proxy-agent", "https-proxy-agent",
    "socks-proxy-agent", "proxy-agent", "pac-resolver", "pac-proxy-agent", "smart-buffer",
    "socks", "basic-auth", "basic-auth-connect", "passport", "passport-local",
    "passport-jwt", "passport-oauth2", "passport-facebook", "passport-google-oauth20",
    "passport-github2", "bcrypt-nodejs", "argon2", "pbkdf2", "crypto-browserify",
    "browserify", "browserify-aes", "browserify-cipher", "browserify-des", "browserify-rsa",
    "browserify-sign", "browserify-zlib", "create-hash", "create-hmac", "create-ecdh",
    "diffie-hellman", "elliptic", "hash.js", "hmac-drbg", "minimalistic-assert",
    "minimalistic-crypto-utils", "bn.js", "brorand", "public-encrypt", "randombytes",
    "randomfill", "sha.js", "ripemd160", "md5.js", "sha3", "keccak",
    # --- expansion wave 4: test tooling, backend frameworks, ORMs, cloud sdks ---
    "jest", "jest-cli", "jest-config", "jest-environment-jsdom", "jest-environment-node",
    "jest-resolve", "jest-util", "jest-worker", "jest-haste-map", "jest-mock", "jest-snapshot",
    "@jest/core", "@jest/globals", "@jest/test-sequencer", "@jest/reporters", "@jest/transform",
    "babel-jest", "ts-jest", "vitest", "mocha", "mocha-junit-reporter", "chai", "chai-as-promised",
    "sinon", "sinon-chai", "nock", "supertest", "proxyquire", "rewire", "istanbul", "nyc",
    "@istanbuljs/nyc-config-typescript", "coveralls", "codecov", "karma", "karma-chrome-launcher",
    "karma-jasmine", "jasmine", "jasmine-core", "ava", "tap", "tape", "uvu", "c8",
    "@testing-library/react", "@testing-library/jest-dom", "@testing-library/user-event",
    "@testing-library/dom", "enzyme", "enzyme-adapter-react-16", "@wojtekmaj/enzyme-adapter-react-17",
    "react-test-renderer", "preact", "preact-compat", "htm", "solid-js", "svelte", "svelte-preprocess",
    "svelte-check", "svelte-loader", "angular", "@angular/core", "@angular/common", "@angular/compiler",
    "@angular/compiler-cli", "@angular/forms", "@angular/platform-browser", "@angular/platform-browser-dynamic",
    "@angular/router", "@angular/animations", "@angular/cli", "@angular/material", "@angular/cdk",
    "zone.js", "rxjs", "inversify", "inversify-express-utils", "awilix", "tsyringe",
    "fastify", "fastify-plugin", "fastify-cors", "fastify-swagger", "@fastify/cors", "@fastify/swagger",
    "fast-glob", "globby", "glob", "micromatch", "picomatch", "chokidar", "watchpack",
    "koa", "koa-router", "koa-bodyparser", "koa-static", "koa-compose", "koa-convert",
    "express-session", "cookie-parser", "body-parser", "multer", "compression", "helmet",
    "cors", "morgan", "serve-static", "finalhandler", "fresh", "etag", "accepts", "negotiator",
    "qs", "querystring", "querystringify", "url-parse", "tough-cookie", "cookie", "cookie-signature",
    "vary", "on-finished", "destroy", "bytes", "content-type", "content-disposition", "depd",
    "ee-first", "encodeurl", "escape-html", "etag", "forwarded", "ipaddr.js", "media-typer",
    "merge-descriptors", "methods", "mime", "mime-types", "mime-db", "parseurl", "path-to-regexp",
    "range-parser", "send", "serve-index", "setprototypeof", "statuses", "type-is", "utils-merge",
    "sequelize", "sequelize-cli", "mongoose", "knex", "objection", "prisma", "@prisma/engines",
    "typeorm", "typeorm-aurora-data-api-driver", "bookshelf", "waterline", "sails-disk",
    "mongodb", "mysql", "mysql2", "pg", "pg-hstore", "pg-pool", "better-sqlite3", "sqlite3",
    "redis", "ioredis", "amqplib", "mqtt", "bull", "bullmq", "node-cron", "agenda", "node-schedule",
    "lodash", "lodash-es", "underscore", "ramda", "date-fns", "dayjs", "moment", "moment-timezone",
    "luxon", "chrono-node", "uuid", "uuidv4", "nanoid", "shortid", "crypto-js", "js-cookie",
    "axios", "node-fetch", "cross-fetch", "isomorphic-fetch", "got", "request", "request-promise-native",
    "undici", "http2-wrapper", "hpagent", "form-data", "busboy", "multiparty", "formidable",
    "jsonwebtoken", "jws", "jwa", "ecdsa-sig-formatter", "safe-buffer", "base64url", "bcryptjs",
    "bcrypt", "scrypt-js", "crypto-random-string", "random-bytes", "uid-safe", "sshpk", "jsbn",
    "tweetnacl", "tweetnacl-util", "nacl", "sodium-native", "libsodium-wrappers",
    "@aws-sdk/client-lambda", "@aws-sdk/client-sqs", "@aws-sdk/client-sns", "@aws-sdk/client-ec2",
    "@aws-sdk/client-iam", "@aws-sdk/client-ssm", "@aws-sdk/client-secrets-manager",
    "@aws-sdk/client-cloudwatch", "@aws-sdk/client-dynamodb", "@aws-sdk/util-dynamodb",
    "@aws-sdk/signature-v4", "@aws-sdk/credential-providers", "@aws-sdk/middleware-serde",
    "@aws-sdk/middleware-signing", "@aws-sdk/config-resolver", "@aws-sdk/node-http-handler",
    "@aws-sdk/fetch-http-handler", "@aws-sdk/abort-controller", "@aws-sdk/protocol-http",
    "@aws-sdk/querystring-builder", "@aws-sdk/smithy-client", "@aws-sdk/util-base64",
    "@aws-sdk/util-utf8", "@aws-sdk/types", "@aws-sdk/util-hex-encoding",
    "@azure/msal-node", "@azure/msal-browser", "@azure/msal-common", "@azure/core-http",
    "@azure/core-rest-pipeline", "@azure/core-client", "@azure/core-auth", "@azure/logger",
    "@azure/cosmos", "@azure/storage-queue", "@azure/storage-file-share", "@azure/service-bus",
    "@azure/event-hubs", "@azure/arm-resources", "@azure/arm-storage",
    "firebase", "firebase-admin", "firebase-functions", "@firebase/app", "@firebase/auth",
    "@firebase/firestore", "@firebase/database", "@firebase/storage", "@firebase/messaging",
    "@firebase/analytics", "@firebase/util", "@firebase/logger", "@firebase/component",
    "@google-cloud/storage", "@google-cloud/firestore", "@google-cloud/pubsub", "@google-cloud/bigquery",
    "google-auth-library", "gaxios", "gtoken", "google-p12-pem", "jws",
    "graphql", "graphql-request", "@graphql-tools/schema", "@graphql-tools/utils", "@graphql-tools/merge",
    "@graphql-tools/load", "@graphql-tools/graphql-file-loader", "@graphql-tools/code-file-loader",
    "graphql-yoga", "apollo-server-core", "apollo-server-express", "apollo-server", "@apollo/subgraph",
    "@apollo/gateway", "graphql-subscriptions", "graphql-ws", "express-graphql", "mercurius",
    "dataloader", "graphql-relay", "graphql-tools", "graphql-iso-date", "graphql-upload",
    "next", "nuxt", "nuxt3", "@nuxtjs/axios", "@nuxtjs/proxy", "remix", "@remix-run/react",
    "@remix-run/node", "@remix-run/express", "sapper", "svelte-kit", "@sveltejs/kit", "@sveltejs/adapter-node",
    "astro", "@astrojs/react", "@astrojs/node", "@astrojs/vercel", "eleventy", "@11ty/eleventy",
    "hexo", "hexo-cli", "hexo-generator-archive", "hexo-generator-category", "hexo-generator-index",
    "hexo-generator-tag", "hexo-renderer-marked", "hexo-renderer-stylus", "hexo-server",
    "vitepress", "docusaurus", "@docusaurus/core", "@docusaurus/theme-classic", "@docusaurus/preset-classic",
    "@docusaurus/mdx-loader", "@docusaurus/types", "@docusaurus/utils", "remark", "remark-parse",
    "remark-rehype", "rehype", "rehype-parse", "rehype-raw", "rehype-sanitize", "rehype-slug",
    "rehype-autolink-headings", "micromark", "mdast-util-from-markdown", "hast-util-to-html",
    "esbuild", "@esbuild/linux-x64", "rollup", "@rollup/plugin-node-resolve", "@rollup/plugin-commonjs",
    "@rollup/plugin-babel", "@rollup/plugin-json", "@rollup/plugin-url", "@rollup/plugin-replace",
    "rollup-plugin-terser", "rollup-plugin-postcss", "rollup-plugin-visualizer", "rollup-plugin-node-externals",
    "parcel", "@parcel/core", "@parcel/transformer-js", "@parcel/transformer-html", "@parcel/transformer-css",
    "@parcel/transformer-react-refresh-wrap", "@parcel/optimizer-terser", "@parcel/packager-js",
    "@parcel/transformer-image", "@parcel/transformer-sass", "vite", "vite-plugin-react", "@vitejs/plugin-react",
    "@vitejs/plugin-legacy", "vite-plugin-pwa", "vite-plugin-mdx", "vite-plugin-svg-icons",
    "tsup", "microbundle", "unbuild", "tsdx", "dts-bundle-generator", "typescript", "ts-node",
    "tsconfig-paths", "tslib", "typescript-eslint", "@typescript-eslint/types", "@typescript-eslint/typescript-estree",
    "@typescript-eslint/scope-manager", "@typescript-eslint/utils", "@typescript-eslint/visitor-keys",
    "eslint", "eslint-scope", "eslint-visitor-keys", "espree", "estraverse", "esutils",
    "esquery", "acorn", "acorn-jsx", "acorn-globals", "acorn-walk", "acorn-loose",
    "terser", "terser-webpack-plugin", "uglify-js", "uglify-es", "html-minifier-terser", "clean-css",
    "csso", "postcss", "postcss-safe-parser", "postcss-value-parser", "postcss-selector-parser",
    "postcss-modules", "postcss-modules-local-by-default", "postcss-modules-extract-imports",
    "postcss-modules-scope", "postcss-modules-values", "cssnano", "cssnano-preset-lite",
    "csstype", "cssesc", "css-what", "nth-check", "boolbase", "domhandler", "domutils",
    "dom-serializer", "domelementtype", "entities", "htmlparser2", "parse5", "parse5-htmlparser2-tree-adapter",
    "cheerio", "linkedom", "jsdom", "happy-dom", "xmldom", "saxes", "xmlchars",
    "webpack", "webpack-cli", "webpack-sources", "webpack-sources3", "webpack-virtual-modules",
    "webpack-bundle-analyzer", "webpack-stats-plugin", "webpackbar", "speed-measure-webpack-plugin",
    "size-limit", "@size-limit/preset-app", "bundle-stats", "bundle-size", "compression-webpack-plugin",
    "speed-measure-webpack-plugin", "circular-dependency-plugin", "duplicate-package-checker-webpack-plugin",
    "eslint-webpack-plugin", "stylelint-webpack-plugin", "terser-webpack-plugin",
    "npx", "npm-run-all", "concurrently", "cross-env", "dotenv", "dotenv-cli", "env-cmd",
    "husky", "lint-staged", "prettier", "prettier-plugin-tailwindcss", "pretty-quick",
    "standard", "eslint-config-standard", "eslint-plugin-promise", "eslint-plugin-n", "eslint-plugin-node",
    "eslint-plugin-standard", "semistandard", "xo", "airbnb", "babel-preset-airbnb",
    "babel-plugin-transform-class-properties", "@babel/plugin-proposal-class-properties",
    "@babel/plugin-proposal-decorators", "@babel/plugin-proposal-object-rest-spread",
    "@babel/plugin-proposal-optional-chaining", "@babel/plugin-proposal-nullish-coalescing-operator",
    "@babel/plugin-proposal-private-methods", "@babel/plugin-proposal-export-default-from",
    "@babel/plugin-syntax-dynamic-import", "@babel/plugin-transform-modules-commonjs",
    "@babel/plugin-transform-typescript", "@babel/plugin-transform-react-jsx",
    "@babel/plugin-transform-runtime", "@babel/plugin-transform-regenerator",
    "@babel/preset-flow", "@babel/flow-parser", "@babel/traverse", "@babel/generator",
    "@babel/template", "@babel/parser", "@babel/types", "@babel/code-frame", "@babel/helper-*",
    "babel-preset-react-app", "babel-plugin-macros", "babel-plugin-named-asset-import",
    "react-app-rewired", "customize-cra", "craco", "@craco/craco",
    "yarn", "pnpm", "@pnpm/logger", "@pnpm/fs.indexed-pkg-importer", "@pnpm/store-path",
    "lerna", "@lerna/child-process", "@lerna/collect-updates", "@lerna/package-graph",
    "nx", "@nrwl/workspace", "@nrwl/web", "@nrwl/react", "@nrwl/node", "@nrwl/jest",
    "turborepo", "@turbo/gen", "turbo", "changesets", "@changesets/cli", "@changesets/assemble-release-plan",
    "semver", "semver-diff", "semver-utils", "compare-versions", "standard-version", "conventional-changelog",
    "conventional-commits-parser", "conventional-changelog-writer", "git-raw-commits", "git-semver-tags",
    "@commitlint/cli", "@commitlint/config-conventional", "@commitlint/lint", "@commitlint/read",
    "@commitlint/ensure", "commitizen", "cz-conventional-changelog", "release-it", "np",
    "gh-pages", "npm-publish", "semantic-release", "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator", "@semantic-release/npm", "@semantic-release/github",
    "source-map-support", "source-map", "source-map-js", "inline-source-map", "sorcery",
    "magic-string", "@jridgewell/trace-mapping", "@jridgewell/resolve-uri", "@jridgewell/sourcemap-codec",
    "@jridgewell/gen-mapping", "@jridgewell/set-array",
]


def _url(name: str) -> str:
    if name.startswith("@"):
        # Scoped packages on the huaweicloud mirror require a trailing slash;
        # keep the literal slash (do not percent-encode it).
        return BASE + name + "/"
    return BASE + name


def _fname(name: str) -> str:
    return name.replace("/", "__") + ".json"


def fetch_one(name: str) -> dict | None:
    path = RAW / _fname(name)
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))  # cached: still yield deps
        except Exception:  # noqa: BLE001
            pass  # corrupt cache: refetch below
    tmp = RAW / (_fname(name) + ".partial")
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(_url(name))
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = resp.read()
            tmp.write_bytes(data)
            tmp.rename(path)
            return json.loads(data)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"[fail] {name}: {last_err!r}", flush=True)
    return None


def deps_of(doc: dict) -> list[str]:
    vs = doc.get("versions", {})
    latest = (doc.get("dist-tags") or {}).get("latest")
    if not latest or latest not in vs:
        return []
    deps = vs[latest].get("dependencies") or {}
    out = []
    for k in deps:
        if not k.startswith("node:"):
            out.append(k)
    return out


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    seen = set()
    queue: list[str] = []
    for s in SEEDS:
        if s not in seen:
            seen.add(s)
            queue.append(s)

    fetched = 0
    failed = 0
    t0 = time.time()
    total_deps = 0
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"=== fetch start target={TARGET} workers={WORKERS} {time.ctime()} ===\n")
        while queue and fetched < TARGET:
            batch, queue = queue[: 4 * WORKERS], queue[4 * WORKERS:]
            results: dict[str, dict | None] = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {ex.submit(fetch_one, n): n for n in batch}
                for fut in as_completed(futs):
                    name = futs[fut]
                    doc = fut.result()
                    results[name] = doc
            new_nodes = []
            for name in batch:
                doc = results.get(name)
                if doc is None:
                    failed += 1
                    continue
                fetched += 1
                deps = deps_of(doc)
                total_deps += len(deps)
                for d in deps:
                    if d not in seen:
                        seen.add(d)
                        new_nodes.append(d)
            queue.extend(new_nodes)
            speed = fetched / max(time.time() - t0, 1e-9)
            eta = (TARGET - fetched) / max(speed, 1e-9) / 60
            msg = (f"fetched={fetched} queued={len(queue)} failed={failed} "
                   f"deps_seen={len(seen)} speed={speed:.1f}/s eta={eta:.0f}min")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        log.write(f"=== fetch end fetched={fetched} failed={failed} seen={len(seen)} "
                  f"total_deps={total_deps} {time.ctime()} ===\n")
    print(f"DONE fetched={fetched} failed={failed} seen={len(seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
