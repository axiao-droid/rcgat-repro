"""Fetch Maven Central artifact metadata in a BFS from seed artifacts.

Written for this reproduction (the project's ``fetch_maven.py`` was a stray
``.pyc`` of an unrelated package and could not be used).

Design (mirrors the documented protocol for the npm side):
  - node       = ``groupId:artifactId``
  - node date  = first publish  (earliest version directory date)
                 and latest release (newest version directory date / lastUpdated);
                 which one the loader treats as ``D(v)`` is configurable, because
                 the paper's realized Maven span (2021-04 .. 2026-09) is only
                 consistent with the *latest release* reading.
  - edge set   = compile-scope dependencies (``<scope>`` absent or ``compile``,
                 optional deps excluded) of the latest release POM, A -> B.
  - text       = ``<name>`` plus ``<description>`` from the latest release POM.
  - resumable  : ``raw_maven/<group>__<artifact>.json`` exists -> reused.

Usage::

    RAW_DIR=.../data/raw_maven TARGET=2500 WORKERS=8 python fetch_maven.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAW = Path(os.environ.get("RAW_DIR", str(Path(__file__).resolve().parent / "raw_maven"))).expanduser()
BASE = os.environ.get("MAVEN_BASE", "https://repo1.maven.org/maven2/")
TARGET = int(os.environ.get("TARGET", "2500"))
WORKERS = int(os.environ.get("WORKERS", "8"))
LOG = Path(os.environ.get("LOG_FILE", str(RAW.parent / "fetch_maven.log"))).expanduser()
UA = "Mozilla/5.0 (compatible; dependency-graph-research/1.0)"

SEEDS = [
    # spring
    "org.springframework:spring-core", "org.springframework:spring-context",
    "org.springframework:spring-web", "org.springframework:spring-webmvc",
    "org.springframework:spring-beans", "org.springframework.boot:spring-boot",
    "org.springframework.boot:spring-boot-starter-web", "org.springframework.boot:spring-boot-starter-data-jpa",
    "org.springframework.boot:spring-boot-starter-security", "org.springframework.boot:spring-boot-starter-test",
    "org.springframework.data:spring-data-jpa", "org.springframework.data:spring-data-commons",
    "org.springframework.security:spring-security-core", "org.springframework.security:spring-security-web",
    "org.springframework:spring-jdbc", "org.springframework:spring-tx", "org.springframework:spring-aop",
    "org.springframework.kafka:spring-kafka", "org.springframework.cloud:spring-cloud-starter-netflix-eureka-client",
    # testing
    "org.junit.jupiter:junit-jupiter-api", "org.junit.jupiter:junit-jupiter-engine",
    "org.junit.jupiter:junit-jupiter-params", "junit:junit", "org.junit.vintage:junit-vintage-engine",
    "org.mockito:mockito-core", "org.assertj:assertj-core", "org.hamcrest:hamcrest",
    "org.testng:testng", "org.spockframework:spock-core", "io.cucumber:cucumber-java",
    "org.testcontainers:testcontainers", "org.testcontainers:junit-jupiter",
    "com.github.tomakehurst:wiremock", "io.rest-assured:rest-assured", "org.seleniumhq.selenium:selenium-java",
    "io.github.bonigarcia:webdrivermanager", "org.pitest:pitest", "org.jacoco:org.jacoco.core",
    # json / xml / text
    "com.fasterxml.jackson.core:jackson-databind", "com.fasterxml.jackson.core:jackson-core",
    "com.fasterxml.jackson.core:jackson-annotations", "com.fasterxml.jackson.dataformat:jackson-dataformat-xml",
    "com.fasterxml.jackson.dataformat:jackson-dataformat-yaml", "com.fasterxml.jackson.datatype:jackson-datatype-jsr310",
    "com.google.code.gson:gson", "org.json:json", "com.jayway.jsonpath:json-path",
    "jakarta.xml.bind:jakarta.xml.bind-api", "javax.xml.bind:jaxb-api", "com.thoughtworks.xstream:xstream",
    "org.dom4j:dom4j", "jdom:jdom", "org.jsoup:jsoup", "de.odysseus.staxon:staxon",
    "com.opencsv:opencsv", "org.apache.commons:commons-csv", "org.yaml:snakeyaml",
    "com.typesafe:config", "org.tomlj:tomlj",
    # commons / utils
    "org.apache.commons:commons-lang3", "commons-io:commons-io", "commons-codec:commons-codec",
    "org.apache.commons:commons-collections4", "commons-beanutils:commons-beanutils",
    "commons-cli:commons-cli", "commons-logging:commons-logging", "org.apache.commons:commons-text",
    "org.apache.commons:commons-math3", "com.google.guava:guava", "org.apache.commons:commons-compress",
    "commons-validator:commons-validator", "org.apache.commons:commons-exec", "com.google.code.findbugs:jsr305",
    "org.checkerframework:checker-qual", "com.google.errorprone:error_prone_annotations",
    "org.projectlombok:lombok", "org.mapstruct:mapstruct", "org.jetbrains:annotations",
    "org.joda:joda-convert", "joda-time:joda-time", "org.apache.commons:commons-pool2",
    # logging
    "org.slf4j:slf4j-api", "org.slf4j:slf4j-simple", "ch.qos.logback:logback-classic",
    "ch.qos.logback:logback-core", "org.apache.logging.log4j:log4j-core",
    "org.apache.logging.log4j:log4j-api", "org.apache.logging.log4j:log4j-slf4j2-impl",
    "org.apache.logging.log4j:log4j-slf4j-impl", "com.google.flogger:flogger", "org.tinylog:tinylog-api",
    # http / web clients and servers
    "org.apache.httpcomponents.client5:httpclient5", "org.apache.httpcomponents.core5:httpcore5",
    "org.apache.httpcomponents:httpclient", "com.squareup.okhttp3:okhttp",
    "com.squareup.retrofit2:retrofit", "io.github.openfeign:feign-core",
    "org.eclipse.jetty:jetty-server", "org.eclipse.jetty:jetty-servlet",
    "org.apache.tomcat.embed:tomcat-embed-core", "io.undertow:undertow-core",
    "org.apache.cxf:cxf-core", "org.jboss.resteasy:resteasy-core",
    "jakarta.servlet:jakarta.servlet-api", "javax.servlet:javax.servlet-api",
    "io.netty:netty-all", "io.netty:netty-handler", "io.netty:netty-codec-http",
    "io.netty:netty-transport", "org.eclipse.jetty.http2:http2-server",
    "org.apache.httpcomponents:httpasyncclient", "com.google.api-client:google-api-client",
    # reactive / concurrency
    "io.projectreactor:reactor-core", "io.projectreactor.netty:reactor-netty",
    "io.reactivex.rxjava3:rxjava", "io.reactivex.rxjava2:rxjava", "org.reactivestreams:reactive-streams",
    "com.typesafe.akka:akka-actor_2.13", "io.vertx:vertx-core", "io.vertx:vertx-web",
    "org.jctools:jctools-core", "com.google.guava:failureaccess", "net.jodah:failsafe",
    "io.github.resilience4j:resilience4j-circuitbreaker", "io.github.resilience4j:resilience4j-core",
    # databases / persistence
    "org.hibernate.orm:hibernate-core", "org.hibernate:hibernate-core",
    "org.hibernate.validator:hibernate-validator", "jakarta.persistence:jakarta.persistence-api",
    "javax.persistence:javax.persistence-api", "org.mybatis:mybatis", "org.mybatis:mybatis-spring",
    "org.jooq:jooq", "com.zaxxer:HikariCP", "com.alibaba:druid", "c3p0:c3p0",
    "org.postgresql:postgresql", "com.mysql:mysql-connector-j", "mysql:mysql-connector-java",
    "org.mariadb.jdbc:mariadb-java-client", "com.h2database:h2", "org.xerial:sqlite-jdbc",
    "org.mongodb:mongodb-driver-sync", "org.mongodb:mongodb-driver-core", "org.mongodb:bson",
    "redis.clients:jedis", "io.lettuce:lettuce-core", "org.redisson:redisson",
    "org.liquibase:liquibase-core", "org.flywaydb:flyway-core", "org.apache.ibatis:ibatis-core",
    # search / analytics / big data
    "org.apache.lucene:lucene-core", "org.apache.lucene:lucene-analyzers-common",
    "co.elastic.clients:elasticsearch-java", "org.elasticsearch.client:elasticsearch-rest-client",
    "org.apache.solr:solr-core", "org.apache.tika:tika-core", "org.apache.poi:poi",
    "org.apache.poi:poi-ooxml", "org.apache.pdfbox:pdfbox", "com.itextpdf:itextpdf",
    "org.apache.parquet:parquet-common", "org.apache.avro:avro", "org.apache.thrift:libthrift",
    "com.google.protobuf:protobuf-java", "io.grpc:grpc-netty", "io.grpc:grpc-protobuf",
    "io.grpc:grpc-core", "io.grpc:grpc-stub", "org.apache.hadoop:hadoop-common",
    "org.apache.spark:spark-core_2.13", "org.apache.flink:flink-core", "org.apache.beam:beam-sdks-java-core",
    "org.apache.camel:camel-core", "org.apache.kafka:kafka-clients", "org.apache.pulsar:pulsar-client",
    "com.rabbitmq:amqp-client", "org.apache.zookeeper:zookeeper", "org.apache.curator:curator-framework",
    "org.apache.hbase:hbase-client", "org.apache.cassandra:cassandra-all",
    # cloud / infra sdks
    "software.amazon.awssdk:s3", "software.amazon.awssdk:core", "software.amazon.awssdk:dynamodb",
    "com.amazonaws:aws-java-sdk-s3", "com.amazonaws:aws-java-sdk-core",
    "com.azure:azure-storage-blob", "com.google.cloud:google-cloud-storage",
    "com.google.cloud:google-cloud-firestore", "io.fabric8:kubernetes-client",
    "io.kubernetes:client-java", "com.github.docker-java:docker-java",
    # jvm languages / frameworks
    "org.jetbrains.kotlin:kotlin-stdlib", "org.jetbrains.kotlin:kotlin-reflect",
    "org.scala-lang:scala-library", "org.jetbrains.kotlinx:kotlinx-coroutines-core",
    "io.quarkus:quarkus-core", "io.micronaut:micronaut-core", "io.dropwizard:dropwizard-core",
    "org.apache.struts:struts2-core", "org.primefaces:primefaces", "org.apache.myfaces.core:myfaces-api",
    "com.vaadin:vaadin-core", "org.grails:grails-core",
    # security / crypto
    "io.jsonwebtoken:jjwt-api", "io.jsonwebtoken:jjwt-impl", "com.auth0:java-jwt",
    "org.bouncycastle:bcprov-jdk18on", "org.bouncycastle:bcpkix-jdk18on",
    "org.apache.shiro:shiro-core", "com.nimbusds:nimbus-jose-jwt",
    # misc modern
    "org.apache.commons:commons-rng-core", "com.github.ben-manes.caffeine:caffeine",
    "org.ehcache:ehcache", "net.sf.ehcache:ehcache", "com.j256.ormlite:ormlite-core",
    "org.openjdk.jmh:jmh-core", "org.openjdk.jol:jol-core", "org.ow2.asm:asm",
    "org.ow2.asm:asm-tree", "org.ow2.asm:asm-commons", "net.bytebuddy:byte-buddy",
    "org.javassist:javassist", "cglib:cglib", "org.apache.groovy:groovy",
    "org.codehaus.groovy:groovy", "org.springframework:spring-instrument",
    "org.apiguardian:apiguardian-api", "org.opentest4j:opentest4j",
]

SEEDS_FILE = os.environ.get("SEEDS_FILE")
if SEEDS_FILE:
    _extra = Path(SEEDS_FILE).expanduser()
    if _extra.exists():
        _seen = set(SEEDS)
        for _line in _extra.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and ":" in _line and _line not in _seen:
                _seen.add(_line)
                SEEDS.append(_line)
        print(f"extra seeds loaded: {len(SEEDS)} total", flush=True)


_VERSION_RE = re.compile(r'href="(?P<name>[^"/]+)/"[^>]*>[^<]*</a>\s*(?P<date>\d{4}-\d{2}-\d{2})')
_DEP_SKIP_SCOPES = {"test", "provided", "system", "import"}
# prerelease markers: prefer the newest *release* version, like a Maven build does
_PRERELEASE_RE = re.compile(
    r"(?i)(?:^|[-._])(?:M\d+|RC\d*|alpha\d*|beta\d*|preview|ea|cr\d*|snapshot|milestone|incubating)(?:$|[-._])"
)


def is_prerelease(version: str) -> bool:
    return bool(_PRERELEASE_RE.search(version))


def _split(ga: str) -> tuple[str, str]:
    g, a = ga.split(":", 1)
    return g, a


def _path(ga: str) -> str:
    g, a = _split(ga)
    return f"{g.replace('.', '/')}/{a}/"


def _fname(ga: str) -> str:
    g, a = _split(ga)
    return f"{g}__{a}.json"


def _get(url: str, binary: bool = False) -> bytes | str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = resp.read()
    return data if binary else data.decode("utf-8", errors="replace")


def fetch_one(ga: str) -> dict | None:
    path = RAW / _fname(ga)
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    last_err: object = None
    for attempt in range(3):
        try:
            listing = _get(BASE + _path(ga))
            versions = [
                {"version": m.group("name"), "date": m.group("date")}
                for m in _VERSION_RE.finditer(listing or "")
            ]
            versions = [v for v in versions if v["version"] != ".." and "-SNAPSHOT" not in v["version"]]
            if not versions:
                raise RuntimeError("no version directories")
            versions.sort(key=lambda v: v["date"])
            releases = [v for v in versions if not is_prerelease(v["version"])]
            latest = (releases or versions)[-1]["version"]
            pom_url = f"{BASE}{_path(ga)}{latest}/{_split(ga)[1]}-{latest}.pom"
            try:
                pom = _get(pom_url)
            except urllib.error.HTTPError:
                # fall back to the newest version that has a readable POM
                pom = None
                for v in reversed(releases or versions):
                    try:
                        pom = _get(f"{BASE}{_path(ga)}{v['version']}/{_split(ga)[1]}-{v['version']}.pom")
                        latest = v["version"]
                        break
                    except urllib.error.HTTPError:
                        continue
                if pom is None:
                    raise RuntimeError("no readable POM")
            record = _parse(ga, latest, versions, pom)
            tmp = path.with_suffix(".partial")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.rename(path)
            return record
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    print(f"[fail] {ga}: {last_err!r}", flush=True)
    return None


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse(ga: str, latest: str, versions: list[dict], pom_text: str) -> dict:
    g, a = _split(ga)
    name, description = "", ""
    deps: dict[str, str] = {}
    try:
        root = ET.fromstring(pom_text)
    except ET.ParseError:
        root = None
    if root is not None:
        parent_group = ""
        for child in root:
            if _strip_ns(child.tag) == "parent":
                for sub in child:
                    if _strip_ns(sub.tag) == "groupId":
                        parent_group = (sub.text or "").strip()
        for child in root:
            tag = _strip_ns(child.tag)
            if tag == "name" and child.text:
                name = " ".join(child.text.split())
            elif tag == "description" and child.text:
                description = " ".join(child.text.split())
            elif tag == "dependencies":
                for dep in child:
                    if _strip_ns(dep.tag) != "dependency":
                        continue
                    dg = da = scope = ver = ""
                    optional = "false"
                    for field in dep:
                        ftag = _strip_ns(field.tag)
                        val = (field.text or "").strip()
                        if ftag == "groupId":
                            dg = val
                        elif ftag == "artifactId":
                            da = val
                        elif ftag == "scope":
                            scope = val
                        elif ftag == "version":
                            ver = val
                        elif ftag == "optional":
                            optional = val
                    if not dg:
                        dg = parent_group
                    if not dg or not da or "${" in dg or "${" in da:
                        continue
                    if scope.lower() in _DEP_SKIP_SCOPES or optional.lower() == "true":
                        continue
                    dep_ga = f"{dg}:{da}"
                    if dep_ga == ga:
                        continue
                    deps[dep_ga] = ver
    return {
        "ga": ga,
        "group": g,
        "artifact": a,
        "latest": latest,
        "first_date": versions[0]["date"],
        "latest_date": versions[-1]["date"],
        "latest_is_prerelease": is_prerelease(latest),
        "n_versions": len(versions),
        "name": name,
        "description": description,
        "deps": deps,
    }


def deps_of(doc: dict) -> list[str]:
    return list((doc.get("deps") or {}).keys())


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    queue: list[str] = []
    for s in SEEDS:
        if s not in seen:
            seen.add(s)
            queue.append(s)
    fetched = failed = 0
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"=== maven fetch start target={TARGET} workers={WORKERS} {time.ctime()} ===\n")
        while queue and fetched < TARGET:
            batch, queue = queue[: 4 * WORKERS], queue[4 * WORKERS:]
            results: dict[str, dict | None] = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {ex.submit(fetch_one, ga): ga for ga in batch}
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
            new_nodes: list[str] = []
            for ga in batch:
                doc = results.get(ga)
                if doc is None:
                    failed += 1
                    continue
                fetched += 1
                for dep in deps_of(doc):
                    if dep not in seen:
                        seen.add(dep)
                        new_nodes.append(dep)
            queue.extend(new_nodes)
            speed = fetched / max(time.time() - t0, 1e-9)
            msg = (f"fetched={fetched} queued={len(queue)} failed={failed} "
                   f"deps_seen={len(seen)} speed={speed:.1f}/s "
                   f"eta={(TARGET - fetched) / max(speed, 1e-9) / 60:.0f}min")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        log.write(f"=== maven fetch end fetched={fetched} failed={failed} seen={len(seen)} {time.ctime()} ===\n")
    print(f"DONE fetched={fetched} failed={failed} seen={len(seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
