"""INDIGO ATLAS Sprint 01 / WS2: Graphiti temporal regression gate.

Runs against real graphiti-core 0.31.2 with the embedded Kuzu driver. No DB server, no network.
Episodes are ingested as graphiti EntityNode / EntityEdge objects that the harness builds by hand.
Each edge carries valid_at / invalid_at / episodes=[source_id]. Supersession uses graphiti's own
`resolve_edge_contradictions`. LLM extraction is replaced by a deterministic mock: the harness
maps each episode straight to an edge. The full `Graphiti.add_episode` LLM pipeline is NOT run.
A NetworkX / dict reference model is used as the comparison oracle.
"""
from __future__ import annotations

import hashlib
import json
import socket
from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest

from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.utils.datetime_utils import ensure_utc
from graphiti_core.errors import GroupsEdgesNotFoundError
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.edge_operations import resolve_edge_contradictions

FIXTURE = json.loads((Path(__file__).parent.parent / "fixtures" / "graphiti-temporal-fixture.json").read_text())
GROUP = "indigo-ws2"
# Late-arriving correction: observed after ep3 but valid earlier, between ep1 and ep2.
LATE = {"id": "ep5", "observed_at": "2026-04-01T00:00:00Z", "valid_from": "2026-01-20T00:00:00Z",
        "entity": "Transpiler::Lua->Python::switch", "claim": "partially_supported", "source": "synthetic/ep5"}


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------- deterministic fakes
class FakeEmbedder(EmbedderClient):
    """Deterministic unit vector derived from sha256(text). Never touches the network."""
    async def create(self, input_data):
        h = hashlib.sha256(str(input_data).encode()).digest()
        v = [b / 255.0 for b in h[:16]]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    async def create_batch(self, input_data_list):
        return [await self.create(x) for x in input_data_list]


class FakeLLM(LLMClient):
    """Canned extraction: returns the episode's (entity, claim) unchanged. Counts calls."""
    def __init__(self):
        super().__init__(LLMConfig(api_key="none", model="fake"), cache=False)
        self.calls = 0

    async def _generate_response(self, messages, response_model=None, max_tokens=0, model_size=None):
        self.calls += 1
        return {"extracted": "canned"}

    def extract(self, ep: dict) -> tuple[str, str]:
        self.calls += 1
        return ep["entity"], ep["claim"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def guard(*a, **k):
        raise RuntimeError("network access attempted during zero-cost benchmark")
    monkeypatch.setattr(socket.socket, "connect", guard)


# ---------------------------------------------------------------- graphiti harness
class GraphitiHarness:
    def __init__(self, db_path: str):
        self.driver = KuzuDriver(db=db_path)
        self.llm, self.emb = FakeLLM(), FakeEmbedder()
        self.nodes: dict[str, EntityNode] = {}

    async def setup(self):
        await self.driver.build_indices_and_constraints()

    async def node(self, name: str) -> EntityNode:
        if name not in self.nodes:
            n = EntityNode(name=name, group_id=GROUP, labels=["Entity"], summary="",
                           name_embedding=await self.emb.create(name))
            await n.save(self.driver)
            self.nodes[name] = n
        return self.nodes[name]

    async def edges(self) -> list[EntityEdge]:
        try:
            out = await EntityEdge.get_by_group_ids(self.driver, [GROUP])
        except GroupsEdgesNotFoundError:
            return []
        # Finding: Kuzu TIMESTAMP round-trips as offset-naive datetimes; normalise to UTC.
        for e in out:
            e.valid_at, e.invalid_at = ensure_utc(e.valid_at), ensure_utc(e.invalid_at)
            e.created_at, e.expired_at = ensure_utc(e.created_at), ensure_utc(e.expired_at)
        return out

    async def ingest(self, ep: dict):
        entity, claim = self.llm.extract(ep)
        src, tgt = await self.node(entity), await self.node(f"status:{claim}")
        new = EntityEdge(group_id=GROUP, source_node_uuid=src.uuid, target_node_uuid=tgt.uuid,
                         name="HAS_STATUS", fact=claim, fact_embedding=await self.emb.create(claim),
                         episodes=[ep["source"]], created_at=ts(ep["observed_at"]),
                         valid_at=ts(ep["valid_from"]), invalid_at=None, attributes={"entity": entity})
        same = [e for e in await self.edges() if e.source_node_uuid == src.uuid]
        # graphiti's own contradiction logic expires older edges, but never deletes them
        for e in resolve_edge_contradictions(new, same):
            await e.save(self.driver)
        # late arrival: a strictly-later existing edge bounds the new edge
        later = [ts_ for ts_ in (e.valid_at for e in same) if ts_ and ts_ > new.valid_at]
        if later:
            new.invalid_at = min(later)
        await new.save(self.driver)

    async def as_of(self, entity: str, at: str):
        t = ts(at)
        hits = [e for e in await self.edges() if e.attributes.get("entity") == entity
                and e.valid_at <= t and (e.invalid_at is None or t < e.invalid_at)]
        if not hits:
            return None  # abstain
        e = max(hits, key=lambda e: e.valid_at)
        return e.fact, e.episodes[0]


# ---------------------------------------------------------------- reference (NetworkX + dict)
class Reference:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def ingest(self, ep):
        self.g.add_edge(ep["entity"], ep["claim"], valid_from=ts(ep["valid_from"]), source=ep["source"])

    def as_of(self, entity, at):
        t = ts(at)
        if entity not in self.g:
            return None
        c = [(d["valid_from"], v, d["source"]) for _, v, d in self.g.out_edges(entity, data=True) if d["valid_from"] <= t]
        if not c:
            return None
        _, claim, src = max(c)
        return claim, src


@pytest.fixture
async def gh(tmp_path):
    h = GraphitiHarness(str(tmp_path / "graph.kuzu"))
    await h.setup()
    for ep in FIXTURE["episodes"]:
        await h.ingest(ep)
    return h


@pytest.fixture
def ref():
    r = Reference()
    for ep in FIXTURE["episodes"]:
        r.ingest(ep)
    return r


Q = FIXTURE["queries"]


@pytest.mark.parametrize("q", Q, ids=[f"{q['entity'].split('::')[1]}@{q['at'][:10]}" for q in Q])
async def test_as_of_query_claim(gh, q):
    assert (await gh.as_of(q["entity"], q["at"]))[0] == q["expected"]


@pytest.mark.parametrize("q", Q, ids=[f"src-{i+1}" for i in range(len(Q))])
async def test_source_provenance(gh, q):
    assert (await gh.as_of(q["entity"], q["at"]))[1] == q["expected_source"]


async def test_reference_agrees_with_graphiti(gh, ref):
    for q in Q:
        assert await gh.as_of(q["entity"], q["at"]) == ref.as_of(q["entity"], q["at"]) == (q["expected"], q["expected_source"])


async def test_history_retained_not_deleted(gh):
    lua = sorted((e.valid_at, e.fact) for e in await gh.edges() if e.attributes["entity"].startswith("Transpiler::Lua"))
    assert [f for _, f in lua] == ["unsupported", "supported", "regressed"]  # all 3 historical claims kept


async def test_unrelated_fact_not_invalidated(gh):
    c = [e for e in await gh.edges() if e.attributes["entity"] == "Transpiler::C->CSharp::loop"]
    assert len(c) == 1 and c[0].invalid_at is None and c[0].expired_at is None


async def test_abstain_before_first_episode_and_unknown_entity(gh, ref):
    assert await gh.as_of("Transpiler::Lua->Python::switch", "2025-12-31T00:00:00Z") is None
    assert await gh.as_of("Transpiler::Rust->Go::match", "2026-03-15T00:00:00Z") is None
    assert ref.as_of("Transpiler::Rust->Go::match", "2026-03-15T00:00:00Z") is None


async def test_restart_persistence(tmp_path):
    db = str(tmp_path / "persist.kuzu")
    h = GraphitiHarness(db)
    await h.setup()
    for ep in FIXTURE["episodes"]:
        await h.ingest(ep)
    before = sorted((e.uuid, e.fact, str(e.invalid_at)) for e in await h.edges())
    await h.driver.close()
    del h
    h2 = GraphitiHarness(db)  # fresh driver on the same on-disk database
    after = sorted((e.uuid, e.fact, str(e.invalid_at)) for e in await h2.edges())
    assert before == after and len(after) == 4
    for q in Q:
        assert await h2.as_of(q["entity"], q["at"]) == (q["expected"], q["expected_source"])


async def test_bitemporal_late_correction(gh):
    await gh.ingest(LATE)
    e = "Transpiler::Lua->Python::switch"
    assert await gh.as_of(e, "2026-01-15T00:00:00Z") == ("unsupported", "synthetic/ep1")
    assert await gh.as_of(e, "2026-01-25T00:00:00Z") == ("partially_supported", "synthetic/ep5")
    assert await gh.as_of(e, "2026-02-15T00:00:00Z") == ("supported", "synthetic/ep2")
    assert await gh.as_of(e, "2026-03-15T00:00:00Z") == ("regressed", "synthetic/ep3")
    late = [x for x in await gh.edges() if x.episodes == ["synthetic/ep5"]][0]
    assert late.created_at > late.valid_at  # transaction time after valid time


async def test_zero_external_calls(gh):
    assert gh.llm.calls == len(FIXTURE["episodes"])  # only canned in-process extraction
    assert len(await gh.emb.create("x")) == 16
