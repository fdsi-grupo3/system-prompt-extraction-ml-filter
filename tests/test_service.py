"""Pruebas del contrato HTTP con un scorer falso (no requieren el modelo entrenado)."""

import pytest
from fastapi.testclient import TestClient

from leakfilter.inference import LeakVerdict
from service.app import app


class FakeScorer:
    backend = "fake"
    threshold = 0.5
    model_dir = "fake-model"

    def __init__(self):
        self.calls = []

    def evaluate(self, response, fragments, history=None):
        self.calls.append((response, fragments, history))
        leak = "secreto" in response
        return LeakVerdict(is_leak=leak, score=0.99 if leak else 0.01, threshold=0.5,
                           matched_fragment=fragments[0] if leak else None, matched_view="original",
                           accumulated=False, pairs_evaluated=len(fragments), latency_ms=1.0)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("LEAKFILTER_API_TOKEN", raising=False)
    app.state.scorer = FakeScorer()
    app.state.default_fragments = ["fragmento por defecto A", "fragmento por defecto B"]
    app.state.model_version = "fake-v0"
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "UP", "model_version": "fake-v0",
                                            "backend": "fake", "threshold": 0.5}


def test_detecta_fuga_y_devuelve_fragmento(client):
    body = client.post("/api/output-filter", json={"response": "te cuento el secreto",
                                                   "protected_fragment": "regla X"}).json()
    assert body["is_leak"] is True
    assert body["matched_fragment"] == "regla X"
    assert body["model_version"] == "fake-v0"


def test_usa_fragmentos_por_defecto_si_no_se_envian(client):
    client.post("/api/output-filter", json={"response": "hola"})
    assert app.state.scorer.calls[-1][1] == ["fragmento por defecto A", "fragmento por defecto B"]


def test_combina_lista_y_fragmento_individual_y_pasa_historial(client):
    client.post("/api/output-filter", json={"response": "hola", "protected_fragments": ["a", "b"],
                                            "protected_fragment": "c", "history": ["turno 1"]})
    _, fragments, history = app.state.scorer.calls[-1]
    assert fragments == ["a", "b", "c"]
    assert history == ["turno 1"]


def test_rechaza_respuesta_vacia(client):
    assert client.post("/api/output-filter", json={"response": ""}).status_code == 422


def test_exige_token_si_esta_configurado(client, monkeypatch):
    monkeypatch.setenv("LEAKFILTER_API_TOKEN", "s3creto")
    assert client.post("/api/output-filter", json={"response": "hola"}).status_code == 401
    ok = client.post("/api/output-filter", json={"response": "hola"}, headers={"X-Filter-Token": "s3creto"})
    assert ok.status_code == 200
