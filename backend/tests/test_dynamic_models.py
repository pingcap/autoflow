import threading
from app.models.entity import get_dynamic_entity_model
from app.models.relationship import get_dynamic_relationship_model
from app.models.chunk import get_dynamic_chunk_model


def dynamic_model_creation(dim, ns):
    entity_model = get_dynamic_entity_model(dim, ns)
    relationship_model = get_dynamic_relationship_model(dim, ns, entity_model)
    chunk_model = get_dynamic_chunk_model(dim, ns)
    return entity_model, relationship_model, chunk_model


def test_concurrent_dynamic_model_creation():
    results = [None] * 10
    threads = []
    for i in range(10):
        t = threading.Thread(
            target=lambda idx: results.__setitem__(
                idx, dynamic_model_creation(128, "test")
            ),
            args=(i,),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Ensure each model is created only once across all threads
    entity_models, relationship_models, chunk_models = zip(*results)
    assert all(m is entity_models[0] for m in entity_models)
    assert all(m is relationship_models[0] for m in relationship_models)
    assert all(m is chunk_models[0] for m in chunk_models)
