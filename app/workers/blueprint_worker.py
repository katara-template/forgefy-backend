"""Blueprint worker — aggregates requirements into a final blueprint JSON.

Implemented in Step 8.
"""
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.blueprint_worker.generate_blueprint")
def generate_blueprint(session_id: str) -> None:
    """Aggregate requirements for a session and write the blueprint row."""
    raise NotImplementedError("Blueprint worker — see Step 8")
