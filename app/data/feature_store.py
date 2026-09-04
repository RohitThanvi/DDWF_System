"""
Feature-store manifest (Delta Lake table) registering each written Zarr
shard with time-partitioning, for point-in-time correctness — i.e. so a
model trained "as of" a given date can only ever see data that was truly
available at that date (no leakage from reanalysis revisions)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ShardManifestEntry:
    variable: str
    shard_uri: str
    valid_time: datetime
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "ERA5"          # ERA5 | CMIP6 | MERRA-2
    qc_passed: bool = True


class FeatureStoreManifest:
    """Placeholder interface over a Delta Lake table
    (`delta-rs` / `deltalake` python package at write time). Kept as a thin
    class so the ingestion DAG and the serving path share one contract."""

    def __init__(self, table_uri: str):
        self.table_uri = table_uri

    def register(self, entry: ShardManifestEntry) -> None:
        raise NotImplementedError("Wire to deltalake.write_deltalake in the ingestion DAG.")

    def query_as_of(self, variable: str, as_of: datetime):
        raise NotImplementedError("Point-in-time query against the Delta table.")
