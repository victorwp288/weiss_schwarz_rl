from pathlib import Path
from weiss_rl.league.registry import SnapshotRegistry

p = Path("runs/_tmp/training/snapshots/registry.json")
r = SnapshotRegistry.load(p)
r.add_snapshot(policy_id="policy_000002", update=2, weights_sha256="aa", path="training/snapshots/policy_000002/weights.pt")
r.add_snapshot(policy_id="policy_000001", update=1, weights_sha256="bb", path="training/snapshots/policy_000001/weights.pt")
r.save(p)

r2 = SnapshotRegistry.load(p)
assert [s.policy_id for s in r2.latest(2)] == ["policy_000001", "policy_000002"]
print("ok")
