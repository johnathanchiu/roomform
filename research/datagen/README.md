# datagen (port pending)

The synthetic building generator migrates here from the internal repo:
procedural layouts → furnish → simulated scan → visibility carve →
`ScanInput`/`EvidenceGrid` npz + patch-graph GT (node + 13-edge
labels). Determinism contract: all sampling seeded via crc32 of stable
names — a dataset is fully regenerable from (repo commit, seed range).
