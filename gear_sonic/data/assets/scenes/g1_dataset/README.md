# G1 dataset M0 scenes

This package contains two small, deterministic USD fixtures for exercising the
G1 dataset-generation path before larger Isaac Sim asset packs are introduced:

- `household_room.usda` provides two furnished room zones joined by a two-metre
  doorway.
- `factory_aisle.usda` provides a rack-lined aisle with a route around a pallet.

Both scenes are self-contained USDA, Z-up, one metre per unit, and use `/World`
as the default prim. The support surface is a collision-backed PhysX `Plane` at
`z=0`; the remaining solids are axis-aligned `Cube` prims with
`PhysicsCollisionAPI`. The canonical clear routes, integrity hashes, split
groups, provenance, and redistribution metadata are in `manifest.json`.

All geometry was authored in this repository from USD primitives. There are no
third-party meshes, materials, or textures. These are functional smoke-test
layouts, not photorealistic production environments.

Run the dependency-free gate from the repository root:

```bash
python scripts/research/check_g1_dataset_scenes.py
```
