import numpy as np

from splat_ecology_core import (
    EcologyConfig, LearnedSubstrate, MockSplatDecoder, SplatEcology,
    SubstrateConfig,
)


def tiny_substrate(seed=3):
    dec = MockSplatDecoder(size=24, seed=9)
    cfg = SubstrateConfig(width=16, height=10, batch=20, seed=seed)
    return LearnedSubstrate.build(dec, cfg)


def test_latent_radius_matches_vertical_shell_coordinate():
    s = tiny_substrate()
    for y in [0, 3, 9]:
        z = s.latent_at(5.5, y)
        assert np.isclose(np.linalg.norm(z), s.radius_for_y(y), atol=1e-5)


def test_learned_substrate_has_nontrivial_resources():
    s = tiny_substrate()
    assert s.capacity.shape == (10, 16, 3)
    assert np.all(s.capacity >= 0.079)
    assert np.all(s.capacity <= 1.001)
    assert float(np.std(s.capacity)) > 0.02
    assert float(np.std(s.volatility)) > 0.01


def test_creature_never_senses_over_its_budget():
    s = tiny_substrate()
    eco = SplatEcology(s, EcologyConfig(initial_population=8, seed=5))
    for _ in range(40):
        eco.step()
        for c in eco.creatures:
            assert c.last_info <= c.sense_budget
            assert c.last_info <= len(eco.NEIGHBOURS)


def test_feeding_depletes_and_regeneration_recovers():
    s = tiny_substrate()
    cfg = EcologyConfig(initial_population=1, seed=2, curiosity=0.0, regeneration=0.0)
    eco = SplatEcology(s, cfg)
    c = eco.creatures[0]
    x, y = s.index(c.x, c.y)
    before = eco.resources.copy()
    eco._feed(c)
    assert np.any(eco.resources[y, x] < before[y, x])
    depleted = eco.resources.copy()
    eco.config.regeneration = 0.5
    eco.creatures = []
    eco.step()
    assert np.sum(eco.resources - depleted) > 0


def test_deterministic_ecology_replay():
    s1 = tiny_substrate(seed=8)
    s2 = tiny_substrate(seed=8)
    cfg1 = EcologyConfig(initial_population=12, seed=77)
    cfg2 = EcologyConfig(initial_population=12, seed=77)
    a = SplatEcology(s1, cfg1)
    b = SplatEcology(s2, cfg2)
    for _ in range(60):
        sa = a.step(); sb = b.step()
    assert sa.population == sb.population
    assert sa.births == sb.births
    assert sa.deaths == sb.deaths
    assert sa.total_information == sb.total_information
    assert np.allclose(a.resources, b.resources)


def test_information_price_changes_energy_accounting():
    s = tiny_substrate(seed=4)
    low = SplatEcology(s, EcologyConfig(initial_population=6, seed=20, info_cost=0.0))
    high = SplatEcology(s, EcologyConfig(initial_population=6, seed=20, info_cost=0.08))
    low.step(); high.step()
    assert np.mean([c.energy for c in high.creatures]) < np.mean([c.energy for c in low.creatures])
