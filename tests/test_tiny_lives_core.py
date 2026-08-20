from tiny_lives_core import AgentConfig, BeliefState, TinyCreature, TinyWorld, WorldConfig, run_episode


def small_world(seed=1):
    wc = WorldConfig(
        size=12,
        food_count=6,
        hazard_count=2,
        wall_fraction=0.02,
        hazard_move_probability=0.0,
        food_respawn_probability=0.0,
        max_food=6,
    )
    return TinyWorld(wc, seed=seed)


def test_hold_preserves_unsensed_content_and_age_increases():
    ac = AgentConfig(memory_horizon=10.0)
    belief = BeliefState(8, ac)
    belief.food[3, 4] = 1.0
    belief.known[3, 4] = True
    belief.age[3, 4] = 0.0
    belief.tick()
    assert belief.food[3, 4] == 1.0
    assert belief.age[3, 4] == 1.0


def test_policy_information_budgets_are_distinct():
    world = small_world()
    ac = AgentConfig(info_budget=18, local_radius=1)
    pos = (6, 6)
    reflex = TinyCreature("reflex", world, ac, seed=2, position=pos)
    full = TinyCreature("full", world, ac, seed=2, position=pos)
    metabolism = TinyCreature("metabolism", world, ac, seed=2, position=pos)

    assert len(reflex.choose_sensing_cells()) == 9
    assert len(metabolism.choose_sensing_cells()) == 18
    assert len(full.choose_sensing_cells()) == world.size * world.size


def test_metabolic_sensing_does_not_exceed_budget():
    world = small_world()
    ac = AgentConfig(info_budget=14, local_radius=1)
    creature = TinyCreature("metabolism", world, ac, seed=7, position=(6, 6))
    for _ in range(5):
        creature.belief.tick()
        coords = creature.choose_sensing_cells()
        assert len(coords) <= ac.info_budget
        assert len(coords) == len(set(coords))
        creature.belief.update(world.observe(coords))


def test_episode_is_deterministic_for_same_seed():
    a = run_episode("metabolism", seed=11, max_steps=80)
    b = run_episode("metabolism", seed=11, max_steps=80)
    assert a == b


def test_full_observer_spends_more_information_than_metabolism():
    full = run_episode("full", seed=3, max_steps=40)
    meta = run_episode("metabolism", seed=3, max_steps=40)
    assert full.mean_cells_sensed_per_step > meta.mean_cells_sensed_per_step
    assert full.sensing_energy > meta.sensing_energy
