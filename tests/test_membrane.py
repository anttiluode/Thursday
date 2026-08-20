import numpy as np

from breathing_aperture import BreathingMembrane, RECEIVERS, project_capped_simplex, receiver_loss


def test_budget_projection_is_exact_and_bounded():
    x = project_capped_simplex(np.array([2.0, -1.0, 0.3, 0.8]), budget=1.7)
    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)
    assert abs(float(x.sum()) - 1.7) < 1e-5


def test_receiver_swap_changes_spectral_allocation():
    m = BreathingMembrane(size=64, bands=10, budget=3.0)
    m.set_receiver("coarse")
    for _ in range(6):
        m.adapt_spectrum()
    coarse = m.gains.copy()

    m.set_receiver("texture")
    for _ in range(10):
        m.adapt_spectrum()
    texture = m.gains.copy()

    assert np.linalg.norm(coarse - texture) > 0.05
    assert abs(float(texture.sum()) - m.budget) < 1e-4


def test_disturbance_opens_then_repairs_aperture():
    m = BreathingMembrane(size=64, bands=10, budget=4.0)
    m.set_receiver("edges")
    # Let the spectral allocation settle, then align HOLD to that membrane.
    for _ in range(100):
        m.adapt_spectrum()
        if m.spectral_saturated:
            break
    assert m.spectral_saturated
    m.reset_belief()
    before = receiver_loss(m.world, m.belief, m.receiver)
    m.disturb("top-right")
    disturbed = receiver_loss(m.world, m.belief, m.receiver)
    assert disturbed > before

    means = []
    losses = []
    for _ in range(35):
        s = m.step()
        means.append(s.aperture_mean)
        losses.append(s.receiver_loss)

    assert max(means[:10]) > 0.005
    assert max(means) < 0.20  # local disturbance should not open half the image
    assert losses[-1] < disturbed
    # Once the receiver-visible contradiction is repaired, permeability recedes.
    assert means[-1] < max(means)


def test_all_receivers_run():
    m = BreathingMembrane(size=48, bands=10, budget=3.0)
    for receiver in RECEIVERS:
        m.set_receiver(receiver)
        s = m.step()
        assert np.isfinite(s.receiver_loss)
        assert abs(s.budget_used - m.budget) < 1e-4
