import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.constraint import ConstraintModel
import warnings

warnings.filterwarnings("ignore")

p = 4e6
v_min = 0.4
rho = 7800
sigma_max = 9.5e6

ogranichenie = {
    "r": (0.2, 1.0),
    "l": (0.2, 1.8),
    "ts": (0.01, 0.15),
    "th": (0.01, 0.1),
}

def g_volume(r, l, ts, th):
    v = (4/3) * np.pi * r**3 + np.pi * r**2 * l
    return v - v_min

def g_sigma(r, l, ts, th):
    sigma_real = max((p * r) / (2 * th), (p * r) / ts)
    return sigma_max - sigma_real

def mass_only(r, l, ts, th):
    v1 = (4/3) * np.pi * ((r + th)**3 - r**3)
    v2 = np.pi * l * ((r + ts)**2 - r**2)
    mass = (v1 + v2) * rho
    return -mass

def gp():

    constraint = ConstraintModel(
        fun=lambda r, l, ts, th: [
            g_volume(r, l, ts, th),
            g_sigma(r, l, ts, th)
        ],
        lb=[0, 0],
        ub=[np.inf, np.inf]
    )

    optimizer = BayesianOptimization(
        mass_only,
        ogranichenie,
        constraint=constraint,
        random_state=50,
        verbose=0,
        allow_duplicate_points=True
    )

    registered = 0
    while registered < 15:
        sample_array = optimizer.space.random_sample()
        sample = dict(zip(optimizer.space.keys, sample_array))

        g1 = g_volume(**sample)
        g2 = g_sigma(**sample)

        if g1 >= 0 and g2 >= 0:
            optimizer.register(
                params=sample,
                target=mass_only(**sample),
                constraint_value=[g1, g2]
            )
            registered += 1

    optimizer.maximize(init_points=0, n_iter=10)

    return optimizer

def get_parameters(rez):
    return rez.max["params"]

def proverka(r, l, ts, th):
    v = (4/3) * np.pi * r**3 + np.pi * r**2 * l
    sigma_real = max((p * r) / (2 * th), (p * r) / ts)
    return {
        "v": float(v),
        "v_ok": bool(g_volume(r, l, ts, th)>0),
        "sigma_max_real": float(sigma_real),
        "sigma_max_real_ok": bool(g_sigma(r, l, ts, th))
    }

if __name__ == "__main__":

    rez = gp()
    params = get_parameters(rez)

    r, l, ts, th = params["r"], params["l"], params["ts"], params["th"]

    print("Оптимальные параметры:")
    print(f"r  = {r:.4f}")
    print(f"l  = {l:.4f}")
    print(f"ts = {ts:.4f}")
    print(f"th = {th:.4f}")

    print("\nПроверка ограничений:")
    print(proverka(r, l, ts, th))

    print("\nМинимальная масса:")
    print(-rez.max["target"])
