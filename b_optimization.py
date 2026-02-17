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
    sigma_max_real = max((p * r) / (2 * th), (p * r) / ts)
    return sigma_max - sigma_max_real

def f(r, l, ts, th):
    if g_volume(r, l, ts, th) < 0 or g_sigma(r, l, ts, th) < 0:
        return -1e20
    v1 = (4/3) * np.pi * ((r + th)**3 - r**3)
    v2 = np.pi * l * ((r + ts)**2 - r**2)
    v_rez = v1 + v2
    mass = v_rez * rho
    return -mass

def mass_only(r, l, ts, th):
    v1 = (4/3) * np.pi * ((r + th)**3 - r**3)
    v2 = np.pi * l * ((r + ts)**2 - r**2)
    v_rez = v1 + v2
    mass = v_rez * rho
    return -mass

def generate_valid_initial_points(n=5):
    points = []
    while len(points) < n:
        x = {
            "r": np.random.uniform(0.2, 1.0),
            "l": np.random.uniform(0.2, 1.8),
            "ts": np.random.uniform(0.01, 0.15),
            "th": np.random.uniform(0.01, 0.1),
        }
        if g_volume(**x) >= 0 and g_sigma(**x) >= 0:
            points.append(x)
    return points

def gp():
    optimizer = BayesianOptimization(f,ogranichenie, random_state=50, verbose=0)
    init_points = generate_valid_initial_points()
    for x in init_points:
        if g_volume(**x) >= 0 and g_sigma(**x) >= 0:
            optimizer.register(params=x, target=mass_only(**x))
        for _ in range(30):
            next_point = optimizer.suggest()
            if g_volume(**next_point) >= 0 and g_sigma(**next_point) >= 0:
                optimizer.register(
                    params=next_point,
                    target=mass_only(**next_point)
                )
    return optimizer

def get_parameters(rez):
    r = rez.max["params"]["r"]
    l = rez.max["params"]["l"]
    ts = rez.max["params"]["ts"]
    th = rez.max["params"]["th"]
    return r, l, ts, th


def proverka(r, l, ts, th):
    v = (4/3) * np.pi * r**3 + np.pi * r**2 * l
    sigma_max_real = max((p * r) / (2 * th), (p * r) / ts)
    check = {
        "v": float(v),
        "v_ok": bool(g_volume(r, l, ts, th) > 0),
        "sigma_max_real": float(sigma_max_real),
        "sigma_max_real_ok": bool(g_sigma(r, l, ts, th) > 0),
    }
    return check

if __name__ == "__main__":

    rez = gp()

    r, l, ts, th = get_parameters(rez)

    print("Оптимальные параметры для первого метолда:")
    print(f"r  = {r:.4f}")
    print(f"l  = {l:.4f}")
    print(f"ts = {ts:.4f}")
    print(f"th = {th:.4f}")

    check = proverka(r, l, ts, th)
    print("\nПроверка ограничений для первого метода:")
    print(check)
    print("\nМинимальная масса для первого метода:")
    print(-rez.max["target"])
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.constraint import ConstraintModel
import warnings

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

    print("Оптимальные параметры для второго метода:")
    print(f"r  = {r:.4f}")
    print(f"l  = {l:.4f}")
    print(f"ts = {ts:.4f}")
    print(f"th = {th:.4f}")

    print("\nПроверка ограничений для второго метода:")
    print(proverka(r, l, ts, th))

    print("\nМинимальная масса для второго метода:")
    print(-rez.max["target"])
