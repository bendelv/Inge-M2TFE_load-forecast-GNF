import numpy as np
from metrics import *
from sklearn.preprocessing import StandardScaler


def main():
    train_load = np.load("../../02_datasets/Sets/train_load.npy")
    load = np.load("../../02_datasets/Sets/test_load.npy")
    DIR_MODEL = "."

    test_std_target = StandardScaler()
    load = test_std_target.fit_transform(load)

    train_std_target = StandardScaler()
    train_std_target.fit_transform(train_load)

    n_days = 200
    gauss_scens = np.random.normal(size=(n_days, 10, 192))

    n_plotdays = 10
    # plot some of generated days and correlation plots
    target = test_std_target.inverse_transform(load[:n_plotdays])
    scens = train_std_target.inverse_transform(gauss_scens[:n_plotdays].reshape(-1, 192))
    scens = scens.reshape(target.shape[0], n_plotdays, 192)
    plot_days_scenarios(DIR_MODEL, target, scens)
    plot_corr_scens(directory=DIR_MODEL, days_scens=gauss_scens)

    print("CRPS score")
    CRPS = []
    for day_scens, target in zip(gauss_scens, load[:n_days]):
        day_CRPS = instant_CRPS(day_scens, target)
        CRPS.append(day_CRPS)

    CRPS = np.array(CRPS)
    mean_CRPS = CRPS.mean(axis=0)
    std_CRPS = CRPS.std(axis=0)
    plot_day_CRPS(mean_CRPS, std_CRPS, DIR_MODEL)

    with open('{}/array_crps.npy'.format(DIR_MODEL), 'wb') as f:
        np.save(f, CRPS)
    f.close()

    print("Energy score")
    es = energy_score(gauss_scens, np.array(load)[:n_days])
    es_mean, es_std = np.array(es).mean(), np.array(es).std()
    print(es_mean, es_std, "\n")
    with open('{}/array_es.npy'.format(DIR_MODEL), 'wb') as f:
        np.save(f, es)
    f.close()

    print("Variogram score")
    vs = variogram_score(gauss_scens, np.array(load)[:n_days], 1)
    vs_mean, vs_std = np.array(vs).mean(), np.array(vs).std()
    print(vs_mean, vs_std)
    with open('{}/array_vs.npy'.format(DIR_MODEL), 'wb') as f:
        np.save(f, vs)
    f.close()

    print("Variogram zone score")
    vsz = variogram_zone_score(gauss_scens, np.array(load)[:n_days], 1)
    print(np.array(vsz))
    vsz_mean, vsz_std = np.array(vsz).mean(axis=0), np.array(vsz).std(axis=0)
    print(vsz_mean, vsz_std)
    with open('{}/array_vsz.npy'.format(DIR_MODEL), 'wb') as f:
        np.save(f, vsz)
    f.close()


    return 0


if __name__ == "__main__":
    main()