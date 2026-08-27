import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from polreps import actcache, sweep
from tests.conftest import CONDITIONS, make_planted_data


def test_planted_signal_peaks_at_planted_layer(planted):
    # loads through the real cache format on purpose: this is the exact path
    # the real sweep will take over ticket 03's output
    acts, _ = actcache.load_cache(
        planted["cache_dir"], expect_prompt_ids=planted["prompt_ids"]
    )
    fold_accs = sweep.layer_accuracies(acts, planted["labels"], planted["groups"])
    accs = fold_accs.mean(axis=1)
    chance = sweep.chance_level(planted["labels"])

    assert accs.argmax() == planted["signal_layer"]
    assert accs[planted["signal_layer"]] > 0.9
    off_layers = np.delete(accs, planted["signal_layer"])
    assert (off_layers < chance + 0.15).all()


def test_shuffled_labels_fall_to_chance_at_every_layer():
    acts, labels, groups, _ = make_planted_data()
    shuffled = sweep.shuffle_labels(labels, seed=1)
    accs = sweep.layer_accuracies(acts, shuffled, groups).mean(axis=1)
    chance = sweep.chance_level(labels)

    # the planted layer must lose its signal too, or the shuffle isn't a null
    assert np.abs(accs - chance).max() < 0.15


def test_shuffle_is_a_permutation_and_seeded():
    labels = np.array(["a", "a", "b", "c"])
    s1 = sweep.shuffle_labels(labels, seed=0)
    s2 = sweep.shuffle_labels(labels, seed=0)

    assert sorted(s1) == sorted(labels)
    assert (s1 == s2).all()
    assert (labels == np.array(["a", "a", "b", "c"])).all()


def test_binary_variant_refuses_conditions_absent_from_data():
    acts, labels, groups, _ = make_planted_data(n_groups=6)
    with pytest.raises(ValueError, match="vocabulary"):
        sweep.sweep_variant(
            acts, labels, groups, n_splits=2, seed=0,
            conditions=("democrat", "libertarian"),
        )


def test_chance_level_is_majority_class_share():
    labels = np.array(["a", "a", "a", "b", "c", "c"])
    assert sweep.chance_level(labels) == 0.5


def test_group_cv_blocks_base_question_leakage():
    # Each (question, condition) carries an idiosyncratic high-dimensional
    # quirk shared by its two rows, and there is no condition signal that
    # generalizes across questions. A row-level split puts one twin in train
    # and one in test, so a probe can score far above chance by memorizing
    # quirks; a base-question split cannot. The sweep must report the honest
    # number.
    rng = np.random.default_rng(2)
    n_groups, d = 12, 200
    rows, labels, groups = [], [], []
    for g in range(n_groups):
        base_q = 2.0 * rng.normal(size=d)
        for cond in CONDITIONS:
            quirk = rng.normal(size=d)
            for _ in range(2):
                rows.append(base_q + quirk + 0.1 * rng.normal(size=d))
                labels.append(cond)
                groups.append(f"q{g}")
    acts = np.array(rows)[None]  # a single layer is enough here
    labels, groups = np.array(labels), np.array(groups)
    chance = sweep.chance_level(labels)

    grouped_acc = sweep.layer_accuracies(acts, labels, groups).mean()

    # deliberately leaky control: identical probe, rows split without groups
    row_split = KFold(n_splits=5, shuffle=True, random_state=0)
    leaky_acc = cross_val_score(sweep.make_probe(), acts[0], labels, cv=row_split).mean()

    assert grouped_acc < chance + 0.15
    assert leaky_acc > chance + 0.3


def test_standardization_is_fold_local():
    # Groups g6/g7 get extra label-uncorrelated variance on the signal
    # dimension, so a scaler that saw the test fold would rescale that
    # dimension and (verified below) flip predictions. The sweep must match a
    # manual loop whose scaler only ever sees training rows, and must not
    # match the leaky variant. Constants chosen (by a seed scan) so the two
    # variants measurably disagree under the pinned sklearn.
    rng = np.random.default_rng(0)
    rows, labels, groups = [], [], []
    for g in range(8):
        for sign, cond in ((-1.0, "democrat"), (1.0, "republican")):
            for _ in range(4):
                x = rng.normal(size=10)
                x[0] += 2.0 * sign
                if g >= 6:
                    x[0] += 5.0 * rng.normal()
                rows.append(x)
                labels.append(cond)
                groups.append(f"g{g}")
    acts = np.array(rows)[None]
    labels, groups = np.array(labels), np.array(groups)

    def manual_accs(scaler_sees_test):
        accs = []
        for tr, te in GroupKFold(n_splits=4).split(acts[0], labels, groups):
            fit_rows = acts[0] if scaler_sees_test else acts[0][tr]
            scaler = StandardScaler().fit(fit_rows)
            clf = LogisticRegression(C=sweep.PROBE_C, max_iter=sweep.PROBE_MAX_ITER)
            clf.fit(scaler.transform(acts[0][tr]), labels[tr])
            accs.append(clf.score(scaler.transform(acts[0][te]), labels[te]))
        return np.array(accs)

    sweep_accs = sweep.layer_accuracies(acts, labels, groups, n_splits=4)[0]
    fold_local = manual_accs(scaler_sees_test=False)
    leaky = manual_accs(scaler_sees_test=True)

    np.testing.assert_allclose(sweep_accs, fold_local)
    # the equality above is only evidence if the leaky route actually differs
    assert np.abs(fold_local - leaky).max() > 0.05
