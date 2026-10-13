import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtrees import C45DecisionTree, CARTDecisionTree, ID3DecisionTree
from dtrees.base import entropy, gini
from dtrees.utils import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"
CAR_COLS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]


@pytest.fixture
def tennis():
    df = pd.read_csv(DATA / "play_tennis.csv")
    return df.drop(columns="PlayTennis"), df["PlayTennis"].to_numpy()


@pytest.fixture
def cars():
    path = DATA / "car.data"
    if not path.exists():
        pytest.skip("data/car.data not found")
    df = pd.read_csv(path, names=CAR_COLS)
    return df.drop(columns="class"), df["class"].to_numpy()


@pytest.fixture
def noisy():
    rng = np.random.default_rng(1)
    x = rng.random(200)
    labels = np.where(x + rng.normal(0, 0.3, 200) > 0.5, "a", "b")
    return pd.DataFrame({"x": x}), labels


# ---------------------------------------------------------------- the maths
def test_entropy_and_gini_known_values():
    labels = ["Yes"] * 9 + ["No"] * 5
    assert entropy(labels) == pytest.approx(0.940, abs=1e-3)
    assert gini(labels) == pytest.approx(1 - (9 / 14) ** 2 - (5 / 14) ** 2)
    assert entropy(["a", "a", "a"]) == 0 and gini(["a", "a", "a"]) == 0


def test_id3_information_gains_match_textbook(tennis):
    X, y = tennis
    gains = {f: ID3DecisionTree.information_gain(X[f].to_numpy(), y) for f in X.columns}
    assert gains["Outlook"] == pytest.approx(0.247, abs=1e-3)
    assert gains["Humidity"] == pytest.approx(0.152, abs=1e-3)
    assert gains["Windy"] == pytest.approx(0.048, abs=1e-3)
    assert gains["Temperature"] == pytest.approx(0.029, abs=1e-3)


# ---------------------------------------------------------------- Play Tennis
@pytest.mark.parametrize("make", [
    lambda: ID3DecisionTree(),
    lambda: C45DecisionTree(prune=False),
    lambda: CARTDecisionTree(),
])
def test_every_algorithm_fits_play_tennis_perfectly(make, tennis):
    X, y = tennis
    assert make().fit(X, y).score(X, y) == 1.0


def test_id3_and_c45_start_with_outlook(tennis):
    X, y = tennis
    assert ID3DecisionTree().fit(X, y).root_.feature == "Outlook"
    assert C45DecisionTree(prune=False).fit(X, y).root_.feature == "Outlook"


# ---------------------------------------------------------------- behaviour
def test_id3_rejects_numeric_features():
    with pytest.raises(ValueError):
        ID3DecisionTree().fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["a", "b", "a"])


@pytest.mark.parametrize("Model", [ID3DecisionTree, C45DecisionTree, CARTDecisionTree])
def test_missing_values_are_rejected(Model):
    with pytest.raises(ValueError):
        Model().fit(pd.DataFrame({"x": ["p", None]}), ["a", "b"])


def test_id3_unseen_value_falls_back_to_majority_class(tennis):
    X, y = tennis
    tree = ID3DecisionTree().fit(X, y)
    new_day = pd.DataFrame([{"Outlook": "Foggy", "Temperature": "Hot", "Humidity": "High", "Windy": "Weak"}])
    assert tree.predict(new_day)[0] == "Yes"  # 9 of the 14 training days were "Yes"


def test_c45_cuts_numbers_at_a_real_data_value():
    X = pd.DataFrame({"x": [1, 2, 3, 10, 11, 12]})
    tree = C45DecisionTree(min_samples_leaf=1, prune=False).fit(X, ["a", "a", "a", "b", "b", "b"])
    assert tree.root_.threshold == 3
    assert list(tree.predict(pd.DataFrame({"x": [0.5, 4, 9, 20]}))) == ["a", "b", "b", "b"]


def test_cart_cuts_numbers_at_the_midpoint():
    X = pd.DataFrame({"x": [1, 2, 3, 10, 11, 12]})
    tree = CARTDecisionTree().fit(X, ["a", "a", "a", "b", "b", "b"])
    assert tree.root_.threshold == 6.5


def test_cart_groups_category_values():
    X = pd.DataFrame({"c": ["a", "b", "c", "d"] * 10})
    labels = np.array(["x", "x", "y", "y"] * 10)
    tree = CARTDecisionTree().fit(X, labels)
    assert tree.root_.threshold == frozenset({"c", "d"})
    assert tree.score(X, labels) == 1.0 and tree.depth() == 1


# ---------------------------------------------------------------- pruning
def test_c45_pruning_makes_the_tree_smaller(noisy):
    X, y = noisy
    full = C45DecisionTree(prune=False).fit(X, y)
    pruned = C45DecisionTree(prune=True).fit(X, y)
    assert pruned.n_leaves() < full.n_leaves()


def test_cart_pruning_shrinks_as_alpha_grows(noisy):
    X, y = noisy
    sizes = [CARTDecisionTree(ccp_alpha=a).fit(X, y).n_leaves() for a in (0.0, 0.005, 0.02)]
    assert sizes[0] > sizes[1] > sizes[2]


# ---------------------------------------------------------------- car data
def test_unpruned_id3_and_cart_memorise_the_car_training_data(cars):
    X, y = cars  # every car is a different combination, so a full tree can always separate them
    assert ID3DecisionTree().fit(X, y).score(X, y) == 1.0
    assert CARTDecisionTree().fit(X, y).score(X, y) == 1.0


@pytest.mark.parametrize("make", [
    lambda: ID3DecisionTree(),
    lambda: C45DecisionTree(),
    lambda: CARTDecisionTree(ccp_alpha=0.001),
])
def test_every_algorithm_beats_the_always_unacc_baseline(make, cars):
    X, y = cars
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42)
    baseline = np.mean(y_test == "unacc")
    assert make().fit(X_train, y_train).score(X_test, y_test) > baseline
