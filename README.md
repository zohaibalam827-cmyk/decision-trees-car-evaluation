# Decision Trees from Scratch: ID3, C4.5 and CART

**Author:** Zohaib Alam
**Dataset:** UCI Car Evaluation (1,728 cars, 6 categorical features, 4 classes)

## Overview
TODO: write 2-3 sentences in your own words: what this project does and why.

## The three algorithms

| | ID3 | C4.5 | CART |
|---|---|---|---|
| Split measure | Information gain | Gain ratio | Gini impurity |
| Feature types | Categorical only | Categorical and numeric | Categorical and numeric |
| Split shape | One branch per value | One branch per value (categories), two branches for numbers | Always two branches |
| Numeric cut | Not supported | At a real data value (x <= t) | At the midpoint between values |
| Pruning | None | Pessimistic error pruning (confidence 0.25) | Cost-complexity pruning (ccp_alpha) |

TODO: explain in your own words why C4.5 divides by split info, and what pruning is for.

## Project structure

    dtrees/
        base.py     entropy, Gini, Node, shared fit/predict/export_text
        id3.py      ID3DecisionTree
        c45.py      C45DecisionTree
        cart.py     CARTDecisionTree
        utils.py    train_test_split, accuracy_score
    data/           car.data, play_tennis.csv
    examples/       explore_car.py, evaluate_id3.py, evaluate_c45.py, evaluate_cart.py
    tests/          test_trees.py

## How to run

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    python examples/explore_car.py     # look at the data
    python examples/evaluate_cart.py   # compare all algorithms on unseen cars
    python -m pytest -q                # run the tests

## Usage

    import pandas as pd
    from dtrees import ID3DecisionTree

    df = pd.read_csv("data/play_tennis.csv")
    X, y = df.drop(columns="PlayTennis"), df["PlayTennis"]
    tree = ID3DecisionTree().fit(X, y)
    print(tree.export_text())
    print(tree.predict(X.head()))

## Results

Always guessing the biggest class ("unacc") scores about 70% on this data, so that is the baseline.
Each row is the mean over 10 random 70/30 train/test splits.

TODO: paste the table printed by `python examples/evaluate_cart.py` here.

## Discussion

TODO: answer in your own words, using your numbers:
1. Which algorithm did best on unseen cars, and how many leaves did it need?
2. How big was the gap between training and test accuracy for the unpruned trees? What does that gap mean?
3. What did pruning change? Did the simpler tree lose accuracy?

## Limitations
- Classification only (no regression trees).
- Missing values are not supported; fit() raises an error.
- The car data has no numeric features, so numeric splits are tested only on small toy data.

## References
- Quinlan, J.R. (1986). Induction of Decision Trees. Machine Learning 1(1).
- Quinlan, J.R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
- Breiman, Friedman, Olshen and Stone (1984). Classification and Regression Trees. Wadsworth.
