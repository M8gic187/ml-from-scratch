"""Demo: MultinomialNB and BernoulliNB on synthetic text-like datasets."""

import numpy as np

from ml_scratch.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from ml_scratch.utils.metrics import accuracy


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def train_test_split(X, y, test_size=0.25, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


# ---------------------------------------------------------------------------
# 1. MultinomialNB — bag-of-words simulation (4 topics)
# ---------------------------------------------------------------------------

def demo_multinomial():
    print("=" * 60)
    print("MultinomialNB — 4-topic bag-of-words simulation")
    print("=" * 60)

    rng = np.random.default_rng(0)
    vocab_size = 20
    lams = np.zeros((4, vocab_size))
    for i in range(4):
        lams[i, i * 5:(i + 1) * 5] = 8.0  # each topic dominates 5 words

    X = np.vstack([rng.poisson(lam=lams[i], size=(150, vocab_size)) for i in range(4)])
    y = np.repeat([0, 1, 2, 3], 150)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)

    print(f"  Train accuracy : {accuracy(y_train, clf.predict(X_train)):.4f}")
    print(f"  Test  accuracy : {accuracy(y_test,  clf.predict(X_test)):.4f}")

    print("\n  Top-2 vocabulary indices per topic (highest P(word|topic)):")
    for i, cls in enumerate(clf.classes_):
        top2 = np.argsort(clf.feature_log_prob_[i])[-2:][::-1]
        probs = np.exp(clf.feature_log_prob_[i, top2])
        print(f"    Topic {cls}: words {top2.tolist()} with P={probs.round(4).tolist()}")

    print("\n  Posterior probabilities for first 5 test samples:")
    proba = clf.predict_proba(X_test[:5])
    header = "  ".join(f"P(y={c})" for c in clf.classes_)
    print(f"  {'Sample':>6}  {header}")
    for i, row in enumerate(proba):
        print(f"  {i:>6}  " + "  ".join(f"{p:7.4f}" for p in row))


# ---------------------------------------------------------------------------
# 2. BernoulliNB — binary keyword presence (spam-like scenario)
# ---------------------------------------------------------------------------

def demo_bernoulli():
    print()
    print("=" * 60)
    print("BernoulliNB — binary keyword presence (spam detection)")
    print("=" * 60)

    rng = np.random.default_rng(1)
    n_features = 16

    # Ham: first 8 keywords unlikely, last 8 keywords more likely
    X_ham  = (rng.uniform(size=(200, n_features)) < np.r_[np.full(8, 0.1), np.full(8, 0.6)]).astype(float)
    # Spam: first 8 keywords very likely (typical spam words), last 8 unlikely
    X_spam = (rng.uniform(size=(200, n_features)) < np.r_[np.full(8, 0.8), np.full(8, 0.1)]).astype(float)

    X = np.vstack([X_ham, X_spam])
    y = np.array([0] * 200 + [1] * 200)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = BernoulliNB(alpha=1.0, binarize=None)
    clf.fit(X_train, y_train)

    print(f"  Train accuracy : {accuracy(y_train, clf.predict(X_train)):.4f}")
    print(f"  Test  accuracy : {accuracy(y_test,  clf.predict(X_test)):.4f}")

    print("\n  Learned P(feature=1 | class) for first 8 features:")
    print(f"  {'Feature':>7}  P(f=1|ham)  P(f=1|spam)")
    for j in range(8):
        p_ham  = np.exp(clf.feature_log_prob_[0, j])
        p_spam = np.exp(clf.feature_log_prob_[1, j])
        print(f"  {j:>7}  {p_ham:10.4f}  {p_spam:11.4f}")

    print("\n  Posterior probabilities for first 5 test samples:")
    proba = clf.predict_proba(X_test[:5])
    print(f"  {'Sample':>6}  P(ham)    P(spam)   Predicted")
    for i, (row, pred) in enumerate(zip(proba, clf.predict(X_test[:5]))):
        label = "ham" if pred == 0 else "spam"
        print(f"  {i:>6}  {row[0]:8.4f}  {row[1]:8.4f}  {label}")


# ---------------------------------------------------------------------------
# 3. Side-by-side comparison on continuous data (all three classifiers)
# ---------------------------------------------------------------------------

def demo_comparison():
    print()
    print("=" * 60)
    print("Side-by-side: GaussianNB vs MultinomialNB vs BernoulliNB")
    print("on Gaussian-sampled data (binarized for Bernoulli)")
    print("=" * 60)

    rng = np.random.default_rng(5)
    centres = np.array([[4, 0], [-4, 0], [0, 4], [0, -4]], dtype=float)
    X = np.vstack([rng.normal(c, 1.0, (120, 2)) for c in centres])
    y = np.repeat([0, 1, 2, 3], 120)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Shift to non-negative for MultinomialNB
    X_shift = X - X.min()
    X_train_s, X_test_s = X_shift[: len(X_train)], X_shift[len(X_train):]

    results = [
        ("GaussianNB",     GaussianNB().fit(X_train,   y_train), X_test),
        ("MultinomialNB",  MultinomialNB().fit(X_train_s, y_train), X_test_s),
        ("BernoulliNB",    BernoulliNB(binarize=0.0).fit(X_train, y_train), X_test),
    ]

    print(f"  {'Classifier':<15}  Train Acc  Test Acc")
    for name, clf, Xt in results:
        tr_acc = accuracy(y_train, clf.predict(X_train_s if name == "MultinomialNB" else X_train))
        te_acc = accuracy(y_test, clf.predict(Xt))
        print(f"  {name:<15}  {tr_acc:.4f}     {te_acc:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_multinomial()
    demo_bernoulli()
    demo_comparison()
