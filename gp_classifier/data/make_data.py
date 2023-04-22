import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons


def make_data(n_samples=1500, shuffle=True, noise=None, random_state=None, plot=False):
    X, y = make_moons(
        n_samples=n_samples, shuffle=shuffle, noise=noise, random_state=random_state
    )
    y = y.reshape(-1, 1)

    if plot:
        import matplotlib.pyplot as plt

        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k")
        plt.show()

    return X, y


def save_data(X, y, filename):
    # Assert that the number of rows in X and y is the same.
    assert X.shape[0] == y.shape[0]

    # Create a dataframe in Pandas. Each column in X is a feature and each column in y is a label. Concatenate them together. Each column of X is called 'feature_0', 'feature_1', 'label_0', 'label_1'. Each column of y is a label. So, we need to add a prefix 'label_' to each column of y.
    df = pd.concat(
        [
            pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]),
            pd.DataFrame(y, columns=[f"label_{i}" for i in range(y.shape[1])]),
        ],
        axis=1,
    )

    # Save the dataframe to a csv file.
    df.to_csv(filename, index=False, sep=",")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=23423)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()

    X, y = make_data(
        n_samples=args.n_samples,
        shuffle=args.shuffle,
        noise=args.noise,
        random_state=args.random_state,
        plot=args.plot,
    )
    save_data(X, y, args.filename)


if __name__ == "__main__":
    main()
