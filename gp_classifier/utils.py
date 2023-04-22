from sklearn import metrics


def eval_metrics(actual, pred):
    acc = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    recall = metrics.recall_score(actual, pred, pos_label=1)
    precision = metrics.precision_score(actual, pred, pos_label=1)

    return {"acc": acc, "f1": f1, "recall": recall, "precision": precision}
