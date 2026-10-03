import pandas as pd
from sklearn.linear_model import Ridge

TEMPLATE_FEATURES = ["bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow"]
ALIAS_MAP = {
    "linens": "linen",
    "towels": "towel",
    "bed": "bedding",
    "pillows": "pillow",
}


def normalize_label(lbl):
    return ALIAS_MAP.get(lbl.strip().lower(), lbl.strip().lower())


def encode_product(labels):
    idx = {f: i + 1 for i, f in enumerate(TEMPLATE_FEATURES)}
    active = [1 if normalize_label(l) in idx else 0 for l in labels]
    return [1.0] + active  # intercept term first


def train_model(past_transactions):
    df = pd.DataFrame(past_transactions)
    X = np.array([encode_product(t) for t in df["tags"].tolist()])
    y = df["price"].values
    model = Ridge(alpha=1e-6, fit_intercept=False)
    model.fit(X, y)
    return model


def propose_price(model, new_labels):
    x_vec = np.array([encode_product(new_labels)])
    return round(float(model.predict(x_vec)[0]), 2)


if __name__ == "__main__":
    data = [
        {"tags": ["Bed", "Linens", "duvet"], "price": 189.0},
        {"tags": ["bed", "cotton", "duvet"], "price": 129.0},
        {"tags": ["bath", "Towels", "cotton"], "price": 34.0},
        {"tags": ["bed", "linen", "Pillows"], "price": 59.0},
    ]
    estimator = train_model(data)
    print(propose_price(estimator, ["bed", "linen", "duvet"]))
