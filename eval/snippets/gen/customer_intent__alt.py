import numpy as np

EMB_DIM = 128

def encode_session_behavior(event_log, item_latents):
    viewed = [e for e in event_log if e["action"] == "view"]
    cart_events = sum(1 for e in event_log if e["action"] == "add_to_cart")
    total_time = sum(e.get("duration_s", 0) for e in viewed)
    latent_list = [item_latents[e["item_id"]] for e in viewed if e["item_id"] in item_latents]
    avg_latent = np.mean(latent_list, axis=0) if latent_list else np.zeros(EMB_DIM)
    return np.hstack([np.array([len(viewed), cart_events, np.log1p(total_time)]), avg_latent])


def estimate_conversion_prob(behavior_vec, theta, intercept=-2.5):
    z = np.dot(behavior_vec, theta) + intercept
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    latents = {"SKU-101": rng.standard_normal(EMB_DIM), "SKU-202": rng.standard_normal(EMB_DIM)}
    logs = [
        {"item_id": "SKU-101", "action": "view", "duration_s": 35},
        {"item_id": "SKU-202", "action": "view", "duration_s": 20},
        {"item_id": "SKU-101", "action": "add_to_cart"},
    ]
    coeffs = np.zeros(EMB_DIM + 3)
    coeffs[:3] = [0.25, 1.8, 0.15]
    vec = encode_session_behavior(logs, latents)
    print(round(estimate_conversion_prob(vec, coeffs), 3))
