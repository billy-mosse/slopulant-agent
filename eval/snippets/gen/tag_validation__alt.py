from collections import defaultdict
import re

CONTROLLED_SET = {"bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow", "decor", "kitchen"}

TERM_MAPPINGS = {
    "linens": "linen",
    "linen-blend": "linen",
    "towels": "towel",
    "pillows": "pillow",
    "bed": "bedding",
    "home-decor": "decor",
}


def clean_term(raw):
    normalized = re.sub(r"\s+", "-", raw.strip().casefold())
    return TERM_MAPPINGS.get(normalized, normalized)


def filter_tags(source_data):
    results = defaultdict(set)
    for item_id, label in source_data:
        candidate = clean_term(label)
        if candidate in CONTROLLED_SET:
            results[item_id].add(candidate)
    return {sku: sorted(group) for sku, group in results.items()}


if __name__ == "__main__":
    test_input = [
        ("BED-001", "Linen"),
        ("BED-001", "Bed"),
        ("BED-001", "sale!!"),
        ("BTH-010", "Towels"),
    ]
    print(filter_tags(test_input))
