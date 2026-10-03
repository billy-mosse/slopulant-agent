"""Compare two email subject lines: two-proportion z-test on open rates."""
import math


def z_test(opens_a, sent_a, opens_b, sent_b):
    pa, pb = opens_a / sent_a, opens_b / sent_b
    p = (opens_a + opens_b) / (sent_a + sent_b)
    se = math.sqrt(p * (1 - p) * (1 / sent_a + 1 / sent_b))
    z = (pa - pb) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return {"lift": pb / pa - 1, "z": z, "p_value": pval}
