"""
Order warehouse bin visits for a picker using aisle serpentine traversal.

Input: warehouse.bins (bin_id, aisle, bay, level, side)
Output: list[dict] with bin_id and visit_order (1-indexed)
"""

from typing import List, Dict


def serpentine_bin_order(bins: List[Dict]) -> List[Dict]:
    """Return bins ordered in serpentine aisle traversal."""
    if not bins:
        return []

    # Group bins by aisle, then sort by bay (ascending), side (L before R), level (ascending)
    side_priority = {'L': 0, 'R': 1}
    bins_sorted = sorted(
        bins,
        key=lambda b: (b['aisle'], b['bay'], side_priority.get(b['side'], 2), b['level'])
    )

    # Serpentine: reverse order for odd-numbered aisles (1-indexed)
    result = []
    current_aisle = None
    for i, bin_ in enumerate(bins_sorted):
        if bin_['aisle'] != current_aisle:
            current_aisle = bin_['aisle']
            # Reverse if current aisle is odd (1-indexed)
            if current_aisle % 2 == 1:
                # Collect all bins in this aisle and reverse
                aisle_bins = [bin_]
                j = i + 1
                while j < len(bins_sorted) and bins_sorted[j]['aisle'] == current_aisle:
                    aisle_bins.append(bins_sorted[j])
                    j += 1
                aisle_bins.reverse()
                for b in aisle_bins:
                    result.append({**b, 'visit_order': len(result) + 1})
                # Skip already processed bins
                for _ in range(len(aisle_bins) - 1):
                    next(iter(bins_sorted[i+1:i+len(aisle_bins)]), None)
                continue
        result.append({**bin_, 'visit_order': len(result) + 1})

    return result
