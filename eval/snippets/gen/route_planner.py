"""
Order delivery stops for a van to minimise driving distance using nearest-neighbour heuristic.

Input:  raw_orders (schema.orders) - order_id, delivery_lat, delivery_lng
Output: ordered_stops (schema.delivery_route) - stop_order, order_id, delivery_lat, delivery_lng, cumulative_distance
"""

import math
from typing import List, Dict, Tuple, Any

EARTH_RADIUS_KM = 6371.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

def compute_route(stops: List[Dict[str, Any]], start_lat: float = None, start_lng: float = None) -> List[Dict[str, Any]]:
    if not stops:
        return []
    route = []
    remaining = stops.copy()
    current_lat = start_lat if start_lat is not None else remaining.pop(0)['delivery_lat']
    current_lng = start_lng if start_lng is not None else remaining.pop(0)['delivery_lng']
    route.append({'stop_order': 1, 'delivery_lat': current_lat, 'delivery_lng': current_lng, 'cumulative_distance': 0.0})
    total_dist = 0.0
    while remaining:
        nearest = min(remaining, key=lambda s: haversine_distance(current_lat, current_lng, s['delivery_lat'], s['delivery_lng']))
        dist = haversine_distance(current_lat, current_lng, nearest['delivery_lat'], nearest['delivery_lng'])
        total_dist += dist
        route.append({
            'stop_order': len(route),
            'order_id': nearest['order_id'],
            'delivery_lat': nearest['delivery_lat'],
            'delivery_lng': nearest['delivery_lng'],
            'cumulative_distance': round(total_dist, 2)
        })
        current_lat, current_lng = nearest['delivery_lat'], nearest['delivery_lng']
        remaining.remove(nearest)
    return route
