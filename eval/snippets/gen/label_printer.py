"""
Generate shipping label payloads for home goods e-commerce orders.

Input: orders.orders_raw, inventory.inventory_items
Output: shipping.shipping_labels
"""

from typing import Dict, Any
import hashlib


def generate_shipping_label_payload(order_id: int, customer_address: Dict[str, str], 
                                    items: list, carrier: str = "USPS") -> Dict[str, Any]:
    """Generate structured payload for shipping label generation."""
    # Validate required address fields
    required_fields = ["street", "city", "state", "zip_code", "country"]
    if not all(f in customer_address for f in required_fields):
        raise ValueError("Missing required address fields")
    
    # Format address string
    address_str = f"{customer_address['street']}, {customer_address['city']}, {customer_address['state']} {customer_address['zip_code']}, {customer_address['country']}"
    
    # Generate barcode string (order_id + checksum)
    checksum = hashlib.md5(f"{order_id}{carrier}".encode()).hexdigest()[:8]
    barcode = f"HL-{order_id:08d}-{checksum}"
    
    # Determine service code
    service_codes = {
        "USPS": {"standard": "1", "express": "2", "priority": "3"},
        "UPS": {"ground": "03", "2day": "02", "overnight": "01"},
        "FedEx": {"ground": "GND", "2day": "2DA", "overnight": "OVR"}
    }
    service_code = service_codes.get(carrier, {}).get("standard", "1")
    
    return {
        "order_id": order_id,
        "customer_address": address_str,
        "items": items,
        "carrier": carrier,
        "service_code": service_code,
        "barcode": barcode,
        "label_format": "PDF_4x6",
        "created_at": "CURRENT_TIMESTAMP"
    }
