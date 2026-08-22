"""Deterministically generate fictional NovaTech workshop documents."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = [
("NovaBook Air","NBA-100","Laptop",1200,"24 months","Available","Lightweight laptop for everyday productivity."),
("NovaBook Pro","NBP-200","Laptop",1800,"36 months","Available","Performance laptop for creative professionals."),
("NovaPhone X","NPX-300","Phone",1000,"24 months","Low stock","Flagship phone with an advanced camera."),
("NovaPhone Lite","NPL-310","Phone",600,"18 months","Available","Approachable phone with all-day battery."),
("NovaWatch","NW-400","Wearable",300,"12 months","Available","Fitness and notification smartwatch."),
("NovaBuds","NBD-500","Headphones",200,"12 months","Available","Noise-cancelling wireless earbuds."),
("NovaPad","NPD-600","Tablet",800,"24 months","Available","Portable tablet for work and entertainment."),
("NovaDock","NDK-700","Accessory",150,"12 months","Back order","USB-C desktop dock with dual-display support."),
("NovaCam","NCM-800","Camera",900,"24 months","Available","Compact camera for travel creators."),
]
FAQ='''NovaTech FAQ\n\nQ: How can I track my order?\nA: Use the tracking link emailed after dispatch.\n\nQ: How long does shipping take?\nA: Standard shipping takes 3–5 business days; express shipping takes 1–2.\n\nQ: Can I cancel an order?\nA: Yes, within two hours of placing it if fulfillment has not begun.\n\nQ: Which payment methods are supported?\nA: Visa, Mastercard, NovaPay, and PayPal.\n\nQ: Can I change my delivery address?\nA: Contact support before the order is dispatched.\n\nQ: How do I contact support?\nA: Email support@novatech.example, Monday–Friday, 09:00–17:00 fictional local time.\n'''
POLICIES='''NovaTech Policies (fictional)\n\nReturn policy: Products can be returned within 21 calendar days after delivery. Opened headphones can only be returned if defective.\n\nRefund rules: Approved refunds are processed within 5 business days after inspection to the original payment method.\n\nWarranty rules: Warranty covers manufacturing defects, but excludes accidental damage and normal battery wear. Proof of purchase is required.\n\nShipping policy: Standard delivery is free for orders of $500 or more. Express delivery costs $25.\n\nDamaged product policy: Report transit damage with photographs within 48 hours of delivery; NovaTech supplies a prepaid return label.\n'''
def main() -> None:
    data = ROOT / "data"; data.mkdir(exist_ok=True)
    blocks = [f"Product: {n}\nSKU: {s}\nCategory: {c}\nPrice: ${p}\nWarranty: {w}\nStock: {st}\nDescription: {d}" for n,s,c,p,w,st,d in PRODUCTS]
    (data / "products.txt").write_text("NovaTech Product Catalog (fictional)\n\n"+"\n\n".join(blocks)+"\n", encoding="utf-8")
    (data / "faq.txt").write_text(FAQ, encoding="utf-8")
    (data / "policies.txt").write_text(POLICIES, encoding="utf-8")
    print("Generated 3 text documents")
if __name__ == "__main__": main()
