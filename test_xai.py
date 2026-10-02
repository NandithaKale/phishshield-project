from app.services.xai_service import explain_url

url = "http://secure-login-paypal.xyz/verify"

print("\nTesting PhishShield XAI")
print("======================")
print("URL:", url)
print("\nTop contributing features:")
print("--------------------------")

results = explain_url(url)

for item in results:
    print(
        f"{item['feature']} | "
        f"Value: {item['value']} | "
        f"SHAP: {item['impact']}"
    )
    print(f"Reason: {item['reason']}")
    print()