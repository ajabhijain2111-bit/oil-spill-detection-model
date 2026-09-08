from inference import detect_oil

result = detect_oil("00000.tif")

print("\n========== TEAM INTEGRATION TEST ==========")

print("Oil detected:",
      result["prediction"]["oil_detected"])

print("Oil percentage:",
      result["prediction"]["cleaned_oil_percentage"])

print("Latitude:",
      result["centroid"]["latitude"])

print("Longitude:",
      result["centroid"]["longitude"])

print("Area:",
      result["area"]["estimated_area_km2"],
      "km²")

print("===========================================")