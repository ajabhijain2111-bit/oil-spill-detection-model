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

print("\n========== SPILL GEOMETRY ==========")

print("Area (pixels):",
      result["geometry"]["shape"]["area_pixels"])

print("Perimeter (pixels):",
      result["geometry"]["shape"]["perimeter_pixels"])

print("Orientation (degrees):",
      result["geometry"]["shape"]["orientation_degrees"])

print("Width (pixels):",
      result["geometry"]["shape"]["width_pixels"])

print("Height (pixels):",
      result["geometry"]["shape"]["height_pixels"])

print("====================================")

print("\n========== SPILL ORIGIN ==========")

origin = result["origin"]

print("Available:",
      origin["available"])

print("Method:",
      origin["method"])

if origin["available"]:

    center = origin["spill_center"]

    print("\nSpill Center:")
    print("Latitude:",
          center["latitude"])
    print("Longitude:",
          center["longitude"])

    axis = origin["principal_axis"]

    print("\nPrincipal Axis:")
    print("Orientation:",
          axis["orientation_degrees"],
          "degrees")

    print("Length:",
          axis["length_pixels"],
          "pixels")

    for candidate in origin["candidates"]:

        print(
            f"\nCandidate Origin {candidate['candidate_id']}:"
        )

        print(
            "Latitude:",
            candidate["latitude"]
        )

        print(
            "Longitude:",
            candidate["longitude"]
        )

print("==================================")