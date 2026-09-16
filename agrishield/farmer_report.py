def farmer_report(lat: float, lon: float, model, live_inputs: dict, prediction: dict, weather: dict | None = None) -> dict:
    """Turn a raw model prediction into a farmer-facing summary.

    prediction is whatever agrishield.model.predict_proba() returned.
    live_inputs is whatever agrishield.inference.live_model_inputs() returned
    (used here just to report a couple of readable numbers alongside the verdict).
    """
    acidic_prob = prediction["probability"].get(1, 0.0)
    predicted_acidic = prediction["predicted"] == 1

    # Thresholds -- same spirit as the if-else layer in the original blueprint,
    # just built around a probability instead of a raw carbon number.
    if acidic_prob >= 0.65:
        verdict = "acidic"
        headline = "Soil likely acidic"
        action = "Consider a lime treatment and a follow-up soil test before planting."
        color = "#EF4444"  # red
    elif acidic_prob >= 0.40:
        verdict = "uncertain"
        headline = "Uncertain -- borderline result"
        action = "A physical soil test is recommended before making changes."
        color = "#F59E0B"  # amber
    else:
        verdict = "stable"
        headline = "Soil pH looks stable"
        action = "No action needed based on this scan."
        color = "#22C55E"  # green

    # Known-limitation flag: only Sentinel-2 bands are missing when the model
    # had to fall back on imputed values for this exact point (rare for a
    # live farmer call, since GEE almost always returns *something* for a
    # real coordinate -- but worth surfacing honestly if it happens).
    used_imputed_satellite = live_inputs.get("B2") is None

    report = {
        "location": {"latitude": lat, "longitude": lon},
        "headline": headline,
        "verdict": verdict,
        "confidence_pct": round(acidic_prob * 100, 1) if predicted_acidic else round((1 - acidic_prob) * 100, 1),
        "recommended_action": action,
        "color": color,
        "context": {
            "ndvi": live_inputs.get("ndvi"),
            "avg_temp_c": live_inputs.get("tmean_c"),
            "avg_annual_rainfall_mm": live_inputs.get("precip_mm"),
            "current_weather": weather.get("description") if weather else None,
        },
        "caveats": (
            ["Satellite data unavailable for this exact spot/date -- result relies more on climate and soil-texture maps."]
            if used_imputed_satellite else []
        ),
    }
    return report
