# AgriShield

Soil-risk screening from public soil surveys, with live satellite and weather at inference.

## Pipeline

1. **Training table** — LUCAS 2018 chemistry joined to LUCAS 2015 texture, plus WoSIS topsoil (organic carbon, pH, clay, sand, silt, nitrogen). Cleaned and written to `data/agrishield_training.csv`.
2. **Model** — random forest predicting acidic soil (`pH < 5.5`) from organic carbon, nitrogen, and texture. pH is not used as a feature.
3. **Inference** — for a new lat/lon, Google Earth Engine supplies Sentinel-2 NDVI and WorldClim climate; OpenWeather supplies current conditions. Those live values are context, not extra labelled training rows.

Current weather is not joined onto historical soil samples. Climate climatology (WorldClim) may be added as columns later; it is not required to train the first model.

## Layout

| Path | Role |
|---|---|
| `agrishield/` | Dataset build, GEE, weather, model, inference |
| `Agrishield.ipynb` | End-to-end notebook |
| `data/agrishield_training.csv` | Harmonised training table |
| `models/` | Saved estimator |
| `LUCAS 2015/`, `LUCAS 2018/`, `WoSIS/` | Raw surveys |

## Setup

```text
pip install -r requirements.txt
cp .env.example .env
```

Set `GEE_PROJECT_ID` and `OPENWEATHER_API_KEY` in `.env`. Rebuild the table with `python build_training_csv.py` or run the notebook from the top.
