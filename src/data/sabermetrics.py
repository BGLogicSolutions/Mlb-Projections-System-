import os

import numpy as np
import pandas as pd
import requests


def get_advanced_stats(year, group="hitting"):
    print(f"Descargando {group} {year}...")
    url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&season={year}&sportId=1"
    res = requests.get(url).json()

    stats_list = []
    if "stats" in res and len(res["stats"]) > 0:
        for split in res["stats"][0]["splits"]:
            p_name = split["player"]["fullName"]
            s = split["stat"]

            if group == "hitting":
                # Métricas de Bateo Avanzadas
                ab = float(s.get("atBats", 1))
                h = float(s.get("hits", 0))
                d2b = float(s.get("doubles", 0))
                d3b = float(s.get("triples", 0))
                hr = float(s.get("homeRuns", 0))
                bb = float(s.get("baseOnBalls", 0))
                hbp = float(s.get("hitByPitch", 0))
                sf = float(s.get("sacrificeFlies", 0))

                # wOBA (Weighted On-Base Average) - Pesos 2024-2026 aprox.
                woba = (
                    (
                        0.69 * bb
                        + 0.72 * hbp
                        + 0.89 * (h - d2b - d3b - hr)
                        + 1.27 * d2b
                        + 1.62 * d3b
                        + 2.10 * hr
                    )
                    / (ab + bb + hbp + sf)
                    if (ab + bb + hbp + sf) > 0
                    else 0
                )

                stats_list.append(
                    {
                        "name": p_name,
                        "year": year,
                        "type": "Hitter",
                        "OBP": float(s.get("obp", 0)),
                        "ISO": float(s.get("slg", 0)) - float(s.get("avg", 0)),
                        "wOBA": woba,
                        "HardHit_pct": float(s.get("hardHitPct", 40.0)),
                        "Barrel_pct": float(s.get("barrelPct", 8.0)),
                        "WAR": float(s.get("war", woba * 10)),  # Proxy WAR
                    }
                )

            elif group == "pitching":
                # Métricas de Pitcheo Avanzadas
                ip = float(s.get("inningsPitched", 1))
                k = float(s.get("strikeOuts", 0))
                bb = float(s.get("baseOnBalls", 0))
                hr = float(s.get("homeRuns", 0))
                hbp = float(s.get("hitByPitch", 0))

                # FIP (Fielding Independent Pitching)
                fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + 3.15 if ip > 0 else 0
                # SIERA (Skill-interactive ERA) - Simplificada
                siera = 3.25 + 1.5 * (bb / max(ip, 1)) - 1.25 * (k / max(ip, 1))

                stats_list.append(
                    {
                        "name": p_name,
                        "year": year,
                        "type": "Pitcher",
                        "FIP": fip,
                        "xFIP": fip * 0.94,
                        "SIERA": siera,
                        "WHIP": float(s.get("whip", 0)),
                        "K_BB_pct": (k - bb) / max(float(s.get("battersFaced", 1)), 1),
                        "ERA_minus": (float(s.get("era", 4.0)) / 4.20) * 100,
                    }
                )
    return pd.DataFrame(stats_list)


if __name__ == "__main__":
    all_data = []
    for y in [2024, 2025, 2026]:
        all_data.append(get_advanced_stats(y, "hitting"))
        all_data.append(get_advanced_stats(y, "pitching"))

    df_final = pd.concat(all_data)
    df_final.to_csv("data/mlb_advanced_intelligence.csv", index=False)
    print("✅ Dataset Sabermétrico Creado.")
