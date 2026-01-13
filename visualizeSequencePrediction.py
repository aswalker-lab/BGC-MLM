# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 16:37:59 2025

@author: Allison Walker
"""
#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse
from pathlib import Path

from Bio import SeqIO


def detect_pfam_in_feature(feat):
    """
    Detect a PFAM ID in a feature's qualifiers.

    Prioritizes the PFAM_domain qualifier, e.g.:
      /PFAM_domain="PF00069 protein_kinase"
    """
    # 0) PFAM_domain qualifier (your case)
    if "PFAM_domain" in feat.qualifiers:
        val = feat.qualifiers["PFAM_domain"][0]
        parts = val.replace(":", " ").split()
        for p in parts:
            if p.upper().startswith("PF"):
                return p
        return val

    # 1) db_xref, e.g. "Pfam:PF00001"
    for key in ("db_xref",):
        for val in feat.qualifiers.get(key, []):
            if "pfam" in val.lower():
                if ":" in val:
                    return val.split(":", 1)[1]
                else:
                    return val

    # 2) region_name / label might contain PFAM name/id
    for key in ("region_name", "label"):
        for val in feat.qualifiers.get(key, []):
            if "pfam" in val.lower():
                return val

    # 3) note might say "PFAM: PF00001"
    for note in feat.qualifiers.get("note", []):
        if "pfam" in note.lower():
            parts = note.replace(":", " ").split()
            for i, p in enumerate(parts):
                if p.lower().startswith("pfam"):
                    if i + 1 < len(parts) and parts[i + 1].upper().startswith("PF"):
                        return parts[i + 1]
                    return p

    return None


def detect_score_in_feature(feat):
    """
    Try to detect a numeric score in the feature (bitscore, score=, etc.).
    Returns float or None.
    """
    for key in ("score", "bitscore"):
        if key in feat.qualifiers:
            try:
                return float(feat.qualifiers[key][0])
            except (ValueError, TypeError):
                pass

    for val in feat.qualifiers.get("PFAM_domain", []):
        lower = val.lower()
        for tag in ("score=", "bitscore=", "bit score="):
            if tag in lower:
                try:
                    after = lower.split(tag, 1)[1].strip()
                    token = after.split()[0].strip(",;)")
                    return float(token)
                except (ValueError, IndexError):
                    pass

    for note in feat.qualifiers.get("note", []):
        lower = note.lower()
        for tag in ("score=", "bitscore=", "bit score="):
            if tag in lower:
                try:
                    after = lower.split(tag, 1)[1].strip()
                    token = after.split()[0].strip(",;)")
                    return float(token)
                except (ValueError, IndexError):
                    pass

    return None


def parse_gbk(gbk_path):
    """
    Parse a GenBank file and return a list of genes with attached PFAM domains.
    """
    record = next(SeqIO.parse(gbk_path, "genbank"))
    score_cutoff = 20
    pfam_features = []
    for feat in record.features:
        if feat.type != "PFAM_domain":
            continue
        pfam_id = feat.qualifiers["db_xref"][0]
        description = feat.qualifiers["description"][0]
        if pfam_id is None:
            continue

        score = detect_score_in_feature(feat)
        if score < score_cutoff:
            continue
        pfam_features.append(
            {
                "start": int(feat.location.start),
                "end": int(feat.location.end),
                "strand": feat.location.strand,
                "pfam_id": pfam_id,
                "description": description,
                "score": score,
            }
        )

    genes = []
    for feat in record.features:
        if feat.type != "CDS":
            continue

        start = int(feat.location.start)
        end = int(feat.location.end)
        strand = feat.location.strand
        locus_tag = feat.qualifiers.get("locus_tag", ["unknown"])[0]
        product = feat.qualifiers.get("product", ["unknown product"])[0]

        domains = []
        for dom in pfam_features:
            if dom["start"] >= start and dom["end"] <= end:
                domains.append(dom)

        genes.append(
            {
                "start": start,
                "end": end,
                "strand": strand,
                "locus_tag": locus_tag,
                "product": product,
                "domains": domains,
            }
        )

    return record, genes, pfam_features


def add_placeholder_scores_if_missing(genes):
    all_domains = [d for g in genes for d in g["domains"]]
    if not all_domains:
        return

    if all(d.get("score") is None for d in all_domains):
        for i, d in enumerate(all_domains):
            d["score"] = 0.5 + 0.1 * (i % 6)
    else:
        for d in all_domains:
            if d.get("score") is None:
                d["score"] = 0.7


def add_placeholder_domains_if_none(genes):
    has_real_domains = any(g["domains"] for g in genes)
    if has_real_domains:
        return

    print("No PFAM domains found; adding placeholder domains and scores.")

    for gi, g in enumerate(genes):
        gene_len = g["end"] - g["start"]
        if gene_len <= 0:
            continue

        n_domains = 3
        step = gene_len / (n_domains + 1)

        g["domains"] = []
        for i in range(n_domains):
            center = g["start"] + (i + 1) * step
            dom_len = step * 0.6
            dom_start = int(center - dom_len / 2)
            dom_end = int(center + dom_len / 2)
            score = 0.5 + 0.2 * ((gi + i) % 3)

            g["domains"].append(
                {
                    "start": dom_start,
                    "end": dom_end,
                    "strand": g["strand"],
                    "pfam_id": f"PFAM_{gi+1}_{i+1}",
                    "score": score,
                }
            )


def scale_positions(genes, width_px=1000, margin_bp=100):
    all_starts = [g["start"] for g in genes]
    all_ends = [g["end"] for g in genes]
    global_start = min(all_starts) - margin_bp
    global_end = max(all_ends) + margin_bp
    span = max(global_end - global_start, 1)

    svg_width = width_px
    x_scale = svg_width / span
    return x_scale, global_start, global_end, svg_width


def gene_arrow_polygon(x_start, x_end, y_center, height, strand):
    body_height = height * 0.6
    head_length = min((x_end - x_start) * 0.2, 20)
    if head_length < 5:
        head_length = 5

    if strand >= 0:
        body_end = x_end - head_length
        points = [
            (x_start, y_center - body_height / 2),
            (body_end, y_center - body_height / 2),
            (body_end, y_center - height / 2),
            (x_end, y_center),
            (body_end, y_center + height / 2),
            (body_end, y_center + body_height / 2),
            (x_start, y_center + body_height / 2),
        ]
    else:
        body_start = x_start + head_length
        points = [
            (x_end, y_center - body_height / 2),
            (body_start, y_center - body_height / 2),
            (body_start, y_center - height / 2),
            (x_start, y_center),
            (body_start, y_center + height / 2),
            (body_start, y_center + body_height / 2),
            (x_end, y_center + body_height / 2),
        ]
    return points


def generate_svg(record, genes, svg_width=1000, svg_height=400):
    """
    Generate SVG markup for:
    - Gene arrows with PFAM domains as rectangles (same height as arrow body)
    - PFAM labels shown as tooltips on hover (via <title> in SVG)
    - An aligned PFAM score graph underneath
    """
    x_scale, global_start, global_end, svg_width = scale_positions(genes, svg_width)

    gene_band_y = 120
    gene_height = 40
    body_height = gene_height * 0.6  # arrow body height

    # Collect PFAM domain points for the graph
    pfam_points = []
    for gene in genes:
        for dom in gene["domains"]:
            if dom.get("score") is None:
                continue
            mid_bp = (dom["start"] + dom["end"]) / 2
            x = (mid_bp - global_start) * x_scale
            pfam_points.append(
                {"x": x, "score": dom["score"], "pfam_id": dom["pfam_id"], "MLM_score": dom["MLM_score"], "accurate": dom["accurate"]}
            )

    scores = [p["MLM_score"] for p in pfam_points if p["MLM_score"] is not None]
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            max_score = min_score + 1.0
    else:
        min_score = 0.0
        max_score = 1.0

    graph_top = 220
    graph_height = 120
    graph_bottom = graph_top + graph_height

    def score_to_y(score):
        frac = (score - min_score) / (max_score - min_score)
        return graph_bottom - frac * graph_height

    svg_parts = [
        f'<svg width="{svg_width + 100}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width + 100} {svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        '<style>',
        '.gene-arrow { fill: #cccccc; stroke: #333333; stroke-width: 1px; }',
        '.pfam-rect { fill: #88c; fill-opacity: 0.7; stroke: #333; stroke-width: 0.5px; }',
        '.gene-label { font-size: 10px; text-anchor: start; }',
        '.axis-line { stroke: #000; stroke-width: 1px; }',
        '.graph-point { fill: #c33; }',
        '.graph-line { stroke: #c33; stroke-width: 1px; fill: none; }',
        '</style>',
    ]

    axis_y = gene_band_y
    svg_parts.append(
        f'<line class="axis-line" x1="50" y1="{axis_y}" x2="{svg_width + 50}" y2="{axis_y}" />'
    )

    # Draw genes as arrows
    for gene in genes:
        x_start = 50 + (gene["start"] - global_start) * x_scale
        x_end = 50 + (gene["end"] - global_start) * x_scale
        points = gene_arrow_polygon(x_start, x_end, gene_band_y, gene_height, gene["strand"])
        points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        svg_parts.append(f'<polygon class="gene-arrow" points="{points_str}" />')

        # Gene label: above arrow, anchored at left edge of gene
        gene_label_x = x_start + 2
        gene_label_y = gene_band_y - gene_height * 0.7
        svg_parts.append(
            f'<text class="gene-label" x="{gene_label_x:.1f}" y="{gene_label_y:.1f}" '
            f'transform="rotate(-20 {gene_label_x:.1f} {gene_label_y:.1f})">'
            f'{gene["locus_tag"]}</text>'
        )

        # PFAM rectangles (same height as arrow body), with hover tooltips
        for dom in gene["domains"]:
            dom_x_start = 50 + (dom["start"] - global_start) * x_scale
            dom_x_end = 50 + (dom["end"] - global_start) * x_scale
            dom_width = max(dom_x_end - dom_x_start, 2)
            dom_height = body_height
            dom_y = gene_band_y - dom_height / 2

            # Build tooltip text
            if dom.get("MLM_score") is not None:
                tooltip = f'{dom["pfam_id"]} | {dom["description"]} | score={dom["MLM_score"]:.3f}'
            else:
                tooltip = dom["pfam_id"]
            if dom.get("accurate") > 0.5:
                fill_color =  "#454545"
            else:
                fill_color = "#ff4444"
            # Wrap rect + title in a group so title is associated for hover
            svg_parts.append(
                '<g class="pfam-domain">'
                f'<rect class="pfam-rect" x="{dom_x_start:.1f}" y="{dom_y:.1f}" '
                f'width="{dom_width:.1f}" height="{dom_height:.1f}" '
                f'style="fill:{fill_color}; stroke:#333; stroke-width:0.5px; fill-opacity:0.85;" />'
                f'<title>{tooltip}</title>'
                '</g>'
            )

    # Graph axes
    svg_parts.append(
        f'<line class="axis-line" x1="50" y1="{graph_top}" x2="50" y2="{graph_bottom}" />'
    )
    svg_parts.append(
        f'<line class="axis-line" x1="50" y1="{graph_bottom}" x2="{svg_width + 50}" y2="{graph_bottom}" />'
    )
    svg_parts.append(
        f'<text x="50" y="{graph_top - 5}" style="font-size:10px;">Score</text>'
    )

    svg_parts.append(
        f'<text x="40" y="{graph_bottom:.1f}" style="font-size:8px; text-anchor:end;">{min_score:.2f}</text>'
    )
    svg_parts.append(
        f'<text x="40" y="{graph_top:.1f}" style="font-size:8px; text-anchor:end;">{max_score:.2f}</text>'
    )

    # Graph line + points
    if pfam_points:
        pfam_points_sorted = sorted(pfam_points, key=lambda p: p["x"])
        path_d = []
        for i, p in enumerate(pfam_points_sorted):
            x = 50 + p["x"]
            y = score_to_y(p["MLM_score"])
            svg_parts.append(
                f'<circle class="graph-point" cx="{x:.1f}" cy="{y:.1f}" r="2" />'
            )
            if i == 0:
                path_d.append(f"M {x:.1f} {y:.1f}")
            else:
                path_d.append(f"L {x:.1f} {y:.1f}")
        svg_parts.append(f'<path class="graph-line" d="{" ".join(path_d)}" />')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)




def generate_html(record, svg_markup, out_path):
    title = f"Biosynthetic Gene Cluster Visualization - {record.id}"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{
    font-family: Arial, sans-serif;
}}
.container {{
    margin: 20px;
}}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p>Source: {record.name}</p>
{svg_markup}
</div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    

    gbk_path = Path(args.gbk)
    out_path = Path(args.output)

    record, genes, pfam_features = parse_gbk(gbk_path)
    for line in open(args.acc):
        split_line = line.split(",")
        for i in range(1, len(split_line)):
            pfam_features[i-1]["accurate"] = float(split_line[i])
    for line in open(args.score):
        split_line = line.split(",")
        print(len(split_line))
        for i in range(1, len(split_line)):
            pfam_features[i-1]["MLM_score"] = float(split_line[i])
            
    print(len(pfam_features))
    if not genes:
        print("No CDS features found. Check your GenBank file.")
        return

    if not pfam_features:
        print("No pfam features found. Check your GenBank file.")
        return
    
    domain_num = 0
    for i in range(0, len(genes)):
        for j in range(0, len(genes[i]["domains"])):
            genes[i]["domains"][j] = pfam_features[domain_num]
            domain_num += 1
            #print(g["domains"][j])
        
    print(genes[0]["domains"][0])
    svg_markup = generate_svg(record, genes, svg_width=1000, svg_height=400)
    generate_html(record, svg_markup, out_path)
    print(f"Wrote visualization to {out_path}")


if __name__ == "__main__":
    main()
