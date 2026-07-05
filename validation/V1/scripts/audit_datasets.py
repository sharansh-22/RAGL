"""
RAGL — Validation Study V1: Datasets Audit
===========================================
Systematically scans Gold Datasets for duplicates, missing files, or anomalies.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

def main():
    ragl_root = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = ragl_root / "evaluation" / "datasets"
    data_dir = ragl_root / "data"
    
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
        
    valid_pdfs = {p.name for p in data_dir.rglob("*.pdf")}
    
    all_queries = []
    all_docs = []
    errors = []
    
    for ds_path in dataset_dir.glob("*.json"):
        with open(ds_path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                errors.append(f"Invalid JSON in {ds_path.name}: {e}")
                continue
                
        for q_data in data:
            q_id = q_data.get("id", "unknown_id")
            query_str = q_data.get("query", "").strip()
            if not query_str:
                errors.append(f"{ds_path.name} -> {q_id}: Missing or empty query.")
            
            all_queries.append(query_str.lower())
            
            expected = q_data.get("expected_sources", [])
            category = q_data.get("category", "")
            if not expected and category != "adversarial":
                errors.append(f"{ds_path.name} -> {q_id}: Missing expected_sources.")
                
            for doc in expected:
                all_docs.append(doc)
                if doc not in valid_pdfs:
                    errors.append(f"{ds_path.name} -> {q_id}: Expected document '{doc}' not found in data/ directory.")
                    
    # Check duplicates
    q_counts = Counter(all_queries)
    duplicates = [q for q, count in q_counts.items() if count > 1]
    for d in duplicates:
        errors.append(f"Duplicate query detected across dataset: '{d}'")
        
    doc_counts = Counter(all_docs)
    
    passed = len(errors) == 0
    
    out_dir = Path(__file__).resolve().parent.parent
    out_file = out_dir / "dataset_validation.json"
    
    with open(out_file, "w") as f:
        json.dump({
            "audit_passed": passed,
            "errors": errors,
            "document_frequency": dict(doc_counts),
            "total_queries": len(all_queries)
        }, f, indent=2)
        
    print(f"Dataset Audit Passed: {passed}")
    if not passed:
        print("Errors:")
        for e in errors:
            print(f" - {e}")
            
    print(f"Saved evidence to {out_file}")

if __name__ == "__main__":
    main()
