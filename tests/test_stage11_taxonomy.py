import yaml
import pytest
from novel_ids.taxonomy.stage11 import load_taxonomy, map_labels, TaxonomyError, TaxonomyNotApprovedError

def document(status="approved", mapping_status="approved"):
    return {"schema_version":1,"name":"synthetic-only","status":status,"classes":["Normal","Attack","DS_A::Rare"],
      "dataset_specific_classes":["DS_A::Rare"],
      "datasets":{"ds_a":{"original_labels":["normal","attack","rare"],"mappings":[
        {"original_label":"normal","unified_class":"Normal","status":mapping_status,"evidence":"synthetic test evidence"},
        {"original_label":"attack","unified_class":"Attack","status":mapping_status,"evidence":"synthetic test evidence"},
        {"original_label":"rare","unified_class":"DS_A::Rare","status":mapping_status,"evidence":"synthetic test evidence"}]},
        "ds_b":{"original_labels":["normal","intrusion"],"mappings":[
        {"original_label":"normal","unified_class":"Normal","status":mapping_status,"evidence":"synthetic test evidence"},
        {"original_label":"intrusion","unified_class":"Attack","status":mapping_status,"evidence":"synthetic test evidence"}]}}}

def save(tmp_path, doc):
    path=tmp_path/"taxonomy.yaml"; path.write_text(yaml.safe_dump(doc,sort_keys=False),encoding="utf-8"); return path

def test_approved_synthetic_taxonomy_loads_and_maps(tmp_path):
    tax=load_taxonomy(save(tmp_path,document()),"training")
    assert tax.classes==("Normal","Attack","DS_A::Rare")
    assert map_labels(tax,"ds_a",["normal","rare"])==[0,2]

def test_proposed_taxonomy_refused_for_training_and_evaluation(tmp_path):
    path=save(tmp_path,document(status="proposed"))
    with pytest.raises(TaxonomyNotApprovedError): load_taxonomy(path,"training")
    with pytest.raises(TaxonomyNotApprovedError): load_taxonomy(path,"evaluation")
    assert load_taxonomy(path,"inspection").status=="proposed"

def test_proposed_individual_mapping_blocks_use(tmp_path):
    with pytest.raises(TaxonomyNotApprovedError): load_taxonomy(save(tmp_path,document(mapping_status="proposed")),"evaluation")

def test_duplicate_mapping_rows_fail(tmp_path):
    doc=document(); doc["datasets"]["ds_a"]["mappings"].append(dict(doc["datasets"]["ds_a"]["mappings"][0]))
    with pytest.raises(TaxonomyError,match="duplicate mapping"): load_taxonomy(save(tmp_path,doc),"inspection")

def test_duplicate_yaml_keys_fail(tmp_path):
    path=tmp_path/"duplicate.yaml"
    path.write_text(
        """schema_version: 1
name: x
name: y
status: proposed
classes: [Normal]
dataset_specific_classes: []
datasets: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(TaxonomyError,match="Duplicate YAML key"): load_taxonomy(path,"inspection")

def test_empty_or_unmapped_class_fails(tmp_path):
    doc=document(); doc["classes"].append("Unused")
    with pytest.raises(TaxonomyError,match="Empty unified classes"): load_taxonomy(save(tmp_path,doc),"inspection")
    doc=document(); doc["datasets"]["ds_a"]["mappings"]=doc["datasets"]["ds_a"]["mappings"][:-1]
    with pytest.raises(TaxonomyError,match="unmapped labels"): load_taxonomy(save(tmp_path,doc),"inspection")

def test_unmapped_runtime_label_fails(tmp_path):
    tax=load_taxonomy(save(tmp_path,document()),"training")
    with pytest.raises(TaxonomyError,match="Unmapped label"): map_labels(tax,"ds_a",["not-in-schema"])
