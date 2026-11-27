"""Approval-gated external taxonomy schema for Stage 11."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import yaml

class TaxonomyError(ValueError): pass
class TaxonomyNotApprovedError(TaxonomyError): pass
class _UniqueKeyLoader(yaml.SafeLoader): pass

def _construct_unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result: raise TaxonomyError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result
_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)

def _dataset_specific_prefix(dataset: str) -> str:
    # Match the user-approved human-readable taxonomy names.
    return {"unsw_nb15": "UNSW_", "nsl_kdd": "NSL_"}.get(dataset, dataset.upper() + "::")

@dataclass(frozen=True)
class Taxonomy:
    name: str
    status: str
    classes: tuple[str, ...]
    dataset_labels: Mapping[str, tuple[str, ...]]
    mappings: Mapping[str, Mapping[str, str]]
    mapping_status: Mapping[str, Mapping[str, str]]
    def map_labels(self, dataset: str, labels: Sequence[str]) -> list[int]:
        if dataset not in self.mappings: raise TaxonomyError(f"Dataset {dataset!r} is not declared")
        mapping, index = self.mappings[dataset], {name:i for i,name in enumerate(self.classes)}
        result=[]
        for raw in labels:
            label=str(raw)
            if label not in mapping: raise TaxonomyError(f"Unmapped label {label!r} for dataset {dataset!r}")
            result.append(index[mapping[label]])
        return result

def _need(ok, message):
    if not ok: raise TaxonomyError(message)

def _parse(doc):
    _need(isinstance(doc,dict), "Taxonomy document must be a mapping")
    required={"schema_version","name","status","classes","dataset_specific_classes","datasets"}
    _need(required.issubset(doc), f"Missing fields: {sorted(required-set(doc))}")
    _need(doc["schema_version"]==1, "Only schema_version 1 is supported")
    _need(isinstance(doc["name"],str) and doc["name"].strip(), "name must be non-empty")
    _need(doc["status"] in {"approved","proposed"}, "status must be approved or proposed")
    classes=doc["classes"]
    _need(isinstance(classes,list) and classes, "classes must be non-empty")
    _need(all(isinstance(x,str) and x.strip() for x in classes), "Class names must be non-empty strings")
    _need(len(classes)==len(set(classes)), "Duplicate unified class names")
    class_set=set(classes); specific=doc["dataset_specific_classes"]
    _need(isinstance(specific,list) and len(specific)==len(set(specific)), "Invalid or duplicate dataset_specific_classes")
    _need(set(specific).issubset(class_set), "Dataset-specific classes must appear in classes")
    raw_ds=doc["datasets"]
    _need(isinstance(raw_ds,dict) and raw_ds, "datasets must be non-empty")
    mappings, statuses, labels_by_ds, used={},{},{},set()
    for ds,entry in raw_ds.items():
        _need(isinstance(ds,str) and ds.strip() and isinstance(entry,dict), "Invalid dataset entry")
        _need({"original_labels","mappings"}.issubset(entry), f"{ds}: original_labels and mappings required")
        original,rows=entry["original_labels"],entry["mappings"]
        _need(isinstance(original,list) and original, f"{ds}: original_labels must be non-empty")
        _need(all(isinstance(x,str) and x.strip() for x in original), f"{ds}: invalid original label")
        _need(len(original)==len(set(original)), f"{ds}: duplicate original_labels")
        _need(isinstance(rows,list) and rows, f"{ds}: mappings must be non-empty list")
        mp,st={},{}
        for row in rows:
            _need(isinstance(row,dict) and {"original_label","unified_class","status","evidence"}.issubset(row), f"{ds}: malformed mapping row")
            src,dst,state,evidence=row["original_label"],row["unified_class"],row["status"],row["evidence"]
            _need(isinstance(src,str) and src.strip(), f"{ds}: empty original label")
            _need(src not in mp, f"{ds}: duplicate mapping for {src!r}")
            _need(dst in class_set, f"{ds}/{src}: target {dst!r} is not a declared class")
            _need(state in {"approved","proposed"}, f"{ds}/{src}: invalid mapping status")
            _need(isinstance(evidence,str) and evidence.strip(), f"{ds}/{src}: evidence required")
            mp[src],st[src]=dst,state; used.add(dst)
        _need(set(original)==set(mp), f"{ds}: unmapped labels {sorted(set(original)-set(mp))}; extra {sorted(set(mp)-set(original))}")
        prefix=_dataset_specific_prefix(ds)
        for target in set(mp.values()):
            if target in specific: _need(target.startswith(prefix), f"{target!r} must be prefixed for {ds!r}")
        for target in specific:
            if target.startswith(prefix): _need(target in mp.values(), f"{ds}: dataset-specific class {target!r} is not mapped")
        mappings[ds],statuses[ds],labels_by_ds[ds]=mp,st,tuple(original)
    _need(class_set==used, f"Empty unified classes with no mappings: {sorted(class_set-used)}")
    for target in specific:
        owners=[ds for ds,mp in mappings.items() if target in mp.values()]
        _need(len(owners)==1 and target.startswith(_dataset_specific_prefix(owners[0])), f"Dataset-specific class {target!r} must map from exactly its prefixed dataset")
    return Taxonomy(doc["name"],doc["status"],tuple(classes),labels_by_ds,mappings,statuses)

def load_taxonomy(path: str|Path, purpose: str="training") -> Taxonomy:
    """Inspection may read proposals; training/evaluation require all approvals."""
    if purpose not in {"training","evaluation","inspection"}: raise TaxonomyError("purpose must be training, evaluation, or inspection")
    with Path(path).open("r",encoding="utf-8") as stream: tax=_parse(yaml.load(stream,Loader=_UniqueKeyLoader))
    if purpose!="inspection":
        if tax.status!="approved": raise TaxonomyNotApprovedError("Overall taxonomy is not approved; use blocked")
        pending=[f"{ds}/{label}" for ds,rows in tax.mapping_status.items() for label,state in rows.items() if state!="approved"]
        if pending: raise TaxonomyNotApprovedError(f"Unapproved mappings block use: {pending}")
    return tax

def map_labels(taxonomy: Taxonomy,dataset: str,labels: Sequence[str])->list[int]:
    return taxonomy.map_labels(dataset,labels)

