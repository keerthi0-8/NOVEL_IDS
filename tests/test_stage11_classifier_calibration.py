import inspect
import pytest
import torch
from novel_ids.classification.stage11 import Stage11Classifier, NativeAblationHeads, build_classifier, compute_class_weights
from novel_ids.taxonomy.stage11 import Taxonomy, TaxonomyNotApprovedError
from novel_ids.calibration.temperature import TemperatureScaler, expected_calibration_error

def test_class_weights_training_only_and_clipped():
    labels=["major"]*1000+["minor"]*1+["rare"]*1
    weights,counts=compute_class_weights(labels,["major","minor","rare"],clip_ratio=15)
    assert counts=={"major":1000,"minor":1,"rare":1}
    assert float(weights.max()/weights.min())<=15.0001
    assert torch.isfinite(weights).all()
    with pytest.raises(ValueError,match="between 10 and 20"): compute_class_weights(labels,["major","minor","rare"],clip_ratio=5)

def test_classifier_uses_only_zt_and_supports_cross_entropy():
    head=Stage11Classifier(3,label_smoothing=0.0)
    z=torch.randn(7,128); logits=head(z)
    assert logits.shape==(7,3) and torch.isfinite(logits).all()
    loss=head.loss(logits,torch.tensor([0,1,2,0,1,2,0]))
    assert torch.isfinite(loss)
    with pytest.raises(TypeError): head(z,torch.zeros(7,dtype=torch.long))

def test_native_separate_heads_are_explicit_ablation_only():
    heads=build_classifier("native_separate_ablation",native_classes={"ds_a":["normal","dos"],"ds_b":["normal","probe"]})
    assert isinstance(heads,NativeAblationHeads)
    assert heads(torch.randn(2,128),"ds_a").shape==(2,2)
    with pytest.raises(ValueError,match="mode must be harmonized or native_separate_ablation"):
        build_classifier("native_separate")
    with pytest.raises(TaxonomyNotApprovedError):
        build_classifier("harmonized")

def test_harmonized_factory_requires_an_approved_external_taxonomy():
    proposed=Taxonomy("synthetic-proposal","proposed",("Normal","Attack"),{}, {}, {})
    with pytest.raises(TaxonomyNotApprovedError,match="approved external taxonomy"):
        build_classifier("harmonized",taxonomy=proposed)
    approved=Taxonomy("synthetic-approved","approved",("Normal","Attack"),{}, {}, {})
    head=build_classifier("harmonized",taxonomy=approved)
    assert isinstance(head,Stage11Classifier)
    assert head.num_classes==2

def test_temperature_scaler_fits_validation_only_and_reduces_ece():
    labels=torch.tensor([0,1]*50)
    predicted=labels.clone(); predicted[-20:]=1-predicted[-20:]
    logits=torch.full((100,2),-4.0); logits[torch.arange(100),predicted]=4.0
    before=expected_calibration_error(logits,labels,n_bins=10)
    scaler=TemperatureScaler().fit(logits,labels)
    calibrated=scaler.transform_logits(logits)
    after=expected_calibration_error(calibrated,labels,n_bins=10)
    assert scaler.temperature is not None and scaler.temperature>1.0
    assert after<before
    assert list(inspect.signature(TemperatureScaler.fit).parameters)==["self","validation_logits","validation_labels","max_iter"]
    assert torch.isfinite(scaler.probabilities(logits)).all()
