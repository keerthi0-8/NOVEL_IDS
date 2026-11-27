import torch
from novel_ids.ood.stage11 import ClassConditionalMahalanobis, energy_score, OODScoreCombiner

def test_mahalanobis_separates_shift_and_excludes_small_class():
    torch.manual_seed(17)
    x0=0.15*torch.randn(30,128); x1=3.0+0.15*torch.randn(30,128); small=8.0+0.1*torch.randn(2,128)
    x=torch.cat([x0,x1,small]); y=torch.tensor([0]*30+[1]*30+[2]*2)
    model=ClassConditionalMahalanobis(min_class_samples=5,shrinkage=0.2,ridge=0.01).fit(x,y)
    assert model.excluded_classes=={2:2}
    in_score=model.score(torch.cat([x0[:8],x1[:8]])).mean()
    shifted=model.score(20.0+0.1*torch.randn(8,128)).mean()
    assert torch.isfinite(in_score) and torch.isfinite(shifted) and shifted>in_score

def test_energy_score_direction_and_combination():
    familiar=energy_score(torch.tensor([[10.0,0.0],[9.0,0.0]]))
    unfamiliar=energy_score(torch.tensor([[0.0,0.0],[0.1,0.0]]))
    assert unfamiliar.mean()>familiar.mean()
    train_m=torch.linspace(0.1,1.0,30); train_e=torch.linspace(-10.0,-8.0,30)
    combiner=OODScoreCombiner(weights=(0.7,0.3)).fit(train_m,train_e)
    scores=combiner.score(torch.tensor([0.5,10.0]),torch.tensor([-9.0,0.0]))
    assert torch.isfinite(scores).all() and scores[1]>scores[0]
