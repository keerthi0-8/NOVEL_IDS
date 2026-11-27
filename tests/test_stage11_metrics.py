import math
import numpy as np
from novel_ids.evaluation.stage11_metrics import classification_metrics

def test_metrics_match_hand_computed_example():
    y=np.array([0,0,1,1,2]); pred=np.array([0,1,1,2,2])
    probs=np.array([[.8,.1,.1],[.2,.7,.1],[.1,.8,.1],[.1,.2,.7],[.1,.2,.7]])
    result=classification_metrics(y,pred,["A","B","C"],probabilities=probs,
        uncertain_mask=np.array([False,True,False,False,True]),
        ood_true=np.array([0,0,1,1,1]),ood_scores=np.array([.1,.2,.8,.9,.7]))
    assert result["confusion_matrix"]==[[1,1,0],[0,1,1],[0,0,1]]
    assert np.allclose(result["normalized_confusion_matrix"],[[.5,.5,0],[0,.5,.5],[0,0,1]])
    assert math.isclose(result["accuracy"],.6)
    assert math.isclose(result["macro_f1"],11/18)
    assert math.isclose(result["weighted_f1"],.6)
    assert math.isclose(result["fpr"],7/36)
    assert math.isclose(result["fnr"],1/3)
    assert result["per_class"]["A"]["support"]==2
    assert result["per_class"]["B"]["support"]==2
    assert result["per_class"]["C"]["support"]==1
    assert math.isclose(result["uncertain_rate"],.4)
    assert math.isclose(result["accuracy_on_accepted"],2/3)
    assert result["ood"]=={"auroc":1.0,"aupr":1.0,"fpr_at_95_tpr":0.0}
    assert result["roc_auc_ovr_macro"] is not None and result["pr_auc_ovr_macro"] is not None

def test_metrics_handle_unsupported_class_without_nan():
    result=classification_metrics([0,0],[0,0],["present","absent"])
    assert result["per_class"]["absent"]["support"]==0
    assert result["normalized_confusion_matrix"][1]==[0.0,0.0]
    assert all(math.isfinite(result[k]) for k in ["accuracy","precision","recall","f1","macro_f1","weighted_f1","fpr","fnr","detection_rate"])
