import numpy as np
from shared.criteo.inference import policy_stats, bootstrap


def test_weighted_ranking_matches_explicit_resampling():
    score=np.array([.1,.9,.3,.6])
    t=np.array([0,1,1,0]);y=np.array([0,1,0,1]);e=np.full(4,.5)
    mu0=np.array([.1,.2,.1,.3]);mu1=mu0+.1
    w=np.array([2,1,0,3]);ids=np.repeat(np.arange(4),w)
    a=policy_stats(score,t,y,e,mu0,mu1,weights=w,k=.5)
    b=policy_stats(score[ids],t[ids],y[ids],e[ids],mu0[ids],mu1[ids],k=.5)
    for key in a:
        np.testing.assert_allclose(a[key],b[key])


def test_all_targeted_policy_equals_random_and_paired_same_scores_zero():
    t=np.array([0,1,1,0]);y=np.array([0,1,0,1]);e=np.full(4,.5)
    scores=np.array([.1,.9,.3,.6]);zeros=np.zeros(4)
    assert abs(policy_stats(scores,t,y,e,zeros,zeros,k=1)['PolicyGain'])<1e-12
    _,ci=bootstrap({'a':scores,'b':scores},t,y,e,zeros,zeros,{'same':('a','b')},B=10)
    assert ci['same']['estimate']==ci['same']['lo']==ci['same']['hi']==0
