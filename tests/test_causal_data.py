"""Regression checks for transaction units and whole-file stratified sampling."""
import numpy as np
import pandas as pd
from shared.causal_data import retailhero, criteo


def test_retail_transactions_are_counted_once_across_chunks(tmp_path):
    pd.DataFrame({'client_id':['a','b','c','d'], 'treatment_flg':[0,1,0,1],
                  'target':[0,1,1,0]}).to_csv(tmp_path/'uplift_train.csv.gz',index=False)
    pd.DataFrame({'client_id':['a','b','c','d'], 'age':[30,-100,45,50],
                  'gender':['M','F','U','F']}).to_csv(tmp_path/'clients.csv.gz',index=False)
    pd.DataFrame({'client_id':['a','a','a','a','b'],
                  'transaction_id':['t1','t2','t1','t2','t3'],
                  'transaction_datetime':['2019-01-01','2019-01-02','2019-01-01','2019-01-02','2019-01-03'],
                  'purchase_sum':[100,50,100,50,20]}).to_csv(tmp_path/'purchases.csv.gz',index=False)
    df,base,expanded,info=retailhero(tmp_path,n=4,chunksize=2)
    df=df.set_index('client_id')
    assert df.loc['a','txn_count']==2
    assert df.loc['a','total_spend']==150
    assert df.loc['a','avg_ticket']==75
    assert df.loc['a','recency_days']==2
    assert pd.isna(df.loc['b','age'])
    assert df.loc['c','txn_count']==0
    assert info['unique_transactions']==3
    assert not {'first_redeem_date','target','treatment_flg'} & set(expanded)


def test_criteo_sampling_preserves_treatment_ratio_and_uses_full_file(tmp_path):
    source=pd.DataFrame({f'f{i}':np.arange(200,dtype=float) for i in range(12)})
    source['treatment']=[0]*30+[1]*170
    source['visit']=np.arange(200)%2
    source['conversion']=np.arange(200)%3==0
    source['exposure']=1
    source.to_csv(tmp_path/'criteo-research-uplift-v2.1.csv.gz',index=False)
    a,features,info=criteo(tmp_path,n=40,seed=42,chunksize=17)
    b,_,_=criteo(tmp_path,n=40,seed=42,chunksize=200)
    assert len(a)==40 and a.row_id.is_unique
    assert a.treatment.mean()==.85
    assert a.row_id.max()>40
    assert 'exposure' not in a and 'visit' not in features
    assert info['population_counts']==[30,170]
    assert set(a.row_id)==set(b.row_id)
