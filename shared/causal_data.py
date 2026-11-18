"""Memory-bounded, outcome-independent feature preparation for the notebooks."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def retailhero(root, n=40_000, seed=42, chunksize=500_000):
    root = Path(root)
    labels = pd.read_csv(root / 'uplift_train.csv.gz', dtype={'client_id': str})
    if n < len(labels):
        labels, _ = train_test_split(labels, train_size=n, random_state=seed,
                                     stratify=labels.treatment_flg)
    clients = pd.read_csv(root / 'clients.csv.gz', dtype={'client_id': str})
    df = labels.merge(clients, on='client_id', validate='one_to_one')
    # Invalid ages become missing; imputation is fitted on training customers only.
    df['age'] = df.age.where(df.age.between(18, 100))
    gender = pd.get_dummies(df.gender, prefix='gender', dtype=float)
    df = pd.concat([df, gender], axis=1)
    base = ['age'] + list(gender.columns)
    wanted = set(df.client_id)
    seen = set()
    pieces = []
    latest = None
    usecols = ['client_id', 'transaction_id', 'transaction_datetime', 'purchase_sum']
    with pd.read_csv(root / 'purchases.csv.gz', usecols=usecols,
                     dtype={'client_id': str, 'transaction_id': str},
                     chunksize=chunksize) as chunks:
        for chunk in chunks:
            # All purchases are documented as pre-SMS. Never infer timing from Y/T.
            dates = pd.to_datetime(chunk.transaction_datetime)
            last = dates.max()
            latest = last if latest is None else max(latest, last)
            chunk = chunk.loc[chunk.client_id.isin(wanted)].copy()
            chunk['date'] = pd.to_datetime(chunk.transaction_datetime)
            chunk = chunk.drop_duplicates(['client_id', 'transaction_id'])
            keys = list(zip(chunk.client_id, chunk.transaction_id))
            keep = [key not in seen for key in keys]
            seen.update(keys)
            chunk = chunk.loc[keep]
            grouped = chunk.groupby('client_id').agg(
                txn_count=('transaction_id', 'size'), total_spend=('purchase_sum', 'sum'),
                last_purchase=('date', 'max'), first_purchase=('date', 'min'))
            pieces.append(grouped)
    # Transaction IDs are deduplicated across chunk boundaries as well as within chunks.
    agg = pd.concat(pieces).groupby(level=0).agg(
        txn_count=('txn_count', 'sum'), total_spend=('total_spend', 'sum'),
        last_purchase=('last_purchase', 'max'), first_purchase=('first_purchase', 'min'))
    cutoff = latest.normalize() + pd.Timedelta(days=1)
    agg['recency_days'] = (cutoff - agg.last_purchase).dt.total_seconds() / 86400
    agg['history_days'] = (agg.last_purchase - agg.first_purchase).dt.total_seconds() / 86400
    agg['avg_ticket'] = agg.total_spend / agg.txn_count
    rfm = ['txn_count', 'total_spend', 'recency_days', 'history_days', 'avg_ticket']
    df = df.merge(agg[rfm], left_on='client_id', right_index=True, how='left', validate='one_to_one')
    df[['txn_count', 'total_spend']] = df[['txn_count', 'total_spend']].fillna(0)
    return df.reset_index(drop=True), base, base + rfm, {'history_reference': str(cutoff), 'unique_transactions': len(seen)}


def criteo(root, n=120_000, seed=42, chunksize=400_000):
    """Uniform priority reservoir within treatment arms; scan ALL rows, never a prefix."""
    path = Path(root) / 'criteo-research-uplift-v2.1.csv.gz'
    rng = np.random.default_rng(seed)
    pools = {0: None, 1: None}
    counts = np.zeros(2, dtype=int)
    features = [f'f{i}' for i in range(12)]
    cols = features + ['treatment', 'visit', 'conversion']
    dtype = {x: 'float32' for x in features}
    dtype.update({x: 'int8' for x in cols if x not in features})
    offset = 0
    with pd.read_csv(path, usecols=cols, dtype=dtype, chunksize=chunksize) as chunks:
        for chunk in chunks:
            chunk['row_id'] = np.arange(offset, offset + len(chunk))
            offset += len(chunk)
            chunk['_priority'] = rng.random(len(chunk))
            for arm in (0, 1):
                part = chunk.loc[chunk.treatment.eq(arm)]
                counts[arm] += len(part)
                pool = part if pools[arm] is None else pd.concat([pools[arm], part])
                pools[arm] = pool.nsmallest(n, '_priority')
    # Final allocation uses the actual full-file treatment ratio, not rounded .85.
    n1 = int(round(n * counts[1] / counts.sum()))
    sampled = pd.concat([pools[0].nsmallest(n-n1, '_priority'), pools[1].nsmallest(n1, '_priority')])
    sampled = sampled.drop(columns='_priority').sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled, features, {'population_counts': counts.tolist(), 'population_treatment_rate': float(counts[1]/counts.sum())}
