"""Search OpenML for obscure datasets."""
from sklearn.datasets import fetch_openml

for did in [43572, 43573, 43574, 43575, 43576, 44089, 44090, 44091, 44092, 44093, 44094, 45060, 45061, 45062, 45063, 45064, 45065]:
    try:
        d = fetch_openml(data_id=did, as_frame=True, parser='auto')
        if 500 < d.data.shape[0] < 50000 and d.data.shape[1] > 5:
            tn = d.target_names if hasattr(d, 'target_names') else '?'
            desc = d.DESCR[:100] if d.DESCR else 'no desc'
            print(f"did={did}: shape={d.data.shape}, target={tn}, desc={desc}")
    except Exception as e:
        pass
