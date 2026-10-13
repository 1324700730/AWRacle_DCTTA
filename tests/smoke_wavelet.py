"""Check the copied DCTTA rec2 wavelet asset and DIR coefficient shape."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adaptation.losses import WaveletTransform
x=torch.rand(1,64,64,3);layer=WaveletTransform(ROOT/'models/dctta/assets/wavelet.mat');y=layer(x)
result={'status':'PASS' if y.shape==(1,4,32,32,3) and torch.isfinite(y).all() else 'FAIL','input_shape':list(x.shape),'coefficient_shape':list(y.shape),'finite':bool(torch.isfinite(y).all()),'wavelet_asset':str(ROOT/'models/dctta/assets/wavelet.mat')}
(ROOT/'experiments/dctta_component_smoke/wavelet.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
