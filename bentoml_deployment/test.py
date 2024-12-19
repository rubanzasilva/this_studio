import bentoml
import xgboost as xgb
import pandas as pd
from pathlib import Path
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.tabular.all import *

# Load the model by setting the model tag
booster = bentoml.xgboost.load_model("mental_health_v1:5v6ao5f5ecct4dwt")

path = Path('data/')
test_df = pd.read_csv(path/'test.csv',index_col='id')

train_df = pd.read_csv(path/'train.csv',index_col='id')

cont_names,cat_names = cont_cat_split(train_df, dep_var='Depression')
splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))
to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
#to = TabularPandas(train_df, procs=[Categorify,Normalize],
                   cat_names = cat_names,
                   cont_names = cont_names,
                   y_names='Depression',
                   y_block=CategoryBlock(),
                   splits=splits)
dls = to.dataloaders(bs=64)
test_dl = dls.test_dl(test_df)
res = tensor(booster.predict(test_dl.xs))
print(res)

