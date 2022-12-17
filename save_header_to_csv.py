import pandas as pd
import numpy as np
from get_cols import get_cols

cols = get_cols(33) # have 33 landmarks
cols = ["class"] + cols
print("cols",["class"]+cols)

df = pd.DataFrame(np.array(cols).reshape(1, 67))
print(df)
df.to_csv('landmarks.csv', mode='a', index=False, header=None) # adds index to csv file