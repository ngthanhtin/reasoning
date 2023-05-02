#%%
import os

f = open("nabird_classes.txt", 'r')
lines = f.readlines()

for line in lines:
    if '\n' in line:
        line = line[:-1]
    id, name = line.split(' ', 1)
    print(id, name)
# %%
