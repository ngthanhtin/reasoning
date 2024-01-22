# %%
import wikipedia as wiki
import pandas as pd
import json

desc_path = 'descriptors/inaturalist2021/chatgpt_descriptors_inaturalist.json'
f = open(desc_path, 'r')
data = json.load(f)
sci_names = list(data.keys())
sci_names[:10]
# %%

comm_names = []
for sci_name in sci_names:
    search_out=wiki.search(sci_name,results=1)
    comm_names.append(search_out[0])

comm_names[:10]
# %%
sci2comm = dict(zip(sci_names, comm_names))
# json_object = json.dumps(sci2comm, indent = 4) 
with open("sci2comm_inat.json", "w") as outfile:
    json.dump(sci2comm, outfile)

# %%
