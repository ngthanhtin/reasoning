#%%
import pandas as pd

NABIRD_CLASS_LABEL_FILE = 'NABIRD_no_annotation_class_name.txt'
CUB_CLASS_LABEL_FILE =  'cub_classes.txt'

#%%
# read cub class name
def split_id_name(s):
    try:
        return s.split('.')[1]
    except Exception as e:
        return None
    
cub_table = pd.read_table(f'{CUB_CLASS_LABEL_FILE}', sep=' ',
                            header=None)
cub_table.columns = ['id', 'name']
cub_table['name'] = cub_table['name'].apply(split_id_name)

cub_table
# %%
cub_classes = cub_table.name.values.tolist()
cub_classes
# %%
# read nabird (no annotation) classes
nabird_table = pd.read_table(f'{NABIRD_CLASS_LABEL_FILE}',
                            header=None)
nabird_table.columns = ['name']
nabird_table

# %%
nabird_classes = nabird_table.name.values.tolist()
nabird_classes

# %%
fix_cub_classes = [cls.replace("_", " ") for cls in cub_classes]
fix_cub_classes = [cls.replace("-", " ") for cls in fix_cub_classes]
fix_nabird_classes = [cls.replace("_", " ") for cls in nabird_classes]
fix_nabird_classes = [cls.replace("-", " ") for cls in fix_nabird_classes]
fix_nabird_classes = [cls.replace("'s", "") for cls in fix_nabird_classes]
# %%
# merge classes after fixing mismatching typos
merged_classes = fix_cub_classes + fix_nabird_classes
merged_classes
len(merged_classes)
# %%
print("Number of overlapped classes: ",len(merged_classes)-len(set(merged_classes)))
print(len(set(merged_classes)))

#%%
overlapped_cub_list = []
overlapped_nabirds_list = []
for fix_nabird_cls, orig_nabird_cls in zip(fix_nabird_classes, nabird_classes):
    for fix_cub_cls, orig_cub_cls in zip(fix_cub_classes, cub_classes):
        if fix_nabird_cls == fix_cub_cls:
            overlapped_cub_list.append(orig_cub_cls)
            overlapped_nabirds_list.append(orig_nabird_cls)
print(len(overlapped_cub_list), len(overlapped_nabirds_list))

df = pd.DataFrame(list(zip(overlapped_cub_list, overlapped_nabirds_list)),
               columns =['CUB Classes', 'NABirds Classes'])
df.to_csv('overlapped_df.csv')

# %%
no_overlapped_classes = list(set(merged_classes))
with open("merged_nabird_cub.txt", 'w') as f:
    for cls in no_overlapped_classes:
        f.write(f'{cls}\n')

# %%
