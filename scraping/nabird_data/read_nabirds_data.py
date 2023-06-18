#%%
import pandas as pd

ROOT_DIR = '/home/tin/reasoning/scraping/nabird_data/'
BOUNDING_BOX_FILE = 'bounding_boxes.txt'
CLASS_FILE = 'nabird_classes.txt'
CLASS_LABEL_FILE = 'image_class_labels.txt'
HIERARCHY_FILE = 'hierarchy.txt'
IMAGE_FILE = 'images.txt'
SIZE_FILE = 'sizes.txt'
TRAIN_TEST_SPLIT_FILE = 'train_test_split.txt'
BIRD_DIR = ROOT_DIR # './scraping/'
# %%
def read_hierarchy(bird_dir):
    """Loads table of class hierarchies. Returns hierarchy table
    parent-child class map, top class levels, and bottom class levels.
    """
    hierarchy = pd.read_table(f'{bird_dir}/{HIERARCHY_FILE}', sep=' ',
                              header=None)
    hierarchy.columns = ['child', 'parent']

    child_graph = {0: []}
    name_level = {0: 0}
    for _, row in hierarchy.iterrows():
        child_graph[row['parent']].append(row['child'])
        child_graph[row['child']] = []
        name_level[row['child']] = name_level[row['parent']] + 1
    
    terminal_levels = set()
    for key, value in name_level.items():
        if not child_graph[key]:
            terminal_levels.add(key)

    parent_map = {row['child']: row['parent'] for _, row in hierarchy.iterrows()}
    return hierarchy, parent_map, set(child_graph[0]), terminal_levels

hierarchy, parent_map, _, terminal_levels = read_hierarchy(BIRD_DIR)
discrete_labels = set(hierarchy.parent.values.tolist())
# hierarchy
# terminal_levels
parent_map


# %%
def read_class_labels(bird_dir, top_levels, parent_map):
    """Loads table of image IDs and labels. Add top level ID to table."""
    def get_class(l):
        return l if l in top_levels else get_class(parent_map[l])

    class_labels = pd.read_table(f'{bird_dir}/{CLASS_LABEL_FILE}', sep=' ',
                                 header=None)
    class_labels.columns = ['image', 'id']
    class_labels['class_id'] = class_labels['id'].apply(get_class)

    return class_labels

class_labels = read_class_labels(BIRD_DIR, terminal_levels, parent_map)
len(set(class_labels.id.values.tolist()))
# %%
def read_classes(bird_dir, terminal_levels):
    """Loads DataFrame with class labels. Returns full class table
    and table containing lowest level classes.
    """
    def make_annotation(s):
        try:
            return s.split('(')[1].split(')')[0]
        except Exception as e:
            return None

    classes = pd.read_table(f'{bird_dir}/{CLASS_FILE}', header=None)
    classes['id'] = classes[0].apply(lambda s: int(s.split(' ')[0]))
    classes['label_name'] = classes[0].apply(lambda s: ' '.join(s.split(' ')[1:]))
    # classes.drop(0, inplace=True, axis=1)
    classes['annotation'] = classes['label_name'].apply(make_annotation)
    classes['name'] = classes['label_name'].apply(lambda s: s.split('(')[0].strip())

    terminal_classes = classes[classes['id'].isin(terminal_levels)]#.reset_index(drop=True)
    return classes, terminal_classes

classes, terminal_classes = read_classes(BIRD_DIR, terminal_levels)
# len(set(classes.name.values.tolist()))
print(len(classes))
print(len(terminal_classes))
print(terminal_classes)
no_annotation_classes = set(terminal_classes.name.values.tolist())
len(no_annotation_classes)
# with open("NABIRD_no_annotation_class_name.txt", 'w') as f:
#     for cls in no_annotation_classes:
#         f.write(f'{cls}\n')
    
# %%
def read_images(bird_dir):
    """Loads image table and converts to DataFrame
    :param bird_dir Directory containing Cornell Metadata.
    :return DataFrame of image file names.
    """
    images = pd.read_table(f'{bird_dir}/{IMAGE_FILE}', sep=' ',
                           header=None)
    images.columns = ['image', 'file']
    return images

#  %%
def read_boxes(bird_dir):
    """Loads DataFrame of bounding box data for each image.
    """
    boxes = pd.read_table(f'{bird_dir}/{BOUNDING_BOX_FILE}', sep=' ',
                          header=None)
    boxes.columns = ['image', 'x', 'y', 'width', 'height']
    return boxes

# %%
def read_meta(bird_dir):
    """Loads all image meta data and performs joins to create train and test DataFrames."""
    hierarcy, parent_map, top_levels, terminal_levels = read_hierarchy(bird_dir=bird_dir)
    class_labels = read_class_labels(bird_dir=bird_dir,
                                     top_levels=top_levels,
                                     parent_map=parent_map)
    classes, terminal_classes = read_classes(bird_dir=bird_dir,
                                             terminal_levels=terminal_levels)

    meta = class_labels.merge(classes).merge(classes.rename(columns={'label_name': 'class_name',
                                                                     'id': 'class_id'})
                                             .drop(columns=['annotation', 'name']))
    name_map = {row['name']: idx for idx, row in meta[['name']].drop_duplicates()
                                                               .reset_index(drop=True)
                                                               .iterrows()}
    terminal_map = {row['label_name']: idx for idx, row in terminal_classes.iterrows()}
    meta['name_id'] = meta['name'].apply(lambda n: name_map[n])
    meta['terminal_id'] = meta['label_name'].apply(lambda n: terminal_map[n])

    images = read_images(bird_dir=bird_dir)
    boxes = read_boxes(bird_dir=bird_dir)
    sizes = read_sizes(bird_dir=bird_dir)
    train_test = read_train_test(bird_dir=bird_dir)
    train_test_meta = images.merge(meta).merge(boxes).merge(sizes).merge(train_test) \
        .sample(frac=1).reset_index(drop=True)
    train_meta = train_test_meta[train_test_meta['is_train'] == 1].drop(columns='is_train').reset_index(drop=True)
    test_meta = train_test_meta[train_test_meta['is_train'] == 0].drop(columns='is_train').reset_index(drop=True)
    return train_meta, test_meta