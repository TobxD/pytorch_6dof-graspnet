"""
Creates the train/test splits for acronym data
"""
import argparse
import os
from pprint import pprint
import numpy as np
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset_root_folder',
    required=True,
    type=str,
    help='path to root directory of the acronym grasps')
parser.add_argument(
    '--split_files_folder',
    required=True,
    type=str,
    help='path to root directory of where to put the split files')
parser.add_argument(
    '--ratio_test_cat',
    required=True,
    type=float,
    help='ratio of categories to be completely test')
parser.add_argument(
    '--ratio_test_per_cat',
    required=True,
    type=float,
    help='ratio of objects within a category to test')
parser.add_argument(
    '--seed',
    default=21,
    type=int)

opt = parser.parse_args()
np.random.seed(opt.seed)

def read_grasp_files():
    grasp_files = os.listdir(opt.dataset_root_folder)
    grasp_data = []
    for filename in grasp_files:
        ind = filename.find("_")
        assert(ind != -1)
        cat = filename[:ind]
        other = filename[ind+1:]
        ind = other.find("_")
        object_name = other[:ind]
        grasp_data.append((cat, object_name, filename))
    return list(sorted(grasp_data))

def split_fraction_of_elements(elements, fraction):
    cutoff = int(fraction*len(elements))
    elements = np.array(elements)
    np.random.shuffle(elements)
    return list(sorted(elements[:cutoff])), list(sorted(elements[cutoff:]))


grasp_files = read_grasp_files()
all_categories = list(sorted(set([cat for (cat, _, _) in grasp_files])))
test_categories, train_categories = split_fraction_of_elements(all_categories, opt.ratio_test_cat)

remaining_obj = [obj_id for (cat, obj_id, _) in grasp_files if cat in train_categories]
test_obj, train_obj = split_fraction_of_elements(remaining_obj, opt.ratio_test_per_cat)

train_obj = set(train_obj)
test_obj = set([obj_id for (cat, obj_id, _) in grasp_files if cat in test_categories] + test_obj)

cat_splits = {cat: {"test": [], "train": []} for cat in all_categories}
for (cat, obj_id, filename) in grasp_files:
    split = "train" if obj_id in train_obj else "test"
    cat_splits[cat][split].append(filename)

for cat in cat_splits:
    output_file = os.path.join(opt.split_files_folder, f"{cat}.json")
    with open(output_file, "w") as f:
        json.dump(cat_splits[cat], f)