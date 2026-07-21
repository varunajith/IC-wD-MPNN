import pickle

with open("/Users/varun/Documents/polymer-chemprop/datasets/vipea/cv9-monomer_split.pkl", "rb") as f:
    folds = pickle.load(f)

print(type(folds))
print(len(folds))
print(type(folds[0]))
print(folds[0].keys())