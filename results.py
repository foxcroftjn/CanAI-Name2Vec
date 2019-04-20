from common import get_surnames
import json
import matplotlib.pyplot as plt
from numpy import argsort
from os import makedirs, path
from random import choice, shuffle
from scipy.spatial.distance import cosine

def get_record_surname_pairs():
    name_pairs = []
    with open("data/records25k_data.tsv") as f:
        for line in f:
            name_pairs.append((line.split('\t')[0], line.split('\t')[1]))
    return name_pairs

def get_search_surname_pairs():
    name_pairs = []
    with open("data/search12.5k_data.tsv") as f:
        for line in f:
            name_pairs.append((line.split('\t')[0], line.split('\t')[1]))
    return name_pairs

def get_known_matches():
    known_matches = get_search_surname_pairs()
    known_matches += get_record_surname_pairs()
    known_matches += [(b, a) for (a, b) in known_matches]
    known_matches += [(name, name) for name in get_surnames()]
    return set(known_matches)

def get_random_surname_pairs(names, known_matches):
    names_a = names[:25000]
    names_b = names[:25000]
    shuffle(names_b)
    random_surname_pairs = set(zip(names_a, names_b))
    for pair in random_surname_pairs.intersection(known_matches):
        new_pair = pair
        while new_pair in known_matches:
            new_pair = (pair[0], choice(names))
        random_surname_pairs.remove(pair)
        random_surname_pairs.add(new_pair)
    return random_surname_pairs

def names_to_indices(names, name_pairs):
    name_index = {}
    for i in range(len(names)):
        name_index[names[i]] = i
    index_pairs = []
    for name_a, name_b in name_pairs:
        index_a = name_index[name_a]
        index_b = name_index[name_b]
        index_pairs.append((index_a, index_b))
    return index_pairs

def get_index_pairs():
    if path.exists('cache/index_pairs.json'):
        print('Loaded set of matching name pairs and set of random name pairs from index_pairs.json')
        return json.load(open('cache/index_pairs.json'))
    else:
        print('Generating set of matching name pairs and set of random name pairs.')
        makedirs('cache', exist_ok=True)
        names = get_surnames()

        matching_name_pairs = get_record_surname_pairs()
        random_surname_pairs = get_random_surname_pairs(names, get_known_matches())

        #convert names to index values
        matching_index_pairs = names_to_indices(names, matching_name_pairs)
        random_index_pairs = names_to_indices(names, random_surname_pairs)

        #cache sets
        json.dump((matching_index_pairs, random_index_pairs), open('cache/index_pairs.json', 'w'))

        return (matching_index_pairs, random_index_pairs)

def get_cosine_similarity(model, index_pairs):
    cos = []
    for index_a, index_b in index_pairs:
        vector_a = model.docvecs[index_a]
        vector_b = model.docvecs[index_b]
        cos.append(cosine(vector_a, vector_b))
    return cos

def save_results(model, histogram_path, matching_name_path, random_name_path):
    bins=50
    matching_index_pairs, random_index_pairs = get_index_pairs()
    matching_cos = get_cosine_similarity(model, matching_index_pairs)
    random_cos = get_cosine_similarity(model, random_index_pairs)
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(18, 10))
    plt.hist(random_cos, color = "red", bins=bins, label = "Random Surname Cosine Distribution")
    plt.hist(matching_cos, color = "dodgerblue" ,bins=bins, label = "Records Dataset Surname Cosine Distribution")
    plt.xlabel("Cosine Distance")
    plt.ylabel("Occurrences")
    plt.legend()
    plt.savefig(histogram_path)
    names = get_surnames()
    with open(matching_name_path, 'w') as f:
        for index in argsort(matching_cos):
            pair = matching_index_pairs[index]
            f.write(f'{names[pair[0]]}, {names[pair[1]]}, {matching_cos[index]}\n')
    with open(random_name_path, 'w') as f:
        for index in argsort(random_cos):
            pair = random_index_pairs[index]
            f.write(f'{names[pair[0]]}, {names[pair[1]]}, {random_cos[index]}\n')
