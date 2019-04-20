def get_surnames():
    names = []
    with open('data/records_surnames_counts_250k.tsv') as f:
        for line in f:
            name = line.split('\t')[0]
            names.append(name)
    return names
