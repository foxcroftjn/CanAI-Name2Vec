from create_model import train_model
from gensim.models.doc2vec import Doc2Vec
from os import makedirs, path
from results import save_results
from sys import argv

#ensure output directories exist
makedirs('models', exist_ok=True)
makedirs('histograms', exist_ok=True)
makedirs('matching_name_distance', exist_ok=True)
makedirs('random_name_distance', exist_ok=True)

#verify and load command line parameters
if len(argv) < 4:
    print('Syntax: python main.py [epochs] [vector_size] [window]')
    exit(1)
try:
    parameters = tuple(int(x) for x in argv[1:])
except:
    print('Error: Expected all parameters to be integers. Exiting.')
    exit(1)

#create model if it doesn't already exist
model_path = 'models/epochs_%d_vectorSize_%d_window_%d.model' % parameters
histogram_path = 'histograms/epochs_%d_vectorSize_%d_window_%d.png' % parameters
matching_name_path = 'matching_name_distance/epochs_%d_vectorSize_%d_window_%d.csv' % parameters
random_name_path = 'random_name_distance/epochs_%d_vectorSize_%d_window_%d.csv' % parameters
if path.exists(model_path):
    print(f"'{model_path}' already exits. Using existing model to re-generate results.")
    model = Doc2Vec.load(model_path)
else:
    print('Generating model with epochs=%d vector_size=%d window=%d' % parameters)
    model = train_model(*parameters)
    model.save(model_path)
    print(f'Saved model to {model_path}')

save_results(model, histogram_path, matching_name_path, random_name_path)
print(f'Saved histogram to {histogram_path}')
print(f'Saved histogram to {histogram_path}')
print(f'Saved matching name distances to {matching_name_path}')
print(f'Saved random name distances to {random_name_path}')
