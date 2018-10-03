from models import *
from utils import *

model_name = 'model0'

def predict(img_dir, weights_dir):
    model = models[model_name](training=False)
    model.load_weights(weights_dir)
    X = []
    img = preprocess(img_dir, padding=True)
    X.append(img)
    X = np.array(X)
    pred = model.predict(X, batch_size=1)

    return ctc_decoder(pred)

dir = '0825_DataSamples 1'
# dir = '0916_Data Samples 2/'
f = open(os.path.join(dir, 'labels.json'), encoding='utf-8')
files = json.load(f).keys()
f.close()
for img in list(files)[:100]:
	pred = predict(os.path.join(dir, img), 'checkpoints/40.29.' + model_name)
	print(img, '= \'' + pred + '\'')
