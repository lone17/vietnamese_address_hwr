from models import *
from utils import *

model_name = 'model4'
weights_dir = 'checkpoints/22.23.' + model_name
model = models[model_name](training=False)
model.load_weights(weights_dir)
dir = '0825_DataSamples 1'
# dir = '0916_Data Samples 2/'

def predict(img_dir, weights_dir):
    img = preprocess(img_dir, padding=True)
    img = np.expand_dims(img, 0)
    pred = model.predict(img, batch_size=1)

    return ctc_decoder(pred)

f = open(os.path.join(dir, 'labels.json'), encoding='utf-8')
files = json.load(f).keys()
f.close()
for img in list(files)[:100]:
	pred = predict(os.path.join(dir, img), weights_dir)
	print(img, '= \'' + pred + '\'')
