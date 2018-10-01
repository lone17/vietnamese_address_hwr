from models import *
from utils import *

model = crnn(training=False)

def predict(img_dir, weights_dir):
    X = []
    model.load_weights(weights_dir)
    img = preprocess(img_dir)
    X.append(img)
    X = np.array(X)
    pred = model.predict(X, batch_size=1)

    return ctc_decoder(pred)

dir = '0825_DataSamples 1'
f = open(os.path.join(dir, 'labels.json'), encoding='utf-8')
files = json.load(f).keys()
f.close()
for img in files:
	pred = predict(os.path.join(dir, img), 'checkpoints/weights-24-72.64')
	print(img, '= \'' + pred + '\'')
