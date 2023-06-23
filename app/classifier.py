from inspect import getmembers, isclass, isfunction

from torchvision.io import read_image
import torchvision.models as tvmodels



def get_func_by_weights_name(weights_name: str):
    name = weights_name.lower().replace("_weights", "")
    fn = getmembers(tvmodels, lambda l: isfunction(l) and l.__name__.lower() == name)[0]
    return fn[1]


def get_all_models_dict():
    weights = getmembers(tvmodels, lambda l: isclass(l) and l.__name__.endswith("_Weights"))
    return {
        w[0].replace("_Weights", ""): (get_func_by_weights_name(w[0]), w[1]) for w in weights
    }


class Classifier:

    def __init__(self, model, weights):
        self.weights = weights.DEFAULT
        self.model = model(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms(antialias=True)

    def classify(self, img_path: str) -> str:
        img = read_image(img_path)
        batch = self.preprocess(img).unsqueeze(0)
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        return self.weights.meta["categories"][class_id]