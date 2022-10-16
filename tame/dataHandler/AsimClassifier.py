import numpy as np
import pandas as pd
import torch
from console_progressbar import ProgressBar
from dataHandler import PreProcessor
from dataHandler.logger import logger
from dataHandler.datasets import AsimDataSet
from torch.utils.data import DataLoader


class AsimClassifier(PreProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def classify_images(self, file_list: [], **kwargs):
        batch_size = kwargs.get("batch_size", 64)
        index_list = np.arange(0, len(file_list))
        data = AsimDataSet(file_list, index_list)
        model, device = self.set_model_and_device()
        dl = DataLoader(data, shuffle=False, batch_size=batch_size, collate_fn=self.my_collate)
        feat_frames = []
        logger.info("Extracting features")
        pb = ProgressBar(total=len(dl), prefix='', suffix='', decimals=3, length=50, fill='=',
                         zfill='>')
        pb.print_progress_bar(0)
        with torch.no_grad():
            for i_batch, sample in enumerate(dl, 0):
                sample_batched = sample[0]
                index = sample[1].numpy()
                input = sample_batched[1].to(device)
                feats = pd.DataFrame(model(input).cpu().numpy(), index=index, columns=self.feat_names)
                feat_frames.append(feats)
                pb.next()
        features = pd.concat(feat_frames)
        del feats, feat_frames
        preds = self.predict_image_proba(features.loc[:, self.feat_names])
        pred_class = np.argmax(preds, axis=1)
        features.loc[:, list(self.classes_6.keys())] = preds
        features.loc[:, "pred"] = pred_class
        features.loc[:, "file"] = file_list
        features = features[["file", "pred"] + list(self.classes_6.keys()) + self.feat_names]
        return features
