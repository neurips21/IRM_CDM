"""
Utility functions for SST-2
"""

import csv
import os.path
import numpy as np
import cv2
import base64


def read_raw_data(path, input_cols, label_col, label_map=None):
    """Read columns from a raw tsv file."""
    with open(path) as f:
        headers = next(f).strip().split("\t")
        inputs, labels = [], []
        for line in f:
            items = line.strip().split("\t")
            inp = tuple(items[headers.index(input_col)]
                        for input_col in input_cols)
            label = items[headers.index(label_col)]
            if label_map is not None:
                label = label_map[label]
            inputs.append(inp)
            labels.append(label)
    return inputs, labels


def write_processed_image_data(outputs, destdir, sep="\t"):
    """Write processed data (one tsv per env)."""
    
    os.makedirs(destdir, exist_ok=True)
    for name, (inputs, labels) in outputs.items():
        fname = os.path.join(destdir, f"{name}.tsv")
        # with open(fname, "w", encoding="utf-8") as f:
        with open(fname, "w") as f:
            writer = csv.writer(f, delimiter="\t", quotechar=None)
            writer.writerow(["image", "label"])
            for inp, label in zip(inputs, labels):
                inp = np.swapaxes(inp,0,2)
                inp = np.swapaxes(inp,0,1)
                inp = inp.astype(np.uint8)
                img_bin = cv2.imencode('.png',inp)[1]
                img_encoded_str = base64.b64encode(img_bin)
                if type(img_encoded_str) == bytes:
                    img_encoded_str = img_encoded_str.decode('utf-8')
                value = [img_encoded_str,label]
                v = '{0}\n'.format(sep.join(map(str, value)))
                # writer.writerow(list(v) + [label])
                f.write(v)
        print("| wrote {} lines to {}".format(
            len(inputs), os.path.join(destdir, name))
        )


def read_processed_image_data(fname):
    """Read processed data as a list of dictionaries.

    Reads from TSV lines with the following header line:
    sentence    label
    """
    examples = []
    with open(fname) as f:
    # with open(fname, encoding="utf-8-sig") as f:
        for (i, line) in enumerate(f):
            if i == 0:
                continue
            img_encoded_str, label = line.split("\t")
            img_encoded_str = img_encoded_str.encode('utf-8')
            bytestring = base64.b64decode(img_encoded_str) 
            arr = np.frombuffer(bytestring, dtype=np.uint8)
            img = cv2.imdecode(arr,cv2.IMREAD_COLOR)
            examples.append({'image': img, 'label': label})
    return examples

