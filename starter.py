# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import msgpack
import glob

from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import os
import numpy as np # linear algebra
import pandas as pd
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def get_image(record):
    """Convert bytes to an image object."""
    return Image.open(BytesIO(record["image"]))

def resize_image(image, target_size=(300, 300)):
    """Resize an image to the target size using PIL."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def main():
    shard_fnames = ["/Users/chasenwamu/Classes/CS230/kaggleSet/shard_0.msg"]
    image_folder = "/Users/chasenwamu/Classes/CS230/kaggleSet/images"

    data = []
    for shard_fname in shard_fnames:
        with open(shard_fname, "rb") as infile:
            unpacker = msgpack.Unpacker(infile, raw=False)
            x = 0
            for record in unpacker:
                print(f"Printing new record {x}")
                x += 1

                image_id = f"image_{x}"
                latitude = record.get("latitude")
                longitude = record.get("longitude")
                image = get_image(record)

                if np.shape(image)[0] < 300 or np.shape(image)[1] < 300:
                    continue


                image = resize_image(image, (300, 300))
                file_path = os.path.join(image_folder, f"{image_id}.jpg")
                image.save(file_path, "JPEG")

                data.append({
                    "image_id": image_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "file_path": file_path
                })

    df = pd.DataFrame(data)
    print(df.head())

    # Save metadata to CSV
    df.to_csv("/Users/chasenwamu/Classes/CS230/kaggleSet/geotagged_images.csv", index=False)

if __name__ == "__main__":
    main()

