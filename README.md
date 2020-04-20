# PointNet
## Installing Dependencies and downloading dataset:
Requirements are as follows:
- Torch
- numpy
- pptk
**Install requirements as follows:**
```
pip install -r requirements.txt
```

**Download dataset as follows:**
For linux using wget (adjust using curl for mac)
```
bash download.sh
```

**Create file in ShapeNet dataset**
Name file class_labels.txt
Paste the following text into it:
```
Airplane        02691156
Bag             02773838
Cap             02954340
Car             02958343
Chair           03001627
Earphone        03261776
Guitar          03467517
Knife           03624134
Lamp            03636649
Laptop          03642806
Motorbike       03790512
Mug             03797390
Pistol          03948459
Rocket          04099429
Skateboard      04225987
Table           04379243
```

## Running the code
### Training:
No arguments required but shown for completion
```
python model/train.py --dataset <DATASET> --action <CLASSIFY> --path <DATASET_PATH>
```
### Testing:
No arguments required but shown for completion
```
python model/test.py --dataset <DATASET> --action <CLASSIFY> --path <DATASET_PATH> --model <MODEL_TYPE> --load <LOAD_PATH>
```
### Visualize and single item classification:
No arguments required but shown for completion
```
python model/classify.py --path <DATA_POINT_PATH> --model <MODEL_TYPE> --load <LOAD_PATH>
```


