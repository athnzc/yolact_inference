# YOLACT inference
Contains the [YOLACT](https://github.com/dbolya/yolact) code, stripped down to include mostly scripts necessary for inference/evaluation.

## Data
The trained weights are uploaded [here](https://drive.google.com/drive/folders/1Y3ABMxa6Ehiq0x05IcRhxCJ2LV9LKjfl?usp=drive_link).
## Quantitative results
Exporting the detected masks and bounding boxes into json files requires exporting your dataset into COCO format first so it can be used with used with the evaluation script. One way to do this is using the VGG annotation tool, which is described in this walkthrough.
### Short walkthrough 
0. Clone the repository with `git clone https://github.com/athnzc/yolact_inference.git`. If this the first time you run the code, you need to build the `conda` environment. The environment has been exported as a YAML file in (`env.yml`). In a terminal, navigate to `yolact_inference` and run `conda env create -f env.yml`. This will create a conda environment named yolact3. Run `conda activate yolact3` to activate it. 
1. Open a terminal, navigate to the extracted frames folder you wish to run the inference on and run `ls $PWD/*.png > filenames.txt`. This will create a file `filenames.txt` with the full paths of the images
2. Download VGG from [here](https://www.robots.ox.ac.uk/~vgg/software/via/). Extract the downloaded zip file and open `via.html` on a browser
3. Click **Project -> Add url or path from text file** and select the file created in step 1
4. The images should have been loaded. Click **Export -> Export annotations (COCO format)**. This will export the dataset in COCO format as a json file so it can be used with the inference script.
5. Open `config.py` in yolact_inference/data and add these lines to create a new dataset:
```
anodos_<video_name> = anodos_eval_set.copy({
    'name': 'anodos_<video_name>',

    'valid_images': '<full path to extracted frames folder>',
    'valid_info': '<full path to dataset COCO file>',
})
```

The dataset name doesn't need to be in this format but I simply include the name of the video file for clarity. 

6. Run the inference script `eval.py`, with the following arguments:
	- `config`: Name of the training configuration. Can be found in `config.py`. The configuration used for this case is `anodos_resnet101_config_2023_06_29_210653`
	- `trained_model`: absolute path of the trained weights file. 
	- `output_coco_json`: To export the results as json files. Doesn't need a value. 
	- `dataset`: name of the dataset defined in `config.py` (anodos_<video_name>)
	- `bbox_det_file`, `mask_det_file`: absolute paths of the files you wish to save the bounding boxes and segmentations. If they don't exist, they will be created, otherwise they will be overwritten.
	- `score_threshold`: Only keep detections with a score >= score_threshold. Optional argument.
	- `top_k`: maximum number of objects that will be detected in every image.  Optional argument, default is 5. 
	- Note that running the script with only `--help` will output even more info on the arguments. 
  - **To run the inference on CPU**: set the `cuda` argument to `False`. 

### Output:
- A bboxfile.json file with info regarding the inferred bounding boxes
- A mask.json file with info regarding the inferred segmentations. Each segmentation is an encoded binary image (a mask) where 1 corresponds to the pixels belonging to the detected object (see [here](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) for more details on the encoding)

## Postprocessing
The `postprocessing` subfolder contains `segmentation_postprocessing.py` for visualizing the quantitative results. The script uses a json file with the encoded segmentations (e.g `mask.json`) and deserializes it into a list. For every list item, it decodes the segmentations to retrieve the binary image, and detects the contours of the image using opencv's `findContours`. It also matches the current image's id with its filename which it retrieves from a file with the dataset exported in COCO format. Every segmentation is saved as an RGB image with the detected contours overlaid on the mask. Finally, it creates the `results.json` file and exports the results. 

### How to run 
1. In `segmentation_postprocessing.py`, edit the variables:
	- `mask_file`: same as mask_det_file
	- `dataset_file`: full path to dataset COCO file	
	- `img_save_folder`: where the visualizations of the masks should be saved (absolute path of the folder, must exist beforehand)
	- `results_save_folder`: where the `results.json` file should be saved (absolute path, must exist beforehand)	
	- all of the above should be in single quotes 
 2. Run from within postprocessing folder as `python segmentation_postprocessing.py`

### Output
- A `results.json` file which is essentially a list of dictionaries with info for every segmentation. The segmentation itself is a tuple of arrays. The size of the tuple equals to how many contours were detected in every mask. Every array in the tuple contains the pixel coordinates of the contour, in the form `x1, y1, x2, y2, ..., xn, yn` 

- A `figures` subfolder with visualizations of the masks and the detected contours drawn on top


