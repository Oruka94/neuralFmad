# neuralFmad
## Description
An Extention implementation of pSp method for face morphing on Feret dataset. 
## Installation
Clone `pixe2style2pixel` from https://github.com/eladrich/pixel2style2pixel. 

Clone this repo and extract to `pixe2style2pixel` folder:

`git clone https://github.com/Oruka94/neuralFmad.git`

## Pretrained Models
Download `pre-trained`pSp model following their instruction.

Download `checkpoint`of ArcFace following instruction of ArcFace  
## Prepare data
Run `align_faces.py`to align data. This creates 2 folders `fereta_aligned` and `feretb_aligned`, which include different images of the same people. Move them to `./data`
## Testing morphing and demorphing
- To gather comprehensive statistics for the entire dataset, use `p2s2p_feret_batch.py`. The morphing process runs on `fereta`, and demorphing is performed on the `morphed data` and `feretb`. By default, the resulting morphed images will be stored in `./data/morphed_fa`, and demorphed images will be saved in `./data/demorphed_fb`.

Additionally, a table containing the ArcFace cosine similarity scores will be saved to a `statistic_results.csv` file.

- For a quick single example test, execute `p2s2p_feret_single.py`. 

`python p2s2p_feret_single.py label_of_person number_accomplices_ list_alpha_for_morphing_and_demorphing `

**Example**: `python p2s2p_feret_single.py 00071 5 0.25,0.5,0.75`

This task will select and morph a specified number of accomplices with the highest ArcFace cosine similarity to the malicious individual.
