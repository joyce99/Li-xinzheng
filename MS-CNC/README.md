
Multi-component Similarity Graphs for Cross-network Node Classification (MS-CNC)

Environment Requirement
===
The code has been tested running under Python 3.6.2. The required packages are as follows:

•	python == 3.6.2

•	tensorflow == 1.13.1

•	numpy == 1.16.2

•	scipy == 1.2.1

•	sklearn == 0.21.1


Datasets
===
input/ contains the 9 datasets used in our paper.

Each ".mat" file stores a network dataset, where

the variable "network" represents an adjacency matrix, 

the variable "attrb" represents a node attribute matrix,

the variable "group" represents a node label matrix. 

Code
===
"model.py" is the implementation of the MS-CNC model.

"test_Blog.py" is an example case of the cross-network node classification task from Blog1 to Blog2 networks.

"test_citation.py" is an example case of the cross-network node classification task from citationv1 to dblpv7 networks.
