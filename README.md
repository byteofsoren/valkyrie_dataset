# valkyrie dataset
This dataset is used in the Valkyrie system at UMA.

Directory structure
```
   |-- data/        # contains the datasets
   |-- READ.md      # This file
   |-- match.py     # Runs the sift matcher on the dataset
```

## How to use.
To run the matcher of the datasets just run the following command in the terminal.
``` bash
$ python3 match.py

```

This program will ask you to select the dataset you want to run the feature matching on in an terminal menu.
This application will then create a directory called `results` where the results of the algorithms are stored.
