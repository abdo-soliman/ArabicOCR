# **Arabic OCR**
This repo is an OCR engine for arabic scipts

## Generate Characters DataSet
To create characters dataset use createDataset.sh script and pass the image directory and text directory as parameters like the following:
```shell
./createDataSet.sh -i /path/to/images -t /path/to/texts
```
This will generate a folder with the name dataSet that contains a folder for each arabic letter that contains images of 25*25 pixels image dataset

To change the destination folder simply pass the path for the output directory using -o flag like the following:
the following:
```shell
./createDataSet.sh -i /path/to/images -t /path/to/texts -o /path/to/output
```

**note**: script will automatically generate output directory and letters sub directories if it doesn't exist so doesn't bother creating it. you can default parameter values and explation for them by using -h or --help flag
