# **Arabic OCR**
This repo is an OCR engine for arabic scipts

## Generate Characters DataSet
To create characters dataset use createDataset.sh script and pass the image directory using -i or --img-dir flag and text directory using -t or --text-dir flag like the following:
```shell
./createDataSet.sh -i /path/to/images -t /path/to/texts
```
This will generate a folder with the name dataSet that contains a folder for each arabic letter that contains images of 25*25 pixels image dataset. 

To change the destination folder simply pass the path for the output directory using -o or --output-dir flag like the following:
the following:
```shell
./createDataSet.sh -i /path/to/images -t /path/to/texts -o /path/to/output
```

There is two types of output files img [to generate image dataset] & csv [to generate csv dataset] you can specify output file type by using -f or --file-type flag like the following:
the following:
```shell
./createDataSet.sh -f img
```
The file type flag is optional and it's default value is csv

You can specify number of image files to be used from input images by using -l or --limit flag like the following:
the following:
```shell
./createDataSet.sh -l integer_value
```
The limit flag is optional and it's default value is -1  which indicates all files.

**note**: Script will automatically generate output directory and letters sub directories if it doesn't exist so doesn't bother creating it. You can default parameter values and explation for them by using -h or --help flag.

## Run
To run you simply run main.py and pass the images directory, output directory and classifier type:
the following:
```shell
python main.py -i /path/to/images -t /path/to/write/output/text_files -c classifier_type
```

To see default values and parameters explanation use the following command:
```shell
python main.py --help
```
