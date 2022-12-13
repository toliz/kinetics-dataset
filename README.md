# Kinetics Datasets Downloader

This repo contains some scripts for dowloading and storing the kinetics datasets in a nice format. The files are downloaded from the [CVDF](http://www.cvdfoundation.org) archive, and the download scripts are adapted from their [repo](https://github.com/cvdfoundation/kinetics-dataset).

## Kinetics 400
You first need to download the archived YouTube videos. The CVDF archive thankfully has them already trimmed to their 10 sec duration, so the total download size is 436 GB. Then use the python script to unzip the downloaded videos and organize them in a `<split>/<class>/<video>` file structure.

```
bash 400/download.sh
python extract.py --split all
python organize.py --split all
rm -rf path/to/kinetics600_targz
```

## Kinetics 600
Similarly, you first need to download the archived YouTube videos. The CVDF archive, additionally to having the videos trimmed to their 10 sec duration, has also zipped the train and val splits according to their classes. Hence, there is no need to organize them as in Kinetics 400. However, for the test subset 

```
bash 400/download.sh
python extract.py --split all
rm -rf path/to/kinetics600_targz
```

## Kinetics 700
Not tested yet.