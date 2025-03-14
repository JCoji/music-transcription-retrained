# Master's Thesis Project: Audio Signal to Sheet Music/Tab Conversion

This project focuses on converting audio signals into sheet music/tab notation using the **Guitar Set** dataset.

## Requirements

- Python 3.11
- Libraries listed in `requirements.txt`

## Downloading Guitar Set ~ 12 min

The **Guitar Set** dataset is not included in the repository due to its size. To download it, follow these steps:

1. Navigate to the `src` folder:

```bash
cd src
```

2. Run the download_data.py script:

```bash
python download_data.py
```

The script will download and extract the dataset into the data/guitar_set folder.

3. Clean temp files

```bash
rm ../data/guitar_set/main_guitar_set.zip

rm -r ../data/guitar_set/temp
```