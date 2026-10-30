import requests

def download_kather():
    base = "https://zenodo.org/records/1214456/files/"
    files = [
        "NCT-CRC-HE-100K.zip",
        "NCT-CRC-HE-100K-NONORM.zip",
        "CRC-VAL-HE-7K.zip",
    ]

    for fname in files:
        url = base + fname + "?download=1"
        print("Downloading", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

def download_kather_5000():
    base = "https://zenodo.org/records/53169/files/"
    filename = "Kather_texture_2016_image_tiles_5000.zip"

    url = base + filename + "?download=1"
    print("Downloading", url)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def download_prostate():
    base = "https://zenodo.org/records/4789576/files/"

    files = [
        "01_case_western_native.tar",
        "02_training_native.tar",
        "03_wns_leica_native.tar",
        "04_wns_hama_native.tar"
    ]

    for fname in files:
        url = base + fname + "?download=1"
        print("Downloading", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

if __name__  == "__main__":
    download_prostate()
