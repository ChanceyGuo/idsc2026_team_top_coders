data

This repository does not store the raw Brugada-HUCA ECG files.

Reason

The project uses the official PhysioNet dataset, and the recommended way to access it is through the official source or the Kaggle dataset prepared for this project.
The raw ECG files are large and not suitable for GitHub storage.
Keeping the raw data outside GitHub also makes the repository cleaner and easier to reproduce.

Recommended Kaggle dataset root:
 /kaggle/input/datasets/chanceyguo/idsc2026-brugada-huca-raw

Expected structure:
metadata.csv
metadata_dictionary.csv
files/files/<patient_id>/<patient_id>.hea
files/files/<patient_id>/<patient_id>.dat

Label usage

This project uses binary mapping:
brugada = 0 means Normal
brugada > 0 means Brugada

Official dataset page:
https://physionet.org/content/brugada-huca/1.0.0/

Required citation

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
