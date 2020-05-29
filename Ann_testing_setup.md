

To test the existing MARS classifiers on the original test set, call from the command line:

`python run_training_orig.py --testonly --videos '/groups/Andersonlab/CMS273/' --clf '/home/kennedya/MARS_train_infer_CMS/trained_classifiers/mars_v1_8/top_xgb500_wnd/'`

`--videos` is the path to a directory with training and test videos. Training videos should be in a folder called `TRAIN_ORIG` and test in a folder `TEST_ORIG`: I'll put these up on the lab server shortly.

`--clf` is the path to the folder containing your trained classifiers. For simplicity I've uploaded the attack classifier + the scaler (for pre-processing) to this repository, so you can just use `'./models/'` here. The classifiers I've uploaded are for top (non-pcf) features.

Note, the code is designed to load the non-windowed version of the features- it applies windowing to the features when it is run (**and it tests the windowed version of the classifiers**), and saves these windowed versions in one big npz file for faster loading in the future. I confirmed on a sample video that the windowing code in this repo produces windowed features that are identical to the windowed features produced by MARS.
