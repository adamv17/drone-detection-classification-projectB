folderPath = '../../datasets/drone-audio-detection-samples/data/';
pds = parquetDatastore(fullfile(folderPath, '*.parquet'));
pds.SelectedVariableNames = ["audio", "label"];

ds_train, label_train = transform(pds, @preprocessParquetBatch);
dataBatch = preview(ds_train)
labelBatch = preview(label_train)