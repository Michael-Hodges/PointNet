mkdir ShapeNet
cd ShapeNet
wget https://shapenet.cs.stanford.edu/iccv17/partseg/train_data.zip
unzip -q train_data.zip
rm train_data.zip
wget https://shapenet.cs.stanford.edu/iccv17/partseg/train_label.zip
unzip -q train_label.zip
rm train_label.zip
wget https://shapenet.cs.stanford.edu/iccv17/partseg/val_data.zip
unzip -q val_data.zip
rm val_data.zip
wget https://shapenet.cs.stanford.edu/iccv17/partseg/val_label.zip
unzip -q val_label.zip
rm val_label.zip
wget https://shapenet.cs.stanford.edu/iccv17/partseg/test_data.zip
unzip -q train_data.zip
rm train_data.zip

cd ..

# Download HDF5 for indoor 3d semantic segmentation (around 1.6GB)
wget https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip
unzip indoor3d_sem_seg_hdf5_data.zip
rm indoor3d_sem_seg_hdf5_data.zip