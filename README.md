### Receive Files
* Make sure both croc and receive_files.sh are executable (they should already be, but chmod +x them if not)
* Run ./receive_files.sh
### Create Final Datamix
* Requires pyarrow to be installed in a venv!
* python create_datamix.py --size 100 --input-dir /path/to/parquets --output-dir /path/to/data-root-dir/hq_data_100bt (I think the existing data root dir is `/checkpoint/optim/sanaelotfi/data`)
* python create_datamix.py --size 20 --input-dir /path/to/parquets --output-dir /path/to/data-root-dir/hq_data_20bt
