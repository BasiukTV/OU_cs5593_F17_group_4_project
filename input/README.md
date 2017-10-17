# Input Data/Files Directory

## Directory Structure
- **data/** -> For input data.
  - **data/raw/** -> For raw data
  - **data/preproc/** -> For preprocessed data
- **models/** -> For serialized data mining models

## Expected File Naming Formats
- **data/raw/YYYY-MM-DD-HH[-LABEL].json.gz** -> For raw archived data.
- **data/raw/YYYY-MM-DD-HH[-LABEL].json** -> For raw unarchived data.

## Legend
- **YYYY** -> Year in 4 digit format. "2017" for example.
- **MM** -> Month in 4 digit fomat. "01" is for January.
- **DD** -> Day in 2 digit fomat. "02" is for second day of the month.
- **HH** -> Hour in 4 digit military fomat. "02" for 2 AM, "15" for 3PM.
- **LABEL** -> For filtering (discovery) of input files and labeling of output files.