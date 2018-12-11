#!/bin/bash

# download and unzip catalogs
wget http://alan-meert-website-aws.s3-website-us-east-1.amazonaws.com/fit_catalog/download/meert_et_al_data_tables_v2.tgz
tar -xvzf meert_et_al_data_tables_v2.tgz
mv ./meert_et_al_data_tables_v2/* ./
rm -r meert_et_al_data_tables_v2
