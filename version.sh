#!/bin/bash

LATEST_VERSION=$(grep "v1\." version.md | tail -n 1)
MAJOR_VERSION=$(echo "$LATEST_VERSION" | cut -d '.' -f 1 | tr -d 'v')
MINOR_VERSION=$(echo "$LATEST_VERSION" | cut -d '.' -f 2)
PATCH_VERSION=$(echo "$LATEST_VERSION" | cut -d '.' -f 3 | cut -d ' ' -f 1)

mkdir -p version

cat <<EOL > version/version.h
#ifndef _DEPLOY_CI_CLOUD_VERSION_H__
#define _DEPLOY_CI_CLOUD_VERSION_H__

#define DEPLOY_CI_CLOUD_MAJOR_VERSION $MAJOR_VERSION
#define DEPLOY_CI_CLOUD_MINOR_VERSION $MINOR_VERSION
#define DEPLOY_CI_CLOUD_PATCH_VERSION $PATCH_VERSION

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define DEPLOY_CI_CLOUD_STRING TOSTRING(DEPLOY_CI_CLOUD_MAJOR_VERSION) "." \\
                                    TOSTRING(DEPLOY_CI_CLOUD_MINOR_VERSION) "." \\
                                    TOSTRING(DEPLOY_CI_CLOUD_PATCH_VERSION) " "

#endif
EOL

SCRIPT_VERSION=$(echo "$LATEST_VERSION" | sed 's/ /_/g')
SCRIPT_TAG="v${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}"

# 更新 script/main.py
sed -i "s/LATEST_VERSION=.*/LATEST_VERSION=\"$LATEST_VERSION\"/" dags/dataset_schedulers.py
sed -i 's/LATEST_VERSION="\(.*\) \(.*\)"/LATEST_VERSION="\1_\2"/' dags/dataset_schedulers.py  # V1.0.28 2024-10-10 改为V1.0.28_2024-10-10
# sed -i "s/LABEL_OCC_TAG=.*/LABEL_OCC_TAG=\"v${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}\"/" dags/dataset_schedulers.py

sed -i "s/LATEST_VERSION=.*/LATEST_VERSION=\"$LATEST_VERSION\"/" scripts/deploy_ci_cloud.sh
sed -i 's/LATEST_VERSION="\(.*\) \(.*\)"/LATEST_VERSION="\1_\2"/' scripts/deploy_ci_cloud.sh  # V1.0.28 2024-10-10 改为V1.0.28_2024-10-10   
# sed -i "s/LABEL_OCC_TAG=.*/LABEL_OCC_TAG=\"v${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}\"/" scripts/deploy_ci_cloud.sh

sed -i "s/LATEST_VERSION=.*/LATEST_VERSION=\"$LATEST_VERSION\"/" dags/dataset_schedulers_segment.py
sed -i "s/LATEST_VERSION=.*/LATEST_VERSION=\"$LATEST_VERSION\"/" scripts/deploy_ci_cloud_segment.sh


echo "版本号更新完成:"
echo "Latest version: $LATEST_VERSION"
echo ""
echo "version.h:"
head -n 6 version/version.h
echo ""
