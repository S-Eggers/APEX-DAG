#!/bin/bash
set -e 

current_version=$(grep '"version"' package.json | head -1 | sed -E 's/.*"([0-9]+)\.([0-9]+)\.([0-9]+)".*/\1.\2.\3/')
major=$(echo "$current_version" | cut -d. -f1)
minor=$(echo "$current_version" | cut -d. -f2)
patch=$(echo "$current_version" | cut -d. -f3)

echo "Current version: $major.$minor.$patch"

new_patch=$((patch + 1))
new_version="$major.$minor.$new_patch"

echo "New version: $new_version"
sed -i.bak -E "s/\"version\": \"[0-9]+\.[0-9]+\.[0-9]+\"/\"version\": \"$new_version\"/" package.json


cd ..
# install ApexDAG as module

pip install -e .

# install Jupyter extension
cd ./apex-dag-jupyter

# actually build and install the version
jlpm install
jlpm build

pip install -e .
