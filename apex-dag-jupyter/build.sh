#!/bin/bash
set -e 

# Ensure we are in the project root before creating the virtual environment
cd ..

# Create and activate a virtual environment if not already in one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating and activating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
else
    echo "Already in a virtual environment: $VIRTUAL_ENV"
fi

current_version=$(grep '"version"' apex-dag-jupyter/package.json | head -1 | sed -E 's/.*"([0-9]+)\.([0-9]+)\.([0-9]+)".*/\1.\2.\3/')
major=$(echo "$current_version" | cut -d. -f1)
minor=$(echo "$current_version" | cut -d. -f2)
patch=$(echo "$current_version" | cut -d. -f3)

echo "Current version: $major.$minor.$patch"

new_patch=$((patch + 1))
new_version="$major.$minor.$new_patch"

echo "New version: $new_version"
sed -i.bak -E 's/"version": "[0-9]+\.[0-9]+\.[0-9]+"/"version": "'$new_version'"/' apex-dag-jupyter/package.json

# install ApexDAG as module

./.venv/bin/pip install -e .

# install Jupyter extension
cd ./apex-dag-jupyter

# actually build and install the version
jlpm install
jlpm build

../.venv/bin/pip install -e .
