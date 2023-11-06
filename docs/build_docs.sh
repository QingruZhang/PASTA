# Requires that you have run `pip install pdoc3`
pkg_name="pastalib"
cd "../$pkg_name"
pdoc --html . --output-dir ../docs --template-dir ../docs
cp -rf ../docs/$pkg_name/* ../docs/
rm -rf ../docs/$pkg_name
cd ../docs
python3 style_docs.py