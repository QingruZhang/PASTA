# Read in the file
with open('index.html', 'r') as f:
    data = f.read()

# Fix a url
data = data.replace('&lt;https://github.com/csinva/imodels</code>&gt;',
                    'https://github.com/csinva/imodels</code>')
data = data.replace('&lt;https://doi.org/10.5281/zenodo.4026887}&gt;',
                    'https://doi.org/10.5281/zenodo.4026887}')


# Write the file out again
with open('index.html', 'w') as f:
    f.write(data)
