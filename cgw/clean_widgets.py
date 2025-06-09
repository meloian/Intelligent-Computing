import nbformat

nb = nbformat.read("ic_cgw.ipynb", as_version=4)
# Прибираємо глобальний metadata.widgets
nb.metadata.pop("widgets", None)
# Прибираємо metadata.widgets у кожній клітинці
for cell in nb.cells:
    cell.metadata.pop("widgets", None)
nbformat.write(nb, "ic_cgw.ipynb")