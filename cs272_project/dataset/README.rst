================================
Notes on subsampling
================================

ASNQ
====================
A manual preprocessing of the source file is required. With the access to the orignal ASNQ dataset, use following shell commands
to break raw train.tsv (3.2GB) into smaller tsv files with 500K lines in each.

.. code-block:: bash

    split -l 500000 ../train.tsv
    for i in *; do mv "$i" "$i.tsv"; done

