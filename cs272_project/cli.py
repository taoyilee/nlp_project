#  MIT License
#  #
#  Copyright (c) 2020, Michael Tao-Yi Lee
#  #
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  #
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#  #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# MIT License
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#
import sys

import click

from cs272_project.config import Configuration

DEBUG = False
CONFIG = Configuration()


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--config')
def cli(debug, config=None):
    global DEBUG
    DEBUG = debug

    if config is not None:
        global CONFIG
        CONFIG = Configuration(infile=config)
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command()  # @cli, not @click!
@click.option('--dataset', help='Splitted dataset location')
@click.option('--outdir', help='Output directory')
@click.option('--samples-each-category', default=100)
def subsample(dataset, outdir, samples_each_category=100):
    from cs272_project.dataset.subsample import asnq_subsampler, dataset_split
    tsv_file = asnq_subsampler(dataset, outdir, samples_each_category)
    dataset_split(tsv_file, training_ratio=0.8, dev_ratio=0.1)


@cli.command()  # @cli, not @click!
@click.option('--train-tsv', help='TSV formatted training dataset')
@click.option('--dev-tsv', help='TSV formatted development dataset')
@click.option('--model', default="gpt2", help='Model Name',
              type=click.Choice(['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], case_sensitive=False))
@click.option('--batch-size', default=1, help='Batch Size')
def fine_tune(train_tsv=None, dev_tsv=None,  model="gpt2", batch_size=4):
    from cs272_project.api import fine_tune
    fine_tune(CONFIG, model, batch_size=batch_size, train_tsv=train_tsv, dev_tsv=dev_tsv)


@cli.command()  # @cli, not @click!
@click.option('--outfile', type=str, default=None, help='Output .ini file location')
def write_config(outfile=None):
    import os
    outfile = os.getcwd() if outfile is None else outfile
    if os.path.isdir(outfile):
        outfile = os.path.join(outfile, "config.ini")
    else:  # directory not created or an *.ini file is given
        if os.path.splitext(outfile)[1].lower != "ini":  # directory not created
            os.makedirs(outfile, exist_ok=True)
            outfile = os.path.join(outfile, "config.ini")

    click.echo(f'Writing sample ini file to {outfile}')
    CONFIG.write(outfile=outfile)


if __name__ == '__main__':
    sys.exit(cli())
