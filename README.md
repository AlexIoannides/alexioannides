# alexioannides

Python source code for generating my personal website - hosted by GitHub pages at [alexioannides.github.io](http://alexioannides.github.io) - using the [Pelican](https://blog.getpelican.com/) framework for static websites, together with [Flex theme](https://github.com/alexandrevicenzi/flex).

The output of the build process is written to the `output` folder in the root directory, that is **not** version controlled using this repository. Instead, the `output` directory has its own repository at [alexioannides](https://github.com/AlexIoannides/alexioannides), that is necessary for hosting with GitHub pages.

## Development Setup

The package's 3rd party dependencies are described in `requirements.txt`. Create a new virtual environment and install these dependencies as follows,

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Building the Website

To build the website we need to call Pelican,

```bash
pelican
```

## Testing the Site Locally

We recommend setting `RELATIVE_URLS = True` when testing (do not forget to revert this before deploying) and then executing the following,

```bash
pelican --listen output
```

A test version of the website will then be available at `http://localhost:8000`.

## Deploying to GitHub Pages

After testing locally, first of all ensure that `RELATIVE_URLS = False`, rebuilding the website if necessary. Then, make sure that you are still in the `output` directory and remember that this is version controlled by a [different repository](https://github.com/AlexIoannides/alexioannides), that now needs new changes to be committed and pushed to `master` as usual - e.g.,

```bash
git add -A
git commit -m "latest changes to alexioannides.github.io"
git push origin master
```

The [updated website](http://alexioannides.github.io) is usually available within a minute or two.
