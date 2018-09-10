#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Alex Ioannides'
SITEURL = 'https://alexioannides.github.io'
SITENAME = AUTHOR
SITETITLE = AUTHOR
SITESUBTITLE = 'Financial engineer, (data) scientist, PhD in Computational Neuroscience/AI and habitual coder'
SITEDESCRIPION = 'Alex Ioannides on data science: data mining, statistics, machine learning, AI, functional programming, R, Python, Scala, Spark, Elasticsearch, AWS, DevOps...'
SITELOGO = 'images/alex_ioannides.png'
FAVICON = 'images/favicon.ico'
PYGMENTS_STYLE = 'monokai'

ROBOTS = 'index, follow'

THEME = 'themes/Flex'
PATH = 'content'
TIMEZONE = 'Europe/London'

DEFAULT_LANG = 'en'

MAIN_MENU = True
USE_FOLDER_AS_CATEGORY = True
MENUITEMS = (('Archives', '/archives.html'),
             ('Categories', '/categories.html'),
             ('Tags', '/tags.html'),)

SOCIAL = (('github', 'https://github.com/alexioannides'),
          ('linkedin', 'https://www.linkedin.com/in/alexioannides/'),
          ('twitter', 'https://twitter.com/ioannides_alex'),
          ('soundcloud', 'https://soundcloud.com/user-616657739'))

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
# LINKS = (('Pelican', 'http://getpelican.com/'),
#          ('Python.org', 'http://python.org/'),
#          ('Jinja2', 'http://jinja.pocoo.org/'))

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
